"""WaZoBia Chatterbox Multilingual TTS Engine for Hugging Face ZeroGPU Spaces.

Features all 3 Sovereign NaijaVoices Models:
  1. samuelolubukun/chatterbox-hausa-naijavoices (Hausa)
  2. samuelolubukun/chatterbox-yoruba-naijavoices (Yorùbá)
  3. samuelolubukun/chatterbox-igbo-naijavoices (Igbo)

Bundled Platform Voices:
  - male_victor (voices/male_victor.wav)
  - female_blessing (voices/female_blessing.wav)

Endpoints:
  - Interactive Gradio Web UI
  - OpenAI-compatible REST API (/v1/audio/speech, /v1/models, /healthz)
  - ZeroGPU compatible via @spaces.GPU
"""

import hmac
import io
import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

import gradio as gr
import gradio_client.utils
import spaces

# Fix Gradio schema bug
_orig_get_type = gradio_client.utils.get_type
def _patched_get_type(schema):
    if isinstance(schema, bool):
        return "boolean"
    return _orig_get_type(schema)
gradio_client.utils.get_type = _patched_get_type

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.security import APIKeyHeader
from huggingface_hub import hf_hub_download
from pydantic import BaseModel, Field
from safetensors.torch import load_file
from starlette.routing import BaseRoute, Match

from chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES

# Enable African sovereign language tags in Chatterbox tokenizer
SUPPORTED_LANGUAGES.update({"ha": "ha", "yo": "yo", "ig": "ig"})

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("Chatterbox-WaZoBia")

# Hausa -> Yoruba -> Igbo ordering
MODELS_CATALOG = {
    "hausa": "samuelolubukun/chatterbox-hausa-naijavoices",
    "yoruba": "samuelolubukun/chatterbox-yoruba-naijavoices",
    "igbo": "samuelolubukun/chatterbox-igbo-naijavoices",
}

LANG_CODE_MAP = {
    "hausa": "ha",
    "yoruba": "yo",
    "igbo": "ig",
}

PLATFORM_VOICES = {
    "male_victor": os.path.join(os.path.dirname(__file__), "voices", "male_victor.wav"),
    "female_blessing": os.path.join(os.path.dirname(__file__), "voices", "female_blessing.wav"),
}

DEFAULT_MODEL_KEY = "hausa"

base_model = None
loaded_state_dicts: Dict[str, Dict[str, Any]] = {}

def get_base_model():
    global base_model
    if base_model is None:
        logger.info("Initializing Base Chatterbox multilingual TTS on CPU...")
        device = "cpu"
        base_model = ChatterboxMultilingualTTS.from_pretrained(device)
    return base_model

def get_model_weights(lang_key: str):
    if lang_key not in MODELS_CATALOG:
        lang_key = DEFAULT_MODEL_KEY
    repo_id = MODELS_CATALOG[lang_key]

    if lang_key not in loaded_state_dicts:
        logger.info(f"Downloading safetensors for Chatterbox {repo_id}")
        model_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
        sd = load_file(model_path)
        loaded_state_dicts[lang_key] = sd
        logger.info(f"Loaded weights for {lang_key}")

    return loaded_state_dicts[lang_key], repo_id

@spaces.GPU(duration=60)
def gpu_synthesize(
    text: str,
    lang_key: str,
    reference_wav_path: str,
    temperature: float = 0.6,
    exaggeration: float = 1.0,
    cfg_weight: float = 1.0,
) -> np.ndarray:
    model = get_base_model()
    sd, repo_id = get_model_weights(lang_key)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Resize embedding & head for the fine-tuned vocab
    old_emb = model.t3.text_emb
    model.t3.text_emb = nn.Embedding(sd["text_emb.weight"].shape[0], old_emb.embedding_dim)
    old_head = model.t3.text_head
    model.t3.text_head = nn.Linear(old_head.in_features, sd["text_head.weight"].shape[0], bias=old_head.bias is not None)

    model.t3.load_state_dict(sd, strict=False)
    model.t3.to(device).eval()

    lang_id = LANG_CODE_MAP.get(lang_key, "ha")
    text_padded = text.strip() + "." if not text.strip().endswith(".") else text.strip()

    with torch.inference_mode():
        out_wav = model.generate(
            text_padded,
            language_id=lang_id,
            audio_prompt_path=reference_wav_path,
            exaggeration=float(exaggeration),
            cfg_weight=float(cfg_weight),
            temperature=float(temperature),
        )

    if isinstance(out_wav, torch.Tensor):
        out_wav = out_wav.detach().cpu().numpy().squeeze()

    model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- POST-PROCESSING: Accurate End-of-Speech Detection & Tail Trimming ---
    trimmed_audio = out_wav
    sr = getattr(model, 'sr', 24000)
    frame_ms = 30
    frame_len = int((frame_ms / 1000.0) * sr)
    if len(trimmed_audio) > frame_len * 5:
        num_frames = len(trimmed_audio) // frame_len
        energies = np.array([
            np.sqrt(np.mean(trimmed_audio[i * frame_len : (i + 1) * frame_len] ** 2))
            for i in range(num_frames)
        ])
        peak_energy = np.percentile(energies, 95)
        
        # 10% peak energy isolates phonemes from vocoder static hiss
        threshold = max(1e-4, peak_energy * 0.10)
        speech_frames = np.where(energies > threshold)[0]
        
        if len(speech_frames) > 0:
            last_speech_frame = speech_frames[-1]
            tail_samples = int(0.25 * sr)
            cutoff_sample = min(len(trimmed_audio), (last_speech_frame + 1) * frame_len + tail_samples)
            if cutoff_sample < len(trimmed_audio):
                trimmed_audio = trimmed_audio[:cutoff_sample].copy()
                fade_len = min(len(trimmed_audio), int(0.06 * sr))
                trimmed_audio[-fade_len:] *= np.linspace(1.0, 0.0, fade_len)

    return trimmed_audio

# ---------------------------------------------------------------------------
# FastAPI Endpoints (OpenAI audio/speech compatible)
# ---------------------------------------------------------------------------
class OpenAISpeechRequest(BaseModel):
    model: str = Field(default=MODELS_CATALOG[DEFAULT_MODEL_KEY])
    input: str
    voice: Optional[str] = Field(default="male_victor")
    response_format: Optional[Literal["wav", "mp3"]] = Field(default="wav")
    speed: Optional[float] = Field(default=1.0)

fastapi_app = FastAPI(
    title="WaZoBia Chatterbox TTS Speech API",
    description="OpenAI-compatible /v1/audio/speech endpoint hosting all 3 Chatterbox models for Hausa, Yoruba, Igbo.",
    version="1.0.0",
)
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_api_key_scheme = APIKeyHeader(name="Authorization", scheme_name="BearerAuth", auto_error=False)

def verify_api_key(auth_header: Optional[str] = Security(_api_key_scheme)):
    expected = os.environ.get("CHATTERBOX_API_KEY")
    if not expected:
        return True
    token = ""
    if auth_header:
        token = auth_header[7:].strip() if auth_header.lower().startswith("bearer ") else auth_header.strip()
    if not token or not hmac.compare_digest(token, expected):
        raise HTTPException(status_code=401, detail="Invalid API key.")
    return True

@fastapi_app.get("/healthz")
def health_check():
    return {
        "status": "healthy",
        "service": "wazobia-chatterbox-tts",
        "models": list(MODELS_CATALOG.values()),
        "platform_voices": list(PLATFORM_VOICES.keys()),
        "time": datetime.utcnow().isoformat(),
    }

@fastapi_app.get("/v1/models")
def list_models(auth=Depends(verify_api_key)):
    return {
        "object": "list",
        "data": [
            {
                "id": repo,
                "object": "model",
                "owned_by": "samuelolubukun",
                "type": "text-to-speech",
            }
            for repo in MODELS_CATALOG.values()
        ],
    }

@fastapi_app.post("/v1/audio/speech")
async def create_speech(req: OpenAISpeechRequest, auth=Depends(verify_api_key)):
    if not req.input.strip():
        raise HTTPException(status_code=400, detail="Input text is required.")

    lang_key = DEFAULT_MODEL_KEY
    for k, v in MODELS_CATALOG.items():
        if req.model.strip().lower() in [k.lower(), v.lower()]:
            lang_key = k
            break

    ref_path = PLATFORM_VOICES.get(req.voice.lower(), PLATFORM_VOICES["male_victor"])

    try:
        wav = await run_in_threadpool(gpu_synthesize, req.input, lang_key, ref_path)
    except Exception as e:
        logger.error(f"Chatterbox synthesis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    buffer = io.BytesIO()
    sf.write(buffer, wav, 24000, format="WAV")
    buffer.seek(0)

    return Response(content=buffer.read(), media_type="audio/wav")

# ---------------------------------------------------------------------------
# Gradio Interface
# ---------------------------------------------------------------------------
def gradio_chatterbox(text_input, model_choice, voice_choice, custom_audio, temp, exag, cfg):
    if not text_input or not text_input.strip():
        return None, "⚠️ Please provide text to synthesize."

    ref_path = custom_audio if custom_audio else PLATFORM_VOICES.get(voice_choice, PLATFORM_VOICES["male_victor"])

    lang_key = DEFAULT_MODEL_KEY
    for k, v in MODELS_CATALOG.items():
        if model_choice in [k, v]:
            lang_key = k
            break

    wav = gpu_synthesize(
        text=text_input,
        lang_key=lang_key,
        reference_wav_path=ref_path,
        temperature=temp,
        exaggeration=exag,
        cfg_weight=cfg,
    )

    out_file = "output_chatterbox.wav"
    sf.write(out_file, wav, 24000)
    info = f"✅ Synthesized using `{MODELS_CATALOG[lang_key]}` ({'Custom Audio' if custom_audio else voice_choice})."
    return out_file, info

with gr.Blocks(title="WaZoBia Chatterbox TTS Engine") as demo:
    gr.Markdown("""# 📻 WaZoBia Chatterbox Voice Synthesis Hub
### Authentic Conversational Speech Synthesis for Hausa, Yorùbá, and Igbo (All 3 Models)
Trained on the **NaijaVoices** dataset with bundled platform voices (**Victor** & **Blessing**) and zero-shot voice cloning.
""")

    with gr.Row():
        with gr.Column(scale=3):
            text_prompt = gr.Textbox(
                label="Text to Synthesize",
                placeholder="Enter text in Hausa, Yoruba, or Igbo...",
                lines=4,
                value="Ina kwana, fatan kowa yana cikin ƙoshin lafiya a yau?",
            )
            synth_btn = gr.Button("🎙️ Generate Speech", variant="primary")
            audio_out = gr.Audio(label="Synthesized Speech", type="filepath")
            status_text = gr.Markdown("")

            gr.Examples(
                examples=[
                    ["Ina kwana, fatan kowa yana cikin ƙoshin lafiya a yau?", MODELS_CATALOG["hausa"], "male_victor"],
                    ["Ẹ n lẹ́ o, báwo ni gbogbo yín ṣe wà lónìí?", MODELS_CATALOG["yoruba"], "female_blessing"],
                    ["Nnọọ, kedu ka ị mere taa?", MODELS_CATALOG["igbo"], "male_victor"],
                ],
                inputs=[text_prompt, gr.Dropdown(choices=list(MODELS_CATALOG.values()), visible=False), gr.Dropdown(choices=["male_victor", "female_blessing"], visible=False)],
            )

        with gr.Column(scale=2):
            model_drop = gr.Dropdown(
                choices=list(MODELS_CATALOG.values()),
                value=MODELS_CATALOG[DEFAULT_MODEL_KEY],
                label="Select Chatterbox Model",
            )
            voice_drop = gr.Dropdown(
                choices=["male_victor", "female_blessing"],
                value="male_victor",
                label="Platform Voice",
            )
            custom_voice = gr.Audio(
                label="Or Upload Custom Reference Voice (3-10s)",
                type="filepath",
                sources=["upload", "microphone"],
            )
            with gr.Accordion("⚙️ Advanced Parameters", open=False):
                temp_slider = gr.Slider(0.1, 1.2, value=0.6, step=0.05, label="Temperature")
                exag_slider = gr.Slider(0.2, 2.0, value=1.0, step=0.1, label="Exaggeration")
                cfg_slider = gr.Slider(0.2, 2.0, value=1.0, step=0.1, label="CFG Weight")

    synth_btn.click(
        gradio_chatterbox,
        inputs=[text_prompt, model_drop, voice_drop, custom_voice, temp_slider, exag_slider, cfg_slider],
        outputs=[audio_out, status_text],
    )

    with gr.Accordion("⚖️ Licenses, Attributions & Ethical Voice Cloning Disclaimer", open=False):
        gr.Markdown("""
### 📜 Dual-License & Dataset Attributions
1. **Base Model Architecture**: ResembleAI Chatterbox (Autoregressive flow-matching T3 architecture).
2. **Fine-Tuning Dataset**: [NaijaVoices Dataset](https://naijavoices.com/) licensed under **[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)**.
3. **Usage Notice**: Intended for academic, educational, cultural preservation, and non-commercial development. Any commercial application must observe NaijaVoices CC BY-NC-SA 4.0 terms.

### 🛡️ Ethical Voice Cloning & Consent Notice
- **Explicit Consent**: You must possess explicit consent from the speaker whose audio reference is uploaded for voice cloning.
- **Strict Prohibition**: Cloning unauthorized voices, creating malicious deepfakes, conducting political deception, or engaging in fraudulent activity is strictly prohibited.
- **Citation**: When publishing work derived from this Space, please acknowledge NaijaVoices (*Emezue et al., 2025*) and author [Samuel Olubukun](https://github.com/samolubukun).
""")

    with gr.Accordion("🔌 OpenAI-Compatible API Usage", open=False):
        gr.Markdown("""
```python
from openai import OpenAI

client = OpenAI(
    base_url="https://<YOUR-SPACE-NAME>.hf.space/v1",
    api_key="your-key",  # If CHATTERBOX_API_KEY is configured
)

res = client.audio.speech.create(
    model="samuelolubukun/chatterbox-hausa-naijavoices",
    voice="male_victor",  # or "female_blessing"
    input="Ina kwana, fatan kowa yana cikin ƙoshin lafiya a yau?",
)
res.stream_to_file("speech.wav")
```
""")

class ApiDelegateRoute(BaseRoute):
    def matches(self, scope):
        if scope["type"] == "http":
            p = scope.get("path", "")
            if p == "/healthz" or p == "/v1" or p.startswith("/v1/"):
                return Match.FULL, {}
        return Match.NONE, {}

    async def handle(self, scope, receive, send):
        await fastapi_app(scope, receive, send)

if __name__ == "__main__":
    port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    demo.queue()
    demo.launch(server_name="0.0.0.0", server_port=port, ssr_mode=False, prevent_thread_lock=True)
    demo.app.router.routes.insert(0, ApiDelegateRoute())
    logger.info("OpenAI API routes (/healthz, /v1/*) attached to Gradio server.")
    demo.block_thread()
