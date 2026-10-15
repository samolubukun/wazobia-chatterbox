---
title: WaZoBia Chatterbox TTS Engine
emoji: 📻
colorFrom: yellow
colorTo: red
sdk: gradio
sdk_version: 5.20.0
app_file: app.py
pinned: false
license: other
short_description: WaZoBia Chatterbox Conversational Speech Synthesis Studio
---

# 📻 WaZoBia Chatterbox Voice Engine: Conversational Speech Synthesis

A production Gradio voice synthesis studio hosting all **3 fine-tuned Chatterbox T3 TTS models** trained specifically on natural, everyday conversational West African speech.

Designed for seamless deployment on **Hugging Face ZeroGPU Spaces** and **local environments**.

---

## 🌟 The 3 Fine-Tuned Models (NaijaVoices)

Chatterbox uses an autoregressive Transformer (T3) paired with an S3Gen flow-matching vocoder, adapted to handle tonal nuances and regional phonemes across Nigeria:

- **Hausa**: [`samuelolubukun/chatterbox-hausa-naijavoices`](https://huggingface.co/samuelolubukun/chatterbox-hausa-naijavoices)
- **Yorùbá**: [`samuelolubukun/chatterbox-yoruba-naijavoices`](https://huggingface.co/samuelolubukun/chatterbox-yoruba-naijavoices)
- **Igbo**: [`samuelolubukun/chatterbox-igbo-naijavoices`](https://huggingface.co/samuelolubukun/chatterbox-igbo-naijavoices)

---

## 🎙️ Voices & Audio Synthesis Features

- **Bundled Platform Voices**:
  - `male_victor`: Authentic Nigerian male speaker reference ([`voices/male_victor.wav`](voices/male_victor.wav)).
  - `female_blessing`: Expressive Nigerian female speaker reference ([`voices/female_blessing.wav`](voices/female_blessing.wav)).
- **Zero-Shot Voice Cloning**: Upload or record a short reference sample (3–10s) to speak in any tone.
- **Adaptive Token Budgeting & Noise Elimination**: Implements length-proportional speech token bounds and calibrated RMS energy offset detection to prevent trailing vocoder noise.

---

## 🚀 Running Locally

```bash
# Clone the repository
git clone https://github.com/samolubukun/wazobia-chatterbox.git
cd wazobia-chatterbox

# Install dependencies
pip install -r requirements.txt

# Launch the Gradio application
python app.py
```

The web application will open at `http://localhost:7860`.

---

## ⚖️ Datasets, Attributions & Citations

1. **Base Architecture**: ResembleAI Chatterbox (Autoregressive T3 + S3Gen vocoder).
2. **Fine-Tuning Dataset**:
   - **NaijaVoices Community** (*Emezue et al., 2025*): Natural conversational African speech corpus licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
3. **Fine-Tuning & Engineering**: [Samuel Olubukun](https://github.com/samolubukun).
