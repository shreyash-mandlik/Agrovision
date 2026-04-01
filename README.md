# 🌿 AgroVision v2 — Smart Farming Platform

**Live Demo:** https://shreyash-mandlik.github.io/Agrovision/
**API:** https://ShreyashM25-agrovision-api.hf.space

AI-powered smart farming platform for Indian farmers.

## What it does
- 🔬 Real crop & disease detection — EfficientNetV2B0, 97-98% accuracy, 38 classes
- 💬 Krishi Chat AI — Gemini-powered multilingual farming assistant (6 languages)
- 🌤 Live weather + 7-day forecast via Open-Meteo
- 🌾 Crop recommendation engine (28 crops, NPK/pH/rainfall scoring)
- 🧪 Fertilizer & pesticide lookup (30 crops × growth stage × problem)
- 🪨 Soil health analyzer
- 📋 Government schemes — PM-KISAN, PMFBY, Maharashtra schemes

## Model
| Property | Value |
|---|---|
| Architecture | EfficientNetV2B0 |
| Dataset | PlantVillage — 54,305 images |
| Classes | 38 crop/weed diseases |
| Accuracy | 97-98% validation |
| Training | 43 epochs, dual Tesla T4, ~6.2 hrs |

## Tech Stack
- Frontend: Vanilla HTML/CSS/JS (single file)
- Backend: Flask + TensorFlow, deployed on Hugging Face Spaces (Docker)
- Chat: Gemini 2.0 Flash API
- Weather: Open-Meteo API
