# 🍀 LottoGenius - Vollständiges Multi-KI System

[![Tägliche KI-Analyse](https://github.com/micki79/lotto-genius/actions/workflows/daily-analysis.yml/badge.svg)](https://github.com/micki79/lotto-genius/actions/workflows/daily-analysis.yml)

## 🤖 7 KI-Systeme integriert

| KI | Beschreibung | Kostenlos |
|----|--------------|-----------|
| 🔮 **Google Gemini** | 1M Tokens/Tag | ✅ |
| ⚡ **Groq** | Ultraschnell | ✅ |
| 🤗 **HuggingFace** | Tausende Modelle | ✅ |
| 🌐 **OpenRouter** | 50+ Modelle | ✅ |
| 🚀 **Together AI** | $25 Startguthaben | ✅ |
| 🧠 **DeepSeek** | Komplett kostenlos | ✅ |
| 🖥️ **Lokale ML** | Immer verfügbar | ✅ |

## 📊 Lokale ML-Algorithmen

- 🧠 Neuronales Netz (simuliert)
- 📈 LSTM Sequenz-Analyse
- 🌲 Random Forest
- 📊 Bayesian Inference
- 🎲 Monte-Carlo Simulation
- 🏆 Ensemble (kombiniert alle)

## 🎯 Superzahl-Analyse (6 Faktoren)

| Faktor | Gewichtung |
|--------|-----------|
| Häufigkeit | 20% |
| Trend | 25% |
| Wochentag | 15% |
| Lücke (überfällig) | 20% |
| Folge-Muster | 15% |
| Anti-Serie | 5% |

## ⚙️ Automatisierung

Das System läuft **vollautomatisch**:

- **Mittwoch 20:00**: Nach Ziehung → Daten holen → Lernen → Neue Tipps
- **Samstag 21:00**: Nach Ziehung → Daten holen → Lernen → Neue Tipps
- **Sonntag 03:00**: Wöchentliche Optimierung

## 🔑 API-Keys einrichten (Optional)

Für externe KI-APIs: **Settings → Secrets → New repository secret**

- `GEMINI_API_KEY` → [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
- `GROQ_API_KEY` → [console.groq.com/keys](https://console.groq.com/keys)
- `HUGGINGFACE_API_KEY` → [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
- `OPENROUTER_API_KEY` → [openrouter.ai/keys](https://openrouter.ai/keys)
- `TOGETHER_API_KEY` → [api.together.xyz/settings/api-keys](https://api.together.xyz/settings/api-keys)
- `DEEPSEEK_API_KEY` → [platform.deepseek.com/api_keys](https://platform.deepseek.com/api_keys)

**Ohne API-Keys funktionieren die lokalen ML-Modelle trotzdem!**

## 📁 Dateien

```
├── index.html          # Die App (157KB)
├── manifest.json       # PWA Manifest
├── sw.js              # Service Worker
├── data/              # Generierte Daten
│   ├── predictions.json
│   ├── learning.json
│   ├── provider_scores.json
│   └── ...
├── scripts/           # Python-Skripte
│   ├── fetch_data.py
│   ├── analyze.py
│   ├── learn.py
│   └── predict.py
└── .github/workflows/ # Automatisierung
    └── daily-analysis.yml
```

## 🚀 Installation

1. Lade alle Dateien auf GitHub hoch
2. Aktiviere GitHub Pages (Settings → Pages → main branch)
3. Optional: Füge API-Keys als Secrets hinzu
4. Starte den Workflow manuell (Actions → Run workflow)

**Deine App:** `https://micki79.github.io/lotto-genius/`

## ⚠️ Hinweis

Lotto ist Glücksspiel. Die KI analysiert Muster, garantiert aber keine Gewinne!

---

🍀 Viel Glück!
