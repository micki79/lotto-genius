# 🍀 LottoGenius - Vollständiges Multi-KI System

[![🍀 LottoGenius Multi-KI Analyse](https://github.com/micki79/lotto-genius/actions/workflows/daily-analysis.yml/badge.svg)](https://github.com/micki79/lotto-genius/actions/workflows/daily-analysis.yml)

**Intelligente Lotto-Vorhersagen für 6 aus 49 mit 7 KI-Systemen und kontinuierlichem Lernen**

🌐 **Live-App:** [https://micki79.github.io/lotto-genius/](https://micki79.github.io/lotto-genius/)

---

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

---

## 📊 Lokale ML-Algorithmen

- 🧠 **Neuronales Netz** (simuliert) - Hot-Cold Balance
- 📈 **LSTM Sequenz-Analyse** - Muster-Erkennung
- 🌲 **Random Forest** - Ensemble-Methode
- 📊 **Bayesian Inference** - Wahrscheinlichkeitsmaximierung
- 🎲 **Monte-Carlo Simulation** - 1000 Durchläufe
- 🏆 **Ensemble** (kombiniert alle)

---

## ⭐ Superzahl-Analyse (6-Faktoren-Algorithmus)

Die Superzahl wird mit 6 verschiedenen Faktoren analysiert:

| Faktor | Gewichtung | Beschreibung |
|--------|------------|--------------|
| 📊 Häufigkeit | 20% | Wie oft wurde jede Superzahl gezogen? |
| 📈 Trend | 25% | Ist sie aktuell "heiß" oder "kalt"? |
| 📅 Wochentag | 15% | Unterschiede Mittwoch vs. Samstag |
| ⏰ Lücke | 20% | Wie lange nicht gezogen (überfällig)? |
| 🔗 Folge-Muster | 15% | Welche Superzahl kommt nach welcher? |
| 🔄 Anti-Serie | 5% | Vermeidet direkte Wiederholungen |

---

## 🔄 Automatische Updates

Der GitHub Actions Workflow läuft automatisch:

| Zeitpunkt | Beschreibung |
|-----------|--------------|
| **Mittwoch 20:00 UTC** | Nach der Ziehung (18:25) |
| **Samstag 21:00 UTC** | Nach der Ziehung (19:25) |
| **Sonntag 03:00 UTC** | Wöchentliche Optimierung |

**Was passiert automatisch:**
1. 📥 Aktuelle Lotto-Daten werden geholt
2. 📊 KI-Analyse wird durchgeführt
3. 🧠 System lernt aus vorherigen Vorhersagen
4. 🔮 Neue Multi-KI Vorhersagen werden generiert
5. 💾 Alles wird automatisch gespeichert

---

## 🧠 Kontinuierliches Lernen

Das System lernt nach jeder Ziehung:

- **Treffer-Analyse:** Wie viele Zahlen waren richtig?
- **Superzahl-Tracking:** Welche Methode trifft die Superzahl am besten?
- **Provider-Ranking:** Welche KI liefert die besten Ergebnisse?
- **Methoden-Optimierung:** 3+ Treffer Quote wird getrackt

---

## 📁 Repository-Struktur

```
lotto-genius/
├── .github/
│   └── workflows/
│       └── daily-analysis.yml    # Automatisierung
├── data/
│   ├── predictions.json          # Aktuelle Vorhersagen
│   ├── learning.json             # Lern-Historie
│   ├── lotto_data.json           # Historische Ziehungen
│   ├── analysis.json             # Statistische Analyse
│   ├── provider_scores.json      # KI-Rankings
│   └── superzahl_history.json    # Superzahl-Erfolge
├── scripts/
│   ├── fetch_data.py             # Daten holen
│   ├── analyze.py                # Statistische Analyse
│   ├── learn.py                  # Kontinuierliches Lernen
│   └── predict.py                # Multi-KI Vorhersagen
├── index.html                    # Haupt-App (PWA)
├── manifest.json                 # PWA Manifest
├── sw.js                         # Service Worker
├── icon-*.png                    # App Icons
└── README.md                     # Diese Datei
```

---

## 🔑 API-Keys einrichten (Optional)

Ohne API-Keys funktionieren die **6 lokalen ML-Modelle** automatisch!

Für externe KIs: **Settings → Secrets and variables → Actions → New repository secret**

| Secret Name | Wo bekommst du den Key? |
|-------------|------------------------|
| `GEMINI_API_KEY` | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| `GROQ_API_KEY` | [console.groq.com/keys](https://console.groq.com/keys) |
| `HUGGINGFACE_API_KEY` | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |
| `OPENROUTER_API_KEY` | [openrouter.ai/keys](https://openrouter.ai/keys) |
| `TOGETHER_API_KEY` | [api.together.xyz/settings/api-keys](https://api.together.xyz/settings/api-keys) |
| `DEEPSEEK_API_KEY` | [platform.deepseek.com/api_keys](https://platform.deepseek.com/api_keys) |

---

## 📱 App-Features

- ✅ **PWA** - Installierbar auf Handy & Desktop
- ✅ **Offline-fähig** - Funktioniert ohne Internet
- ✅ **IndexedDB** - Daten werden lokal gespeichert
- ✅ **Responsive** - Optimiert für alle Bildschirmgrößen
- ✅ **Dark Mode** - Augenschonendes Design
- ✅ **Auto-Update** - Lädt neue Vorhersagen automatisch

---

## 🚀 Installation

### Option 1: Als Web-App nutzen
Einfach öffnen: [https://micki79.github.io/lotto-genius/](https://micki79.github.io/lotto-genius/)

### Option 2: Als App installieren
1. Öffne die URL im Browser
2. Klicke auf "Zum Startbildschirm hinzufügen" oder "App installieren"

### Option 3: Eigenes Repository
1. Fork dieses Repository
2. Aktiviere GitHub Pages (Settings → Pages → main branch)
3. Füge optional API-Keys als Secrets hinzu
4. Workflow manuell starten (Actions → Run workflow)

---

## 📊 Datenquellen

- **Historische Daten:** [johannesfriedrich.github.io](https://johannesfriedrich.github.io/LottoNumberArchive/Lotto_6gus49_json.json)
- **Aktuelle Ziehungen:** Automatisch nach Mi/Sa Ziehungen

---

## ⚠️ Disclaimer

Dieses System dient **nur zu Unterhaltungszwecken**! 

Lotto ist ein Glücksspiel. Keine KI kann garantierte Gewinne vorhersagen. Spiele verantwortungsvoll und setze nur Geld ein, das du bereit bist zu verlieren.

---

## 📜 Lizenz

MIT License - Freie Nutzung für alle!

---

## 👨‍💻 Entwickelt von

**micki79** mit Hilfe von Claude AI

🍀 **Viel Glück!** 🍀
