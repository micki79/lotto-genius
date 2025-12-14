# Claude Regeln für dieses Projekt

## Pflicht-Regeln

### 1. Alles ausführen
- IMMER alle Aufgaben aus dem Prompt vollständig ausführen
- Nichts auslassen oder halbfertig lassen
- Bei mehreren Aufgaben: Alle erledigen

### 2. GitHub-Links immer angeben
Nach jeder Änderung (Commit/Push) IMMER den direkten Link angeben:
- Pull Request erstellen: `https://github.com/micki79/lotto-genius/compare/main...[branch-name]`
- App testen: `https://micki79.github.io/lotto-genius/`

### 3. Vollständiger Funktionstest
Nach jeder Code-Änderung IMMER einen vollständigen Funktionstest durchführen:
- HTML-Syntax validieren
- JavaScript-Syntax validieren (mit Node.js)
- Service Worker prüfen
- manifest.json validieren
- Alle Pfade auf Korrektheit prüfen (relativ für GitHub Pages)
- Keine null/undefined Referenzen
- Keine Division durch 0

### 4. Sprache
- Kommunikation auf Deutsch
- Code-Kommentare können auf Englisch sein

### 5. Datendateien aktuell halten
Bei Änderungen an Spielen oder Datenstrukturen IMMER:
- Alle Datendateien im `data/` Ordner auf aktuellen Stand bringen
- fetch_data.py anpassen wenn neue Datenquellen benötigt werden
- Sicherstellen, dass alle Spiele echte historische Daten haben
- Verfügbare Datendateien:
  - `lotto_data.json` - Lotto 6aus49 (3000+ Ziehungen)
  - `eurojackpot_data.json` - Eurojackpot
  - `spiel77_data.json` - Spiel 77
  - `super6_data.json` - Super 6
  - `gluecksspirale_data.json` - Glücksspirale
  - `predictions.json` - KI-Vorhersagen
  - `learning.json` - Lernhistorie

## Projekt-Info
- **App-URL:** https://micki79.github.io/lotto-genius/
- **Repo:** https://github.com/micki79/lotto-genius
- **Typ:** PWA (Progressive Web App)
- **Pfade:** Immer relativ (`./`) für GitHub Pages Kompatibilität
