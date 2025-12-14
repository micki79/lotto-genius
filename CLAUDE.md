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

## Projekt-Info
- **App-URL:** https://micki79.github.io/lotto-genius/
- **Repo:** https://github.com/micki79/lotto-genius
- **Typ:** PWA (Progressive Web App)
- **Pfade:** Immer relativ (`./`) für GitHub Pages Kompatibilität
