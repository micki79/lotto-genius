#!/usr/bin/env python3
"""
🧠 LottoGenius - EUROJACKPOT Kontinuierliches Lern-System

Lernt aus jeder Ziehung und verbessert das System:
1. Vergleicht Vorhersagen mit echten Zahlen
2. Berechnet Treffer (0-5 Hauptzahlen, 0-2 Eurozahlen)
3. Aktualisiert Strategie-Gewichte (bessere Strategien = höheres Gewicht)
4. Trackt Eurozahlen-Erfolge separat
5. Berechnet Gewinnklassen
6. Speichert alles für langfristiges Lernen

SELBSTLERNSYSTEM:
- Strategien mit mehr Treffern bekommen höhere Gewichte
- Schlechte Strategien werden heruntergestuft
- Das System verbessert sich automatisch
"""
import json
import os
import sys
from datetime import datetime
from collections import Counter

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


# Füge scripts-Verzeichnis zum Pfad hinzu für ML-Modelle Import
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Importiere echte ML-Modelle
try:
    from ml_models import (
        train_eurojackpot_ml,
        EurojackpotEnsembleML
    )
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# Eurojackpot Gewinnklassen
GEWINNKLASSEN = {
    (5, 2): {'klasse': 1, 'name': 'Jackpot (5+2)', 'quote': '10-90 Mio €'},
    (5, 1): {'klasse': 2, 'name': '5+1', 'quote': '200k-800k €'},
    (5, 0): {'klasse': 3, 'name': '5+0', 'quote': '50k-200k €'},
    (4, 2): {'klasse': 4, 'name': '4+2', 'quote': '2k-5k €'},
    (4, 1): {'klasse': 5, 'name': '4+1', 'quote': '100-300 €'},
    (4, 0): {'klasse': 6, 'name': '4+0', 'quote': '50-100 €'},
    (3, 2): {'klasse': 7, 'name': '3+2', 'quote': '40-80 €'},
    (2, 2): {'klasse': 8, 'name': '2+2', 'quote': '15-25 €'},
    (3, 1): {'klasse': 9, 'name': '3+1', 'quote': '12-20 €'},
    (3, 0): {'klasse': 10, 'name': '3+0', 'quote': '10-15 €'},
    (1, 2): {'klasse': 11, 'name': '1+2', 'quote': '8-12 €'},
    (2, 1): {'klasse': 12, 'name': '2+1', 'quote': '6-10 €'},
}

def load_json(filename, default=None):
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return default if default else {}

def save_json(filename, data):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, filename), 'w') as f:
        json.dump(data, f, indent=2)

def get_gewinnklasse(main_matches, euro_matches):
    """Ermittelt die Gewinnklasse basierend auf Treffern"""
    key = (main_matches, euro_matches)
    return GEWINNKLASSEN.get(key, None)

def calculate_new_weight(old_weight, matches, euro_matches, predictions_count):
    """
    Berechnet neues Strategie-Gewicht basierend auf Performance.

    Formel:
    - Basis-Gewicht startet bei 1.0
    - +0.1 pro Hauptzahl-Treffer
    - +0.15 pro Eurozahl-Treffer
    - +0.3 Bonus für Gewinnklasse erreicht
    - Gewicht bewegt sich langsam (exponentieller gleitender Durchschnitt)
    """
    # Sofort-Score für diese Vorhersage
    immediate_score = matches * 0.1 + euro_matches * 0.15

    # Bonus für Gewinnklasse
    if matches >= 3 or (matches >= 1 and euro_matches >= 2) or (matches >= 2 and euro_matches >= 1):
        immediate_score += 0.3

    # Großer Bonus für gute Treffer
    if matches >= 4:
        immediate_score += 0.5
    if matches == 5:
        immediate_score += 1.0

    # Exponentieller gleitender Durchschnitt
    # Alpha = 0.2 bedeutet: 20% neuer Wert, 80% alter Wert
    alpha = 0.2
    new_weight = old_weight * (1 - alpha) + (0.5 + immediate_score) * alpha

    # Begrenzen auf sinnvollen Bereich
    return max(0.1, min(3.0, round(new_weight, 3)))

def learn_from_results():
    """Hauptfunktion: Lernt aus Eurojackpot-Vorhersagen"""

    print("=" * 60)
    print("🧠 LottoGenius - EUROJACKPOT Lern-System")
    print("=" * 60)
    print(f"📅 {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
    print()

    # Lade alle Daten
    ej_data = load_json('eurojackpot_data.json', {'draws': []})
    predictions = load_json('eurojackpot_predictions.json', {'predictions': [], 'history': []})
    learning = load_json('eurojackpot_learning.json', {
        'entries': [],
        'stats': {},
        'by_method': {},
        'by_provider': {},
        'gewinnklassen': {}
    })
    provider_scores = load_json('eurojackpot_provider_scores.json', {})
    strategy_weights = load_json('eurojackpot_strategy_weights.json', {'strategies': {}})
    euro_history = load_json('eurozahl_history.json', {'entries': [], 'stats': {}})

    draws = ej_data.get('draws', [])
    if not draws:
        print("⚠️ Keine Eurojackpot-Daten vorhanden")
        return

    last_draw = draws[0]
    actual_numbers = set(last_draw['numbers'])
    actual_euro = set(last_draw['eurozahlen'])
    draw_date = last_draw['date']

    print(f"📅 Letzte Ziehung: {draw_date}")
    print(f"🎱 Zahlen: {sorted(actual_numbers)} | Eurozahlen: {sorted(actual_euro)}")
    print()

    # Finde unverarbeitete Vorhersagen
    unverified = [p for p in predictions.get('predictions', []) if not p.get('verified')]

    if not unverified:
        print("ℹ️ Keine unverifizierten Vorhersagen vorhanden")
        recent_history = [p for p in predictions.get('history', [])[-50:] if not p.get('verified')]
        if recent_history:
            unverified = recent_history
            print(f"📚 {len(unverified)} Vorhersagen aus Historie gefunden")
        else:
            print("📚 Auch keine unverifizierten in Historie")
            return

    print(f"📊 Verarbeite {len(unverified)} Vorhersagen...")
    print("-" * 50)

    new_entries = []
    method_updates = {}
    provider_updates = {}
    euro_entries = []
    gewinnklassen_count = Counter()

    for pred in unverified:
        pred_numbers = set(pred.get('numbers', []))
        pred_euro = set(pred.get('eurozahlen', []))
        method = pred.get('method', 'unknown')
        provider = pred.get('provider', method)
        strategy_id = pred.get('method', method)

        # Berechne Treffer
        main_matches = len(pred_numbers & actual_numbers)
        euro_matches = len(pred_euro & actual_euro)

        # Ermittle Gewinnklasse
        gewinnklasse = get_gewinnklasse(main_matches, euro_matches)

        # Erstelle Lern-Eintrag
        entry = {
            'id': f"ej_{draw_date}_{method}_{datetime.now().timestamp()}",
            'date': datetime.now().isoformat(),
            'draw_date': draw_date,
            'predicted_numbers': list(pred_numbers),
            'actual_numbers': list(actual_numbers),
            'main_matches': main_matches,
            'predicted_euro': list(pred_euro),
            'actual_euro': list(actual_euro),
            'euro_matches': euro_matches,
            'gewinnklasse': gewinnklasse['klasse'] if gewinnklasse else None,
            'method': method,
            'provider': provider,
            'confidence': pred.get('confidence', 50),
            'strategy': pred.get('strategy', '')
        }
        new_entries.append(entry)

        # Gewinnklassen zählen
        if gewinnklasse:
            gewinnklassen_count[gewinnklasse['klasse']] += 1

        # Eurozahlen-Eintrag
        euro_entries.append({
            'date': draw_date,
            'predicted': list(pred_euro),
            'actual': list(actual_euro),
            'matches': euro_matches,
            'method': method
        })

        # Methoden-Statistik
        if method not in method_updates:
            method_updates[method] = {
                'predictions': 0,
                'total_main_matches': 0,
                'total_euro_matches': 0,
                'three_plus': 0,
                'four_plus': 0,
                'five': 0,
                'gewinnklasse_count': 0
            }
        mu = method_updates[method]
        mu['predictions'] += 1
        mu['total_main_matches'] += main_matches
        mu['total_euro_matches'] += euro_matches
        if main_matches >= 3:
            mu['three_plus'] += 1
        if main_matches >= 4:
            mu['four_plus'] += 1
        if main_matches == 5:
            mu['five'] += 1
        if gewinnklasse:
            mu['gewinnklasse_count'] += 1

        # Provider-Statistik
        if provider not in provider_updates:
            provider_updates[provider] = {
                'predictions': 0,
                'total_main_matches': 0,
                'total_euro_matches': 0
            }
        pu = provider_updates[provider]
        pu['predictions'] += 1
        pu['total_main_matches'] += main_matches
        pu['total_euro_matches'] += euro_matches

        # === STRATEGIE-GEWICHTE AKTUALISIEREN ===
        if strategy_id not in strategy_weights['strategies']:
            strategy_weights['strategies'][strategy_id] = {
                'weight': 1.0,
                'total_predictions': 0,
                'total_main_matches': 0,
                'total_euro_matches': 0,
                'wins_3plus': 0,
                'wins_4plus': 0,
                'wins_5': 0,
                'gewinnklassen': 0
            }

        sw = strategy_weights['strategies'][strategy_id]
        old_weight = sw['weight']

        # Neues Gewicht berechnen
        new_weight = calculate_new_weight(old_weight, main_matches, euro_matches, sw['total_predictions'])

        sw['weight'] = new_weight
        sw['total_predictions'] += 1
        sw['total_main_matches'] += main_matches
        sw['total_euro_matches'] += euro_matches
        if main_matches >= 3:
            sw['wins_3plus'] += 1
        if main_matches >= 4:
            sw['wins_4plus'] += 1
        if main_matches == 5:
            sw['wins_5'] += 1
        if gewinnklasse:
            sw['gewinnklassen'] += 1

        # Markiere als verifiziert
        pred['verified'] = True
        pred['result'] = {
            'main_matches': main_matches,
            'euro_matches': euro_matches,
            'gewinnklasse': gewinnklasse,
            'draw_date': draw_date,
            'new_weight': new_weight
        }

        # Output
        if gewinnklasse:
            match_indicator = f"🎉 KLASSE {gewinnklasse['klasse']}"
        elif main_matches >= 3:
            match_indicator = "🎯"
        elif main_matches >= 2:
            match_indicator = "✓"
        else:
            match_indicator = "•"

        weight_change = "↑" if new_weight > old_weight else ("↓" if new_weight < old_weight else "=")
        print(f"  {match_indicator} {method}: {main_matches}/5 + {euro_matches}/2 | Gewicht: {old_weight:.2f} {weight_change} {new_weight:.2f}")

    print("-" * 50)

    # Speichere neue Lern-Einträge
    learning['entries'].extend(new_entries)
    learning['entries'] = learning['entries'][-2000:]  # Behalte letzte 2000

    # Aktualisiere Methoden-Statistiken
    for method, update in method_updates.items():
        if method not in learning['by_method']:
            learning['by_method'][method] = {
                'total_predictions': 0,
                'total_main_matches': 0,
                'total_euro_matches': 0,
                'three_plus': 0,
                'four_plus': 0,
                'five': 0,
                'gewinnklasse_count': 0
            }

        m = learning['by_method'][method]
        m['total_predictions'] += update['predictions']
        m['total_main_matches'] += update['total_main_matches']
        m['total_euro_matches'] += update['total_euro_matches']
        m['three_plus'] += update['three_plus']
        m['four_plus'] += update['four_plus']
        m['five'] += update['five']
        m['gewinnklasse_count'] += update['gewinnklasse_count']

        # Berechne Accuracy
        if m['total_predictions'] > 0:
            m['main_accuracy'] = round((m['total_main_matches'] / (m['total_predictions'] * 5)) * 100, 2)
            m['euro_accuracy'] = round((m['total_euro_matches'] / (m['total_predictions'] * 2)) * 100, 2)
            m['three_plus_rate'] = round((m['three_plus'] / m['total_predictions']) * 100, 2)

    # Aktualisiere Provider-Scores
    for provider, update in provider_updates.items():
        if provider not in provider_scores:
            provider_scores[provider] = {
                'total_predictions': 0,
                'total_main_matches': 0,
                'total_euro_matches': 0,
                'main_accuracy': 0,
                'euro_accuracy': 0
            }

        p = provider_scores[provider]
        p['total_predictions'] += update['predictions']
        p['total_main_matches'] += update['total_main_matches']
        p['total_euro_matches'] += update['total_euro_matches']

        if p['total_predictions'] > 0:
            p['main_accuracy'] = round((p['total_main_matches'] / (p['total_predictions'] * 5)) * 100, 2)
            p['euro_accuracy'] = round((p['total_euro_matches'] / (p['total_predictions'] * 2)) * 100, 2)

    # Eurozahlen-Historie
    euro_history['entries'].extend(euro_entries)
    euro_history['entries'] = euro_history['entries'][-500:]

    # Eurozahlen-Statistik
    euro_total = len(euro_history['entries'])
    euro_correct = sum(e['matches'] for e in euro_history['entries'])
    euro_history['stats'] = {
        'total': euro_total,
        'total_matches': euro_correct,
        'avg_matches': round(euro_correct / euro_total, 2) if euro_total > 0 else 0,
        'perfect_matches': sum(1 for e in euro_history['entries'] if e['matches'] == 2),
        'last_update': datetime.now().isoformat()
    }

    # Gewinnklassen-Statistik
    for klasse, count in gewinnklassen_count.items():
        klasse_key = str(klasse)
        if klasse_key not in learning.get('gewinnklassen', {}):
            learning['gewinnklassen'][klasse_key] = 0
        learning['gewinnklassen'][klasse_key] += count

    # Gesamt-Statistik
    total_entries = len(learning['entries'])
    if total_entries > 0:
        total_main = sum(e['main_matches'] for e in learning['entries'])
        total_euro = sum(e['euro_matches'] for e in learning['entries'])
        three_plus = sum(1 for e in learning['entries'] if e['main_matches'] >= 3)
        gewinn_count = sum(1 for e in learning['entries'] if e.get('gewinnklasse'))

        learning['stats'] = {
            'total_entries': total_entries,
            'avg_main_matches': round(total_main / total_entries, 2),
            'avg_euro_matches': round(total_euro / total_entries, 2),
            'three_plus_rate': round((three_plus / total_entries) * 100, 2),
            'gewinnklasse_rate': round((gewinn_count / total_entries) * 100, 2),
            'last_update': datetime.now().isoformat(),
            'last_draw': draw_date
        }

    # Strategie-Gewichte speichern
    strategy_weights['last_update'] = datetime.now().isoformat()

    # Speichere alles
    save_json('eurojackpot_predictions.json', predictions)
    save_json('eurojackpot_learning.json', learning)
    save_json('eurojackpot_provider_scores.json', provider_scores)
    save_json('eurojackpot_strategy_weights.json', strategy_weights)
    save_json('eurozahl_history.json', euro_history)

    # Output Zusammenfassung
    print()
    print(f"✅ {len(new_entries)} Vorhersagen gelernt!")
    print()
    print("📊 Gesamt-Statistik:")
    print(f"   • Einträge: {learning['stats'].get('total_entries', 0)}")
    print(f"   • Ø Hauptzahlen: {learning['stats'].get('avg_main_matches', 0):.2f}/5")
    print(f"   • Ø Eurozahlen: {learning['stats'].get('avg_euro_matches', 0):.2f}/2")
    print(f"   • 3+ Treffer: {learning['stats'].get('three_plus_rate', 0):.1f}%")
    print(f"   • Gewinnklasse: {learning['stats'].get('gewinnklasse_rate', 0):.1f}%")
    print()

    # Gewinnklassen heute
    if gewinnklassen_count:
        print("🎰 Gewinnklassen heute:")
        for klasse, count in sorted(gewinnklassen_count.items()):
            info = [v for k, v in GEWINNKLASSEN.items() if v['klasse'] == klasse][0]
            print(f"   Klasse {klasse} ({info['name']}): {count}x")
        print()

    # Top Strategien nach Gewicht
    print("🏆 Top 10 Strategien (nach Selbstlernen):")
    sorted_strategies = sorted(
        strategy_weights['strategies'].items(),
        key=lambda x: x[1].get('weight', 1.0),
        reverse=True
    )
    for i, (strategy_id, stats) in enumerate(sorted_strategies[:10], 1):
        avg_main = stats['total_main_matches'] / stats['total_predictions'] if stats['total_predictions'] > 0 else 0
        print(f"   {i}. {strategy_id}: Gewicht={stats['weight']:.3f} | Ø={avg_main:.2f}/5 | n={stats['total_predictions']}")

    print()

    # Provider-Ranking
    print("🤖 Provider-Ranking:")
    ranked_providers = sorted(
        provider_scores.items(),
        key=lambda x: x[1].get('main_accuracy', 0),
        reverse=True
    )
    for i, (prov, stats) in enumerate(ranked_providers[:5], 1):
        print(f"   {i}. {prov}: {stats.get('main_accuracy', 0):.1f}% Haupt | {stats.get('euro_accuracy', 0):.1f}% Euro")

    print()
    print("=" * 60)
    print("✅ Lernen abgeschlossen - Strategie-Gewichte aktualisiert!")
    print("=" * 60)

    # ML-Training wenn angefordert
    if ML_AVAILABLE and ('--full-train' in sys.argv or '--train-ml' in sys.argv):
        print("\n" + "=" * 60)
        print("🧠 EUROJACKPOT ML-TRAINING")
        print("=" * 60)
        train_eurojackpot_ml(draws)

if __name__ == "__main__":
    learn_from_results()
