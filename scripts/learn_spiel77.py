#!/usr/bin/env python3
"""
🧠 LottoGenius - Spiel 77 Lern-System mit Selbstlernen

Lernt aus jeder Ziehung:
1. Vergleicht Vorhersagen mit echten Zahlen
2. Zählt übereinstimmende Endziffern (Gewinnklassen)
3. Aktualisiert Strategie-Gewichte
4. Trackt Performance über Zeit
"""
import json
import os
import sys
from datetime import datetime
from collections import Counter

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
NUM_DIGITS = 7


# Füge scripts-Verzeichnis zum Pfad hinzu für ML-Modelle Import
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Importiere echte ML-Modelle
try:
    from ml_models import train_digit_game_ml
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# Spiel 77 Gewinnklassen (von hinten gezählt)
GEWINNKLASSEN = {
    1: {'endziffern': 7, 'name': 'Jackpot (alle 7)', 'weight_boost': 5.0},
    2: {'endziffern': 6, 'name': '6 Endziffern', 'weight_boost': 4.0},
    3: {'endziffern': 5, 'name': '5 Endziffern', 'weight_boost': 3.0},
    4: {'endziffern': 4, 'name': '4 Endziffern', 'weight_boost': 2.5},
    5: {'endziffern': 3, 'name': '3 Endziffern', 'weight_boost': 2.0},
    6: {'endziffern': 2, 'name': '2 Endziffern', 'weight_boost': 1.5},
    7: {'endziffern': 1, 'name': '1 Endziffer', 'weight_boost': 1.2}
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
        json.dump(data, f, indent=2, ensure_ascii=False)

def get_digits(number_str):
    return [int(d) for d in str(number_str).zfill(NUM_DIGITS)]

def count_matching_end_digits(predicted, actual):
    """Zählt übereinstimmende Endziffern (von hinten)"""
    pred_digits = get_digits(predicted)[::-1]  # Umkehren
    actual_digits = get_digits(actual)[::-1]

    matches = 0
    for p, a in zip(pred_digits, actual_digits):
        if p == a:
            matches += 1
        else:
            break  # Bei erster Nicht-Übereinstimmung aufhören

    return matches

def determine_gewinnklasse(end_matches):
    """Bestimmt Gewinnklasse basierend auf Endziffern"""
    for klasse, info in GEWINNKLASSEN.items():
        if end_matches >= info['endziffern']:
            return klasse, info
    return None, None

class StrategyWeightUpdater:
    """Aktualisiert Strategie-Gewichte basierend auf Performance"""

    def __init__(self):
        self.weights_file = 'spiel77_strategy_weights.json'
        self.weights = self.load_weights()

    def load_weights(self):
        data = load_json(self.weights_file, {})
        if not data.get('strategies'):
            data = {
                'strategies': {},
                'last_update': datetime.now().isoformat(),
                'total_predictions': 0,
                'learning_rate': 0.15
            }
        return data

    def update_strategy(self, strategy_name, end_matches, gewinnklasse=None):
        """Aktualisiert eine Strategie"""
        if strategy_name not in self.weights['strategies']:
            self.weights['strategies'][strategy_name] = {
                'weight': 1.0,
                'total_matches': 0,
                'total_predictions': 0,
                'gewinnklassen': {},
                'last_updated': None
            }

        strat = self.weights['strategies'][strategy_name]
        strat['total_predictions'] += 1
        strat['total_matches'] += end_matches

        if gewinnklasse:
            gk_key = str(gewinnklasse)
            if gk_key not in strat['gewinnklassen']:
                strat['gewinnklassen'][gk_key] = 0
            strat['gewinnklassen'][gk_key] += 1

        # Neues Gewicht berechnen
        old_weight = strat['weight']
        avg_matches = strat['total_matches'] / strat['total_predictions']
        lr = self.weights.get('learning_rate', 0.15)

        # Normalisiert auf 0-7 Matches
        new_weight = old_weight * (1 - lr) + (avg_matches / 7) * 5 * lr

        # Bonus für Gewinnklasse
        if gewinnklasse:
            boost = GEWINNKLASSEN.get(gewinnklasse, {}).get('weight_boost', 1.0)
            new_weight *= boost

        strat['weight'] = max(0.1, min(5.0, new_weight))
        strat['last_updated'] = datetime.now().isoformat()

        return strat['weight']

    def save(self):
        self.weights['last_update'] = datetime.now().isoformat()
        self.weights['total_predictions'] = sum(
            s.get('total_predictions', 0) for s in self.weights['strategies'].values()
        )
        save_json(self.weights_file, self.weights)

    def get_ranking(self, top_n=10):
        strategies = self.weights.get('strategies', {})
        sorted_strats = sorted(
            strategies.items(),
            key=lambda x: x[1].get('weight', 1.0),
            reverse=True
        )
        return sorted_strats[:top_n]

def learn_from_results():
    """Lernt aus Spiel 77 Ergebnissen"""

    print("=" * 60)
    print("🧠 LottoGenius - Spiel 77 Lern-System")
    print("=" * 60)
    print(f"📅 {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
    print()

    # Lade Daten
    data = load_json('spiel77_data.json', {'draws': []})
    predictions = load_json('spiel77_predictions.json', {'predictions': [], 'history': []})
    learning = load_json('spiel77_learning.json', {'entries': [], 'stats': {}})

    draws = data.get('draws', [])
    if not draws:
        print("⚠️ Keine Spiel 77 Daten vorhanden!")
        return

    last_draw = draws[0]
    actual_number = str(last_draw['number']).zfill(NUM_DIGITS)
    draw_date = last_draw['date']

    print(f"📅 Letzte Ziehung: {draw_date}")
    print(f"🎰 Gewinnzahl: {actual_number}")
    print()

    # Weight Updater
    weight_updater = StrategyWeightUpdater()

    # Unverarbeitete Vorhersagen
    unverified = [p for p in predictions.get('predictions', []) if not p.get('verified')]

    if not unverified:
        recent_history = [p for p in predictions.get('history', [])[-50:] if not p.get('verified')]
        if recent_history:
            unverified = recent_history
            print(f"📚 {len(unverified)} Vorhersagen aus Historie")
        else:
            print("ℹ️ Keine unverifizierten Vorhersagen")
            return

    print(f"📊 Verarbeite {len(unverified)} Vorhersagen...")
    print("-" * 40)

    new_entries = []
    gewinnklassen_found = []

    for pred in unverified:
        pred_number = str(pred.get('number', '0000000')).zfill(NUM_DIGITS)
        method = pred.get('method', 'unknown')

        # Zähle Endziffern
        end_matches = count_matching_end_digits(pred_number, actual_number)

        # Zähle auch exakte Matches (Position)
        pred_digits = get_digits(pred_number)
        actual_digits = get_digits(actual_number)
        exact_matches = sum(1 for p, a in zip(pred_digits, actual_digits) if p == a)

        # Gewinnklasse
        gewinnklasse, gk_info = determine_gewinnklasse(end_matches)

        # Update Strategie-Gewicht
        new_weight = weight_updater.update_strategy(method, end_matches, gewinnklasse)

        # Lern-Eintrag
        entry = {
            'date': datetime.now().isoformat(),
            'draw_date': draw_date,
            'predicted': pred_number,
            'actual': actual_number,
            'end_matches': end_matches,
            'exact_matches': exact_matches,
            'method': method,
            'gewinnklasse': gewinnklasse,
            'new_weight': new_weight
        }
        new_entries.append(entry)

        if gewinnklasse:
            gewinnklassen_found.append({
                'klasse': gewinnklasse,
                'name': gk_info['name'],
                'method': method,
                'number': pred_number
            })

        # Markieren
        pred['verified'] = True
        pred['result'] = {
            'end_matches': end_matches,
            'exact_matches': exact_matches,
            'gewinnklasse': gewinnklasse
        }

        # Output
        if gewinnklasse:
            indicator = f"🏆 KLASSE {gewinnklasse}"
        elif end_matches >= 1:
            indicator = f"✓ {end_matches} Endz."
        else:
            indicator = "•"

        print(f"  {indicator} {method}: {pred_number} | {end_matches} Endz. | {exact_matches} exakt [W:{new_weight:.2f}]")

    print("-" * 40)

    # Speichern
    weight_updater.save()

    if gewinnklassen_found:
        print()
        print("🏆 GEWINNKLASSEN GEFUNDEN:")
        for gk in gewinnklassen_found:
            print(f"   • Klasse {gk['klasse']} ({gk['name']}): {gk['method']}")

    # Learning speichern
    learning['entries'].extend(new_entries)
    learning['entries'] = learning['entries'][-1000:]

    total = len(learning['entries'])
    if total > 0:
        avg_end = sum(e.get('end_matches', 0) for e in learning['entries']) / total
        avg_exact = sum(e.get('exact_matches', 0) for e in learning['entries']) / total

        gk_counts = Counter(e.get('gewinnklasse') for e in learning['entries'] if e.get('gewinnklasse'))

        learning['stats'] = {
            'total_entries': total,
            'avg_end_matches': round(avg_end, 2),
            'avg_exact_matches': round(avg_exact, 2),
            'gewinnklassen': dict(gk_counts),
            'last_update': datetime.now().isoformat()
        }

    save_json('spiel77_predictions.json', predictions)
    save_json('spiel77_learning.json', learning)

    print()
    print(f"✅ {len(new_entries)} Vorhersagen gelernt!")
    print()
    print("📊 Gesamt-Statistik:")
    print(f"   • Einträge: {learning['stats'].get('total_entries', 0)}")
    print(f"   • Ø Endziffern: {learning['stats'].get('avg_end_matches', 0):.2f}")
    print(f"   • Ø Exakte: {learning['stats'].get('avg_exact_matches', 0):.2f}")

    print()
    print("📈 Top 10 Strategien:")
    for i, (name, data) in enumerate(weight_updater.get_ranking(10), 1):
        w = data.get('weight', 1.0)
        t = data.get('total_predictions', 0)
        m = data.get('total_matches', 0)
        avg = m / t if t > 0 else 0
        print(f"   {i:2}. {name}: W={w:.2f} | {t} Vorh. | Ø {avg:.1f} Endz.")

    print()
    print("=" * 60)
    print("✅ Spiel 77 Lernen abgeschlossen!")
    print("=" * 60)

    # ML-Training wenn angefordert
    if ML_AVAILABLE and ('--full-train' in sys.argv or '--train-ml' in sys.argv):
        print("\n" + "=" * 60)
        print("🧠 SPIEL 77 ML-TRAINING")
        print("=" * 60)
        train_digit_game_ml('spiel77', NUM_DIGITS, draws)

if __name__ == "__main__":
    learn_from_results()
