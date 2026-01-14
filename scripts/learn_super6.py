#!/usr/bin/env python3
"""
🧠 LottoGenius - Super 6 Lern-System mit Selbstlernen

Lernt aus jeder Ziehung:
1. Zählt übereinstimmende Endziffern (Gewinnklassen 1-6)
2. Aktualisiert Strategie-Gewichte
3. Trackt Performance über Zeit
"""
import json
import os
import sys
from datetime import datetime
from collections import Counter

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
NUM_DIGITS = 6


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

GEWINNKLASSEN = {
    1: {'endziffern': 6, 'name': 'Jackpot (alle 6)', 'weight_boost': 5.0},
    2: {'endziffern': 5, 'name': '5 Endziffern', 'weight_boost': 3.5},
    3: {'endziffern': 4, 'name': '4 Endziffern', 'weight_boost': 2.5},
    4: {'endziffern': 3, 'name': '3 Endziffern', 'weight_boost': 2.0},
    5: {'endziffern': 2, 'name': '2 Endziffern', 'weight_boost': 1.5},
    6: {'endziffern': 1, 'name': '1 Endziffer', 'weight_boost': 1.2}
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
    pred_digits = get_digits(predicted)[::-1]
    actual_digits = get_digits(actual)[::-1]
    matches = 0
    for p, a in zip(pred_digits, actual_digits):
        if p == a:
            matches += 1
        else:
            break
    return matches

def determine_gewinnklasse(end_matches):
    for klasse, info in GEWINNKLASSEN.items():
        if end_matches >= info['endziffern']:
            return klasse, info
    return None, None

class StrategyWeightUpdater:
    def __init__(self):
        self.weights_file = 'super6_strategy_weights.json'
        self.weights = self.load_weights()

    def load_weights(self):
        data = load_json(self.weights_file, {})
        if not data.get('strategies'):
            data = {'strategies': {}, 'last_update': datetime.now().isoformat(), 'learning_rate': 0.15}
        return data

    def update_strategy(self, name, end_matches, gewinnklasse=None):
        if name not in self.weights['strategies']:
            self.weights['strategies'][name] = {
                'weight': 1.0, 'total_matches': 0, 'total_predictions': 0, 'gewinnklassen': {}
            }
        strat = self.weights['strategies'][name]
        strat['total_predictions'] += 1
        strat['total_matches'] += end_matches

        if gewinnklasse:
            gk_key = str(gewinnklasse)
            strat['gewinnklassen'][gk_key] = strat['gewinnklassen'].get(gk_key, 0) + 1

        old_weight = strat['weight']
        avg = strat['total_matches'] / strat['total_predictions']
        lr = self.weights.get('learning_rate', 0.15)
        new_weight = old_weight * (1 - lr) + (avg / NUM_DIGITS) * 5 * lr

        if gewinnklasse:
            new_weight *= GEWINNKLASSEN.get(gewinnklasse, {}).get('weight_boost', 1.0)

        strat['weight'] = max(0.1, min(5.0, new_weight))
        strat['last_updated'] = datetime.now().isoformat()
        return strat['weight']

    def save(self):
        self.weights['last_update'] = datetime.now().isoformat()
        save_json(self.weights_file, self.weights)

    def get_ranking(self, n=10):
        strats = self.weights.get('strategies', {})
        return sorted(strats.items(), key=lambda x: x[1].get('weight', 1.0), reverse=True)[:n]

def learn_from_results():
    print("=" * 60)
    print("🧠 LottoGenius - Super 6 Lern-System")
    print("=" * 60)
    print(f"📅 {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}\n")

    data = load_json('super6_data.json', {'draws': []})
    predictions = load_json('super6_predictions.json', {'predictions': [], 'history': []})
    learning = load_json('super6_learning.json', {'entries': [], 'stats': {}})
    draws = data.get('draws', [])

    if not draws:
        print("⚠️ Keine Super 6 Daten!")
        return

    last_draw = draws[0]
    actual = str(last_draw['number']).zfill(NUM_DIGITS)
    draw_date = last_draw['date']

    print(f"📅 Letzte Ziehung: {draw_date}")
    print(f"🎲 Gewinnzahl: {actual}\n")

    weight_updater = StrategyWeightUpdater()
    unverified = [p for p in predictions.get('predictions', []) if not p.get('verified')]

    if not unverified:
        hist = [p for p in predictions.get('history', [])[-50:] if not p.get('verified')]
        if hist:
            unverified = hist
        else:
            print("ℹ️ Keine unverifizierten Vorhersagen")
            return

    print(f"📊 Verarbeite {len(unverified)} Vorhersagen...")
    print("-" * 40)

    new_entries = []
    gk_found = []

    for pred in unverified:
        pred_num = str(pred.get('number', '000000')).zfill(NUM_DIGITS)
        method = pred.get('method', 'unknown')
        end_matches = count_matching_end_digits(pred_num, actual)
        exact = sum(1 for p, a in zip(get_digits(pred_num), get_digits(actual)) if p == a)
        gk, gk_info = determine_gewinnklasse(end_matches)

        new_weight = weight_updater.update_strategy(method, end_matches, gk)

        new_entries.append({
            'date': datetime.now().isoformat(), 'draw_date': draw_date,
            'predicted': pred_num, 'actual': actual,
            'end_matches': end_matches, 'exact_matches': exact,
            'method': method, 'gewinnklasse': gk
        })

        if gk:
            gk_found.append({'klasse': gk, 'name': gk_info['name'], 'method': method})

        pred['verified'] = True
        pred['result'] = {'end_matches': end_matches, 'gewinnklasse': gk}

        ind = f"🏆 KLASSE {gk}" if gk else (f"✓ {end_matches} Endz." if end_matches else "•")
        print(f"  {ind} {method}: {pred_num} | {end_matches} Endz. [W:{new_weight:.2f}]")

    print("-" * 40)
    weight_updater.save()

    if gk_found:
        print("\n🏆 GEWINNKLASSEN:")
        for g in gk_found:
            print(f"   • Klasse {g['klasse']} ({g['name']}): {g['method']}")

    learning['entries'].extend(new_entries)
    learning['entries'] = learning['entries'][-1000:]

    total = len(learning['entries'])
    if total > 0:
        learning['stats'] = {
            'total_entries': total,
            'avg_end_matches': round(sum(e.get('end_matches', 0) for e in learning['entries']) / total, 2),
            'gewinnklassen': dict(Counter(e.get('gewinnklasse') for e in learning['entries'] if e.get('gewinnklasse'))),
            'last_update': datetime.now().isoformat()
        }

    save_json('super6_predictions.json', predictions)
    save_json('super6_learning.json', learning)

    print(f"\n✅ {len(new_entries)} Vorhersagen gelernt!")
    print(f"\n📊 Statistik:")
    print(f"   • Einträge: {learning['stats'].get('total_entries', 0)}")
    print(f"   • Ø Endziffern: {learning['stats'].get('avg_end_matches', 0):.2f}")

    print("\n📈 Top 10 Strategien:")
    for i, (name, data) in enumerate(weight_updater.get_ranking(10), 1):
        w = data.get('weight', 1.0)
        t = data.get('total_predictions', 0)
        m = data.get('total_matches', 0)
        print(f"   {i:2}. {name}: W={w:.2f} | {t} Vorh. | Ø {m/t if t else 0:.1f} Endz.")

    print(f"\n{'='*60}\n✅ Super 6 Lernen abgeschlossen!\n{'='*60}")

    # ML-Training wenn angefordert
    if ML_AVAILABLE and ('--full-train' in sys.argv or '--train-ml' in sys.argv):
        print("\n" + "=" * 60)
        print("🧠 SUPER 6 ML-TRAINING")
        print("=" * 60)
        train_digit_game_ml('super6', NUM_DIGITS, draws)

if __name__ == "__main__":
    learn_from_results()
