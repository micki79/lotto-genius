#!/usr/bin/env python3
"""
🎰 LottoGenius - Spiel 77 Vorhersage-System mit Selbstlernen

15+ Strategien für 7-stellige Zahlen:
- Positions-basierte Vorhersagen
- Muster-basierte Strategien
- ML-Algorithmen (Markov, Monte-Carlo, Bayesian)
- Selbstlernendes Gewichtungssystem
- TOP 10 beste Tipps Auswahl
"""
import json
import os
import sys
from datetime import datetime, timedelta
import random
from collections import Counter, defaultdict

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
NUM_DIGITS = 7  # Spiel 77 hat 7 Ziffern

# Importiere echte ML-Modelle
try:
    from ml_models import (
        get_digit_game_ml_predictions,
        train_digit_game_ml,
        DigitGameEnsembleML
    )
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

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
    """Konvertiert Zahlenstring in Liste von Ziffern"""
    return [int(d) for d in str(number_str).zfill(NUM_DIGITS)]

def digits_to_number(digits):
    """Konvertiert Ziffernliste in Zahlenstring"""
    return ''.join(str(d) for d in digits)

# =====================================================
# STRATEGIE-GEWICHTUNGS-MANAGER (SELBSTLERNEN)
# =====================================================

class StrategyWeightManager:
    """Verwaltet Strategie-Gewichtungen mit Selbstlernen"""

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
            save_json(self.weights_file, data)
        return data

    def get_weight(self, strategy_name):
        strat = self.weights.get('strategies', {}).get(strategy_name, {})
        return strat.get('weight', 1.0)

    def save(self):
        self.weights['last_update'] = datetime.now().isoformat()
        save_json(self.weights_file, self.weights)

# =====================================================
# 15+ LOKALE STRATEGIEN
# =====================================================

class Spiel77Strategies:
    """15+ Strategien für Spiel 77 Vorhersagen"""

    def __init__(self, draws, analysis, weight_manager):
        self.draws = draws
        self.analysis = analysis
        self.weight_manager = weight_manager
        self._compute_base_stats()

    def _compute_base_stats(self):
        """Berechnet Grundstatistiken"""
        # Positions-Häufigkeit
        self.position_freq = {i: Counter() for i in range(NUM_DIGITS)}
        for draw in self.draws[:200]:
            digits = get_digits(draw['number'])
            for pos, digit in enumerate(digits):
                self.position_freq[pos][digit] += 1

        # Hot/Cold Ziffern
        recent_digits = []
        for draw in self.draws[:30]:
            recent_digits.extend(get_digits(draw['number']))
        self.hot_digits = [d for d, _ in Counter(recent_digits).most_common(5)]
        self.cold_digits = [d for d, _ in Counter(recent_digits).most_common()[-5:]]

        # Gaps
        self.gaps = {}
        for d in range(10):
            for i, draw in enumerate(self.draws):
                if d in get_digits(draw['number']):
                    self.gaps[d] = i
                    break
            else:
                self.gaps[d] = len(self.draws)

        self.overdue = [d for d, _ in sorted(self.gaps.items(), key=lambda x: x[1], reverse=True)[:5]]

    def strategy_position_hot(self):
        """Strategie 1: Häufigste Ziffer pro Position"""
        result = []
        for pos in range(NUM_DIGITS):
            if self.position_freq[pos]:
                result.append(self.position_freq[pos].most_common(1)[0][0])
            else:
                result.append(random.randint(0, 9))
        return digits_to_number(result), "Häufigste Ziffer pro Position"

    def strategy_position_random_top3(self):
        """Strategie 2: Zufällig aus Top 3 pro Position"""
        result = []
        for pos in range(NUM_DIGITS):
            top3 = [d for d, _ in self.position_freq[pos].most_common(3)]
            if top3:
                result.append(random.choice(top3))
            else:
                result.append(random.randint(0, 9))
        return digits_to_number(result), "Zufällig aus Top 3 pro Position"

    def strategy_hot_digits(self):
        """Strategie 3: Nur heiße Ziffern"""
        pool = self.hot_digits if len(self.hot_digits) >= 5 else list(range(10))
        result = [random.choice(pool) for _ in range(NUM_DIGITS)]
        return digits_to_number(result), "Heiße Ziffern bevorzugt"

    def strategy_cold_digits(self):
        """Strategie 4: Kalte Ziffern bevorzugt"""
        pool = self.cold_digits if len(self.cold_digits) >= 5 else list(range(10))
        result = [random.choice(pool) for _ in range(NUM_DIGITS)]
        return digits_to_number(result), "Kalte Ziffern bevorzugt"

    def strategy_overdue_mix(self):
        """Strategie 5: Überfällige Ziffern gemischt"""
        pool = self.overdue + self.hot_digits[:3]
        result = [random.choice(pool) if pool else random.randint(0, 9) for _ in range(NUM_DIGITS)]
        return digits_to_number(result), "Überfällige + heiße Ziffern"

    def strategy_balanced_odd_even(self):
        """Strategie 6: Ausgewogene Gerade/Ungerade"""
        odd = [1, 3, 5, 7, 9]
        even = [0, 2, 4, 6, 8]
        result = []
        for i in range(NUM_DIGITS):
            if i % 2 == 0:
                result.append(random.choice(odd))
            else:
                result.append(random.choice(even))
        return digits_to_number(result), "Ausgewogene Gerade/Ungerade"

    def strategy_sum_optimized(self):
        """Strategie 7: Optimierte Quersumme"""
        target = self.analysis.get('sum_distribution', {}).get('average', 31)
        best_combo = None
        best_diff = float('inf')

        for _ in range(500):
            combo = [random.randint(0, 9) for _ in range(NUM_DIGITS)]
            diff = abs(sum(combo) - target)
            if diff < best_diff:
                best_diff = diff
                best_combo = combo

        return digits_to_number(best_combo), f"Optimierte Quersumme ~{int(target)}"

    def strategy_no_doubles(self):
        """Strategie 8: Keine Doppelziffern"""
        result = []
        for _ in range(NUM_DIGITS):
            available = list(range(10))
            if result:
                available = [d for d in available if d != result[-1]]
            result.append(random.choice(available))
        return digits_to_number(result), "Keine aufeinanderfolgenden Doppel"

    def strategy_markov_chain(self):
        """Strategie 9: Markov-Ketten basiert"""
        markov = self.analysis.get('markov', {}).get('most_likely_next', {})

        result = [random.randint(0, 9)]  # Start
        for _ in range(NUM_DIGITS - 1):
            last = result[-1]
            next_digit = markov.get(last, random.randint(0, 9))
            if isinstance(next_digit, dict):
                next_digit = random.randint(0, 9)
            result.append(next_digit)

        return digits_to_number(result), "Markov-Ketten Vorhersage"

    def strategy_monte_carlo(self):
        """Strategie 10: Monte-Carlo Simulation"""
        simulations = 500
        results = {i: Counter() for i in range(NUM_DIGITS)}

        for _ in range(simulations):
            for pos in range(NUM_DIGITS):
                freq = self.position_freq[pos]
                if freq:
                    weights = [freq.get(d, 1) for d in range(10)]
                    chosen = random.choices(range(10), weights=weights)[0]
                else:
                    chosen = random.randint(0, 9)
                results[pos][chosen] += 1

        final = [results[pos].most_common(1)[0][0] for pos in range(NUM_DIGITS)]
        return digits_to_number(final), "Monte-Carlo Simulation"

    def strategy_bayesian(self):
        """Strategie 11: Bayesian Inference"""
        result = []
        for pos in range(NUM_DIGITS):
            freq = self.position_freq[pos]
            total = sum(freq.values()) or 1

            # Prior: Gleichverteilung
            prior = {d: 1/10 for d in range(10)}
            # Likelihood aus Häufigkeit
            likelihood = {d: (freq.get(d, 0) + 1) / (total + 10) for d in range(10)}
            # Posterior
            posterior = {d: prior[d] * likelihood[d] for d in range(10)}

            best = max(posterior.items(), key=lambda x: x[1])[0]
            result.append(best)

        return digits_to_number(result), "Bayesian Wahrscheinlichkeit"

    def strategy_delta_based(self):
        """Strategie 12: Delta-basiert"""
        delta_info = self.analysis.get('delta', {})
        common_deltas = delta_info.get('most_common_deltas', [3, 2, 4, 1, 5])

        start = random.randint(2, 7)
        result = [start]

        for _ in range(NUM_DIGITS - 1):
            delta = random.choice(common_deltas[:5]) if common_deltas else random.randint(1, 4)
            direction = random.choice([-1, 1])
            next_val = (result[-1] + delta * direction) % 10
            result.append(next_val)

        return digits_to_number(result), "Delta-Muster basiert"

    def strategy_fibonacci_heavy(self):
        """Strategie 13: Fibonacci-Ziffern bevorzugt"""
        fib = [1, 2, 3, 5, 8]
        non_fib = [0, 4, 6, 7, 9]

        result = []
        for i in range(NUM_DIGITS):
            if random.random() < 0.6:
                result.append(random.choice(fib))
            else:
                result.append(random.choice(non_fib))

        return digits_to_number(result), "Fibonacci-Ziffern bevorzugt"

    def strategy_prime_heavy(self):
        """Strategie 14: Primziffern bevorzugt"""
        primes = [2, 3, 5, 7]
        non_primes = [0, 1, 4, 6, 8, 9]

        result = []
        for i in range(NUM_DIGITS):
            if random.random() < 0.5:
                result.append(random.choice(primes))
            else:
                result.append(random.choice(non_primes))

        return digits_to_number(result), "Primziffern bevorzugt"

    def strategy_last_draw_variation(self):
        """Strategie 15: Variation der letzten Ziehung"""
        if not self.draws:
            return self.strategy_position_hot()

        last = get_digits(self.draws[0]['number'])
        result = []
        for d in last:
            variation = random.choice([-2, -1, 0, 1, 2])
            result.append((d + variation) % 10)

        return digits_to_number(result), "Variation der letzten Ziehung"

    def strategy_symmetry(self):
        """Strategie 16: Symmetrische Zahl"""
        half = NUM_DIGITS // 2
        first_half = [random.randint(0, 9) for _ in range(half)]
        middle = [random.randint(0, 9)] if NUM_DIGITS % 2 == 1 else []
        second_half = first_half[::-1]

        result = first_half + middle + second_half
        return digits_to_number(result), "Symmetrische/Palindrom Zahl"

    def strategy_neural_weighted(self):
        """Strategie 17: Neuronales Netz Simulation"""
        result = []
        for pos in range(NUM_DIGITS):
            weights = {}
            for d in range(10):
                w = 0
                if d in self.hot_digits: w += 2.0
                if d in self.cold_digits: w += 0.5
                if d in self.overdue: w += 1.5
                # Position-Bonus
                pos_freq = self.position_freq[pos].get(d, 0)
                w += pos_freq * 0.1
                weights[d] = w + random.random() * 0.5

            best = max(weights.items(), key=lambda x: x[1])[0]
            result.append(best)

        return digits_to_number(result), "Neuronales Netz gewichtet"

    def get_all_strategies(self):
        """Gibt alle Strategien mit Gewichtung zurück"""
        strategies = [
            ('sp77_position_hot', self.strategy_position_hot),
            ('sp77_position_top3', self.strategy_position_random_top3),
            ('sp77_hot_digits', self.strategy_hot_digits),
            ('sp77_cold_digits', self.strategy_cold_digits),
            ('sp77_overdue_mix', self.strategy_overdue_mix),
            ('sp77_odd_even', self.strategy_balanced_odd_even),
            ('sp77_sum_optimized', self.strategy_sum_optimized),
            ('sp77_no_doubles', self.strategy_no_doubles),
            ('sp77_markov', self.strategy_markov_chain),
            ('sp77_monte_carlo', self.strategy_monte_carlo),
            ('sp77_bayesian', self.strategy_bayesian),
            ('sp77_delta', self.strategy_delta_based),
            ('sp77_fibonacci', self.strategy_fibonacci_heavy),
            ('sp77_prime', self.strategy_prime_heavy),
            ('sp77_last_variation', self.strategy_last_draw_variation),
            ('sp77_symmetry', self.strategy_symmetry),
            ('sp77_neural', self.strategy_neural_weighted),
        ]

        weighted = [(name, fn, self.weight_manager.get_weight(name)) for name, fn in strategies]
        weighted.sort(key=lambda x: x[2], reverse=True)
        return weighted

# =====================================================
# TOP 10 AUSWAHL
# =====================================================

def calculate_prediction_score(prediction, analysis):
    """Berechnet Qualitätsscore für eine Vorhersage"""
    score = 0
    number = prediction.get('number', '0000000')
    digits = get_digits(number)

    # 1. Strategie-Gewicht (25%)
    strategy_weight = prediction.get('strategy_weight', 1.0)
    score += strategy_weight * 25

    # 2. Quersummen-Optimierung (20%)
    digit_sum = sum(digits)
    optimal = analysis.get('sum_distribution', {}).get('optimal_range', [25, 40])
    if optimal[0] <= digit_sum <= optimal[1]:
        score += 20
    elif abs(digit_sum - sum(optimal)/2) < 10:
        score += 10

    # 3. Positions-Häufigkeit (20%)
    pos_freq = analysis.get('position_frequency', {})
    for i, d in enumerate(digits):
        key = f'position_{i+1}'
        if key in pos_freq:
            if d in pos_freq[key].get('most_common', []):
                score += 3

    # 4. Keine langen Wiederholungen (15%)
    has_triple = any(number[i] == number[i+1] == number[i+2] for i in range(len(number)-2))
    if not has_triple:
        score += 15

    # 5. Gerade/Ungerade Balance (10%)
    odd_count = sum(1 for d in digits if d % 2 == 1)
    if 3 <= odd_count <= 5:
        score += 10

    # 6. Hot Digits Bonus (10%)
    hot = analysis.get('digit_frequency', {}).get('hot_digits', [])
    hot_count = sum(1 for d in digits if d in hot)
    score += hot_count * 2

    return round(score, 2)

def select_top_10_predictions(all_predictions, analysis):
    """Wählt die TOP 10 Vorhersagen aus"""
    scored = []
    for pred in all_predictions:
        score = calculate_prediction_score(pred, analysis)
        pred_copy = pred.copy()
        pred_copy['quality_score'] = score
        scored.append(pred_copy)

    scored.sort(key=lambda x: x['quality_score'], reverse=True)

    # Duplikate entfernen
    seen = set()
    unique = []
    for pred in scored:
        if pred['number'] not in seen:
            seen.add(pred['number'])
            unique.append(pred)

    return unique[:10]

# =====================================================
# HAUPTFUNKTION
# =====================================================

def generate_predictions():
    """Generiert Spiel 77 Vorhersagen"""

    print("=" * 60)
    print("🎰 LottoGenius - Spiel 77 Vorhersage mit Selbstlernen")
    print("=" * 60)
    print(f"📅 {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
    print()

    # Lade Daten
    data = load_json('spiel77_data.json', {'draws': []})
    analysis = load_json('spiel77_analysis.json', {})
    predictions = load_json('spiel77_predictions.json', {'predictions': [], 'history': []})

    draws = data.get('draws', [])
    if not draws:
        print("⚠️ Keine Spiel 77 Daten vorhanden!")
        return

    print(f"📊 Analysiere {len(draws)} historische Ziehungen...")

    # Weight Manager
    weight_manager = StrategyWeightManager()

    # Archiviere alte Vorhersagen
    if predictions.get('predictions'):
        predictions['history'].extend(predictions['predictions'])
        predictions['history'] = predictions['history'][-500:]

    # Strategien
    strategies = Spiel77Strategies(draws, analysis, weight_manager)
    all_predictions = []

    # ===== ECHTE ML-MODELLE =====
    if ML_AVAILABLE:
        print("\n🧠 ECHTE ML-Modelle (Neural Network, Markov, Bayesian):")
        try:
            ml_predictions = get_digit_game_ml_predictions('spiel77', NUM_DIGITS, draws)
            for pred in ml_predictions:
                pred['strategy_weight'] = 2.5 if pred.get('is_champion') else 2.0
                pred['timestamp'] = datetime.now().isoformat()
                pred['verified'] = False
                all_predictions.append(pred)
                print(f"   ✅ {pred['method_name']}: {pred['number']}")
        except Exception as e:
            print(f"   ❌ ML-Fehler: {e}")

    print("\n🎰 Lokale Strategien (17 Methoden):")

    weighted_strategies = strategies.get_all_strategies()

    for idx, (name, fn, weight) in enumerate(weighted_strategies):
        try:
            number, description = fn()
            all_predictions.append({
                'number': number,
                'method': name,
                'strategy': description,
                'strategy_weight': weight,
                'confidence': 50 + weight * 10 + random.random() * 10,
                'timestamp': datetime.now().isoformat(),
                'verified': False
            })
            print(f"   ✅ {name} [{weight:.2f}]: {number}")
        except Exception as e:
            print(f"   ❌ {name}: {e}")

    # Ensemble-Voting
    print("\n🏆 Ensemble-Voting:")
    if len(all_predictions) >= 5:
        position_votes = {i: Counter() for i in range(NUM_DIGITS)}
        for pred in all_predictions:
            weight = pred.get('strategy_weight', 1.0)
            digits = get_digits(pred['number'])
            for pos, d in enumerate(digits):
                position_votes[pos][d] += weight

        champion = [position_votes[pos].most_common(1)[0][0] for pos in range(NUM_DIGITS)]
        champion_number = digits_to_number(champion)

        all_predictions.insert(0, {
            'number': champion_number,
            'method': 'ensemble_champion',
            'strategy': f'Gewichtetes Voting aus {len(all_predictions)} Vorhersagen',
            'strategy_weight': 3.0,
            'confidence': 95,
            'timestamp': datetime.now().isoformat(),
            'verified': False,
            'is_champion': True
        })
        print(f"   ✅ Champion: {champion_number}")

    # TOP 10 Auswahl
    print("\n🏅 Wähle TOP 10 beste Tipps...")
    top_10 = select_top_10_predictions(all_predictions, analysis)

    print(f"\n{'='*60}")
    print("🏅 TOP 10 BESTE TIPPS FÜR SPIEL 77:")
    print(f"{'='*60}")
    for i, pred in enumerate(top_10, 1):
        print(f"   {i:2}. {pred['number']}  (Score: {pred.get('quality_score', 0):.1f}, {pred['method']})")

    # Speichern
    now = datetime.now()
    days_to_wed = (2 - now.weekday() + 7) % 7
    days_to_sat = (5 - now.weekday() + 7) % 7
    if days_to_wed == 0 and now.hour >= 19:
        days_to_wed = 7
    if days_to_sat == 0 and now.hour >= 20:
        days_to_sat = 7
    next_days = min(days_to_wed if days_to_wed > 0 else 7, days_to_sat if days_to_sat > 0 else 7)
    next_draw = now + timedelta(days=next_days)

    predictions['predictions'] = top_10
    predictions['all_predictions'] = all_predictions
    predictions['last_update'] = datetime.now().isoformat()
    predictions['next_draw'] = next_draw.strftime('%d.%m.%Y')
    predictions['stats'] = {
        'total_generated': len(all_predictions),
        'top_10_selected': len(top_10),
        'strategies_used': len(weighted_strategies)
    }

    save_json('spiel77_predictions.json', predictions)
    weight_manager.save()

    print(f"\n{'='*60}")
    print(f"✅ TOP 10 Spiel 77 Vorhersagen generiert!")
    print(f"📊 Insgesamt {len(all_predictions)} Tipps analysiert")
    print(f"📅 Nächste Ziehung: {next_draw.strftime('%d.%m.%Y')}")
    print(f"{'='*60}")

if __name__ == "__main__":
    generate_predictions()
