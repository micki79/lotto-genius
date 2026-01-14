#!/usr/bin/env python3
"""
🌀 LottoGenius - Glücksspirale Vorhersage-System mit Selbstlernen

17+ Strategien für 7-stellige Zahlen:
- Positions-basierte Vorhersagen
- ML-Algorithmen (Markov, Monte-Carlo, Bayesian)
- Selbstlernendes Gewichtungssystem
- TOP 10 beste Tipps Auswahl

Besonderheit: Glücksspirale läuft nur Samstags!
Hauptgewinn: 10.000€ monatliche Rente (20 Jahre) oder 2,1 Mio € einmalig
"""
import json
import os
import sys
from datetime import datetime, timedelta
import random
from collections import Counter, defaultdict

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
NUM_DIGITS = 7

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
    return [int(d) for d in str(number_str).zfill(NUM_DIGITS)]

def digits_to_number(digits):
    return ''.join(str(d) for d in digits)

class StrategyWeightManager:
    def __init__(self):
        self.weights_file = 'gluecksspirale_strategy_weights.json'
        self.weights = self.load_weights()

    def load_weights(self):
        data = load_json(self.weights_file, {})
        if not data.get('strategies'):
            data = {'strategies': {}, 'last_update': datetime.now().isoformat(), 'learning_rate': 0.15}
            save_json(self.weights_file, data)
        return data

    def get_weight(self, name):
        return self.weights.get('strategies', {}).get(name, {}).get('weight', 1.0)

    def save(self):
        self.weights['last_update'] = datetime.now().isoformat()
        save_json(self.weights_file, self.weights)

class GluecksspiraleStrategies:
    def __init__(self, draws, analysis, weight_manager):
        self.draws = draws
        self.analysis = analysis
        self.weight_manager = weight_manager
        self._compute_stats()

    def _compute_stats(self):
        self.position_freq = {i: Counter() for i in range(NUM_DIGITS)}
        for draw in self.draws[:150]:
            digits = get_digits(draw['number'])
            for pos, digit in enumerate(digits):
                self.position_freq[pos][digit] += 1

        recent = []
        for draw in self.draws[:25]:
            recent.extend(get_digits(draw['number']))
        freq = Counter(recent)
        self.hot_digits = [d for d, _ in freq.most_common(5)]
        self.cold_digits = [d for d, _ in freq.most_common()[-5:]]

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
        result = [self.position_freq[pos].most_common(1)[0][0] if self.position_freq[pos] else random.randint(0,9) for pos in range(NUM_DIGITS)]
        return digits_to_number(result), "Häufigste pro Position"

    def strategy_position_top3(self):
        result = []
        for pos in range(NUM_DIGITS):
            top3 = [d for d, _ in self.position_freq[pos].most_common(3)]
            result.append(random.choice(top3) if top3 else random.randint(0,9))
        return digits_to_number(result), "Zufällig aus Top 3"

    def strategy_hot_digits(self):
        pool = self.hot_digits if len(self.hot_digits) >= 3 else list(range(10))
        return digits_to_number([random.choice(pool) for _ in range(NUM_DIGITS)]), "Heiße Ziffern"

    def strategy_cold_digits(self):
        pool = self.cold_digits if len(self.cold_digits) >= 3 else list(range(10))
        return digits_to_number([random.choice(pool) for _ in range(NUM_DIGITS)]), "Kalte Ziffern"

    def strategy_overdue_mix(self):
        pool = self.overdue + self.hot_digits[:3]
        return digits_to_number([random.choice(pool) if pool else random.randint(0,9) for _ in range(NUM_DIGITS)]), "Überfällige + Hot"

    def strategy_odd_even(self):
        odd, even = [1,3,5,7,9], [0,2,4,6,8]
        result = [random.choice(odd if i % 2 == 0 else even) for i in range(NUM_DIGITS)]
        return digits_to_number(result), "Gerade/Ungerade Balance"

    def strategy_sum_optimized(self):
        target = self.analysis.get('sum_distribution', {}).get('average', 31)
        best = None
        best_diff = float('inf')
        for _ in range(500):
            combo = [random.randint(0, 9) for _ in range(NUM_DIGITS)]
            diff = abs(sum(combo) - target)
            if diff < best_diff:
                best_diff = diff
                best = combo
        return digits_to_number(best), f"Quersumme ~{int(target)}"

    def strategy_no_doubles(self):
        result = []
        for _ in range(NUM_DIGITS):
            avail = [d for d in range(10) if not result or d != result[-1]]
            result.append(random.choice(avail))
        return digits_to_number(result), "Keine Doppel"

    def strategy_markov(self):
        markov = self.analysis.get('markov', {}).get('transition_probabilities', {})
        result = [random.randint(0, 9)]
        for _ in range(NUM_DIGITS - 1):
            last = result[-1]
            probs = markov.get(last, {})
            if probs:
                next_d = max(probs.items(), key=lambda x: x[1])[0]
            else:
                next_d = random.randint(0, 9)
            result.append(int(next_d) if isinstance(next_d, str) else next_d)
        return digits_to_number(result), "Markov-Ketten"

    def strategy_monte_carlo(self):
        results = {i: Counter() for i in range(NUM_DIGITS)}
        for _ in range(500):
            for pos in range(NUM_DIGITS):
                freq = self.position_freq[pos]
                weights = [freq.get(d, 1) for d in range(10)]
                chosen = random.choices(range(10), weights=weights)[0]
                results[pos][chosen] += 1
        final = [results[pos].most_common(1)[0][0] for pos in range(NUM_DIGITS)]
        return digits_to_number(final), "Monte-Carlo"

    def strategy_bayesian(self):
        result = []
        for pos in range(NUM_DIGITS):
            freq = self.position_freq[pos]
            total = sum(freq.values()) or 1
            prior = {d: 1/10 for d in range(10)}
            likelihood = {d: (freq.get(d, 0) + 1) / (total + 10) for d in range(10)}
            posterior = {d: prior[d] * likelihood[d] for d in range(10)}
            result.append(max(posterior.items(), key=lambda x: x[1])[0])
        return digits_to_number(result), "Bayesian"

    def strategy_delta(self):
        common_deltas = self.analysis.get('delta', {}).get('most_common_deltas', [3,2,1])[:5]
        start = random.randint(2, 7)
        result = [start]
        for _ in range(NUM_DIGITS - 1):
            delta = random.choice(common_deltas) if common_deltas else random.randint(1, 3)
            result.append((result[-1] + random.choice([-1, 1]) * delta) % 10)
        return digits_to_number(result), "Delta-Muster"

    def strategy_fibonacci(self):
        fib = [1, 2, 3, 5, 8]
        non_fib = [0, 4, 6, 7, 9]
        result = [random.choice(fib) if random.random() < 0.6 else random.choice(non_fib) for _ in range(NUM_DIGITS)]
        return digits_to_number(result), "Fibonacci"

    def strategy_prime(self):
        primes = [2, 3, 5, 7]
        non_primes = [0, 1, 4, 6, 8, 9]
        result = [random.choice(primes) if random.random() < 0.5 else random.choice(non_primes) for _ in range(NUM_DIGITS)]
        return digits_to_number(result), "Primziffern"

    def strategy_last_variation(self):
        if not self.draws:
            return self.strategy_position_hot()
        last = get_digits(self.draws[0]['number'])
        result = [(d + random.choice([-2, -1, 0, 1, 2])) % 10 for d in last]
        return digits_to_number(result), "Letzte Variation"

    def strategy_symmetry(self):
        half = NUM_DIGITS // 2
        first = [random.randint(0, 9) for _ in range(half)]
        middle = [random.randint(0, 9)]
        second = first[::-1]
        return digits_to_number(first + middle + second), "Symmetrisch"

    def strategy_neural(self):
        result = []
        for pos in range(NUM_DIGITS):
            weights = {}
            for d in range(10):
                w = 0
                if d in self.hot_digits: w += 2.0
                if d in self.cold_digits: w += 0.5
                if d in self.overdue: w += 1.5
                w += self.position_freq[pos].get(d, 0) * 0.1
                weights[d] = w + random.random() * 0.5
            result.append(max(weights.items(), key=lambda x: x[1])[0])
        return digits_to_number(result), "Neuronales Netz"

    def get_all_strategies(self):
        strategies = [
            ('gs_position_hot', self.strategy_position_hot),
            ('gs_position_top3', self.strategy_position_top3),
            ('gs_hot', self.strategy_hot_digits),
            ('gs_cold', self.strategy_cold_digits),
            ('gs_overdue', self.strategy_overdue_mix),
            ('gs_odd_even', self.strategy_odd_even),
            ('gs_sum', self.strategy_sum_optimized),
            ('gs_no_doubles', self.strategy_no_doubles),
            ('gs_markov', self.strategy_markov),
            ('gs_monte_carlo', self.strategy_monte_carlo),
            ('gs_bayesian', self.strategy_bayesian),
            ('gs_delta', self.strategy_delta),
            ('gs_fibonacci', self.strategy_fibonacci),
            ('gs_prime', self.strategy_prime),
            ('gs_last_var', self.strategy_last_variation),
            ('gs_symmetry', self.strategy_symmetry),
            ('gs_neural', self.strategy_neural),
        ]
        weighted = [(n, f, self.weight_manager.get_weight(n)) for n, f in strategies]
        weighted.sort(key=lambda x: x[2], reverse=True)
        return weighted

def calculate_score(pred, analysis):
    score = pred.get('strategy_weight', 1.0) * 25
    digits = get_digits(pred['number'])
    digit_sum = sum(digits)
    opt = analysis.get('sum_distribution', {}).get('optimal_range', [25, 40])
    if opt[0] <= digit_sum <= opt[1]:
        score += 20
    odd_count = sum(1 for d in digits if d % 2 == 1)
    if 3 <= odd_count <= 5:
        score += 15
    hot = analysis.get('digit_frequency', {}).get('hot_digits', [])
    score += sum(2 for d in digits if d in hot)
    return round(score, 2)

def select_top_10(predictions, analysis):
    scored = []
    for p in predictions:
        p_copy = p.copy()
        p_copy['quality_score'] = calculate_score(p, analysis)
        scored.append(p_copy)
    scored.sort(key=lambda x: x['quality_score'], reverse=True)
    seen = set()
    unique = [p for p in scored if not (p['number'] in seen or seen.add(p['number']))]
    return unique[:10]

def generate_predictions():
    print("=" * 60)
    print("🌀 LottoGenius - Glücksspirale Vorhersage mit Selbstlernen")
    print("=" * 60)
    print(f"📅 {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}\n")

    data = load_json('gluecksspirale_data.json', {'draws': []})
    analysis = load_json('gluecksspirale_analysis.json', {})
    predictions = load_json('gluecksspirale_predictions.json', {'predictions': [], 'history': []})
    draws = data.get('draws', [])

    if not draws:
        print("⚠️ Keine Glücksspirale Daten!")
        return

    print(f"📊 Analysiere {len(draws)} Ziehungen...")
    weight_manager = StrategyWeightManager()

    if predictions.get('predictions'):
        predictions['history'].extend(predictions['predictions'])
        predictions['history'] = predictions['history'][-500:]

    strategies = GluecksspiraleStrategies(draws, analysis, weight_manager)
    all_predictions = []

    # ===== ECHTE ML-MODELLE =====
    if ML_AVAILABLE:
        print("\n🧠 ECHTE ML-Modelle (Neural Network, Markov, Bayesian):")
        try:
            ml_predictions = get_digit_game_ml_predictions('gluecksspirale', NUM_DIGITS, draws)
            for pred in ml_predictions:
                pred['strategy_weight'] = 2.5 if pred.get('is_champion') else 2.0
                pred['timestamp'] = datetime.now().isoformat()
                pred['verified'] = False
                all_predictions.append(pred)
                print(f"   ✅ {pred['method_name']}: {pred['number']}")
        except Exception as e:
            print(f"   ❌ ML-Fehler: {e}")

    print("\n🌀 Lokale Strategien (17 Methoden):")
    for name, fn, weight in strategies.get_all_strategies():
        try:
            number, desc = fn()
            all_predictions.append({
                'number': number, 'method': name, 'strategy': desc,
                'strategy_weight': weight, 'confidence': 50 + weight * 10,
                'timestamp': datetime.now().isoformat(), 'verified': False
            })
            print(f"   ✅ {name} [{weight:.2f}]: {number}")
        except Exception as e:
            print(f"   ❌ {name}: {e}")

    # Ensemble
    print("\n🏆 Ensemble-Voting:")
    if len(all_predictions) >= 5:
        pos_votes = {i: Counter() for i in range(NUM_DIGITS)}
        for p in all_predictions:
            for pos, d in enumerate(get_digits(p['number'])):
                pos_votes[pos][d] += p.get('strategy_weight', 1.0)
        champion = digits_to_number([pos_votes[pos].most_common(1)[0][0] for pos in range(NUM_DIGITS)])
        all_predictions.insert(0, {
            'number': champion, 'method': 'ensemble_champion', 'strategy': 'Gewichtetes Voting',
            'strategy_weight': 3.0, 'confidence': 95, 'timestamp': datetime.now().isoformat(),
            'verified': False, 'is_champion': True
        })
        print(f"   ✅ Champion: {champion}")

    print("\n🏅 Wähle TOP 10...")
    top_10 = select_top_10(all_predictions, analysis)

    print(f"\n{'='*60}\n🏅 TOP 10 BESTE TIPPS FÜR GLÜCKSSPIRALE:\n{'='*60}")
    for i, p in enumerate(top_10, 1):
        print(f"   {i:2}. {p['number']}  (Score: {p.get('quality_score', 0):.1f}, {p['method']})")

    # Nächster Samstag
    now = datetime.now()
    days_to_sat = (5 - now.weekday() + 7) % 7
    if days_to_sat == 0 and now.hour >= 20:
        days_to_sat = 7
    next_draw = now + timedelta(days=days_to_sat if days_to_sat > 0 else 7)

    predictions['predictions'] = top_10
    predictions['all_predictions'] = all_predictions
    predictions['last_update'] = datetime.now().isoformat()
    predictions['next_draw'] = next_draw.strftime('%d.%m.%Y')

    save_json('gluecksspirale_predictions.json', predictions)
    weight_manager.save()

    print(f"\n{'='*60}")
    print(f"✅ TOP 10 Glücksspirale Vorhersagen generiert!")
    print(f"📅 Nächste Ziehung (Samstag): {next_draw.strftime('%d.%m.%Y')}")
    print(f"🎁 Hauptgewinn: 10.000€/Monat für 20 Jahre!")
    print(f"{'='*60}")

if __name__ == "__main__":
    generate_predictions()
