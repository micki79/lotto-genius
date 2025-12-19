#!/usr/bin/env python3
"""
🍀 LottoGenius - VOLLSTÄNDIGES Multi-KI Vorhersage-System mit Selbstlernen

Integriert 7 kostenlose KI-APIs + 15 lokale Strategien:
1. Google Gemini (1M Tokens/Tag)
2. Groq (ultraschnell)
3. HuggingFace (tausende Modelle)
4. OpenRouter (50+ Modelle)
5. Together AI ($25 Startguthaben)
6. DeepSeek (komplett kostenlos)
7. Lokale ML-Algorithmen (15 Strategien + 6 ML-Modelle)

Features:
- 6-Faktoren Superzahl-Analyse
- Selbstlernendes Gewichtungssystem
- TOP 10 beste Tipps Auswahl
- Kontinuierliches Lernen aus Ergebnissen
"""
import json
import os
from datetime import datetime, timedelta
import random
from collections import Counter
import requests
import time
import hashlib
import math

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

# =====================================================
# 7 KOSTENLOSE KI-PROVIDER KONFIGURATION
# =====================================================

KI_PROVIDERS = {
    'gemini': {
        'name': 'Google Gemini',
        'emoji': '🔮',
        'url': 'https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent',
        'free_tier': '1M Tokens/Tag',
        'env_key': 'GEMINI_API_KEY'
    },
    'groq': {
        'name': 'Groq (Llama)',
        'emoji': '⚡',
        'url': 'https://api.groq.com/openai/v1/chat/completions',
        'free_tier': '1000 Requests/Tag',
        'env_key': 'GROQ_API_KEY'
    },
    'huggingface': {
        'name': 'HuggingFace',
        'emoji': '🤗',
        'url': 'https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1',
        'free_tier': 'Kostenloser Tier',
        'env_key': 'HUGGINGFACE_API_KEY'
    },
    'openrouter': {
        'name': 'OpenRouter',
        'emoji': '🌐',
        'url': 'https://openrouter.ai/api/v1/chat/completions',
        'free_tier': '50+ Modelle',
        'env_key': 'OPENROUTER_API_KEY'
    },
    'together': {
        'name': 'Together AI',
        'emoji': '🚀',
        'url': 'https://api.together.xyz/v1/chat/completions',
        'free_tier': '$25 Startguthaben',
        'env_key': 'TOGETHER_API_KEY'
    },
    'deepseek': {
        'name': 'DeepSeek',
        'emoji': '🧠',
        'url': 'https://api.deepseek.com/v1/chat/completions',
        'free_tier': 'Komplett kostenlos',
        'env_key': 'DEEPSEEK_API_KEY'
    },
    'local_ml': {
        'name': 'Lokale ML-Modelle',
        'emoji': '🖥️',
        'url': None,
        'free_tier': 'Immer verfügbar',
        'env_key': None
    }
}

# =====================================================
# HILFSFUNKTIONEN
# =====================================================

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

def get_api_key(provider):
    """Holt API-Key aus Environment Variable oder secrets.json"""
    env_key = KI_PROVIDERS.get(provider, {}).get('env_key')
    if env_key:
        key = os.environ.get(env_key)
        if key:
            return key

    # Fallback: secrets.json
    secrets = load_json('secrets.json', {})
    return secrets.get(provider)

# =====================================================
# STRATEGIE-GEWICHTUNGS-MANAGER (SELBSTLERNEN)
# =====================================================

class StrategyWeightManager:
    """
    Verwaltet Strategie-Gewichtungen basierend auf historischer Performance.
    Lernt kontinuierlich aus Vorhersage-Ergebnissen.
    """

    def __init__(self):
        self.weights_file = 'strategy_weights.json'
        self.weights = self.load_weights()

    def load_weights(self):
        """Lädt Gewichtungen oder initialisiert mit Standardwerten"""
        data = load_json(self.weights_file, {})
        if not data.get('strategies'):
            # Standardgewichtungen für alle 15 Strategien
            data = {
                'strategies': {
                    'hot_cold': {'weight': 1.0, 'hits': 0, 'total': 0, 'last_updated': None},
                    'cold_numbers': {'weight': 1.0, 'hits': 0, 'total': 0, 'last_updated': None},
                    'overdue': {'weight': 1.0, 'hits': 0, 'total': 0, 'last_updated': None},
                    'odd_even_33': {'weight': 1.0, 'hits': 0, 'total': 0, 'last_updated': None},
                    'odd_even_42': {'weight': 1.0, 'hits': 0, 'total': 0, 'last_updated': None},
                    'sum_optimized': {'weight': 1.0, 'hits': 0, 'total': 0, 'last_updated': None},
                    'decade_balance': {'weight': 1.0, 'hits': 0, 'total': 0, 'last_updated': None},
                    'delta_pattern': {'weight': 1.0, 'hits': 0, 'total': 0, 'last_updated': None},
                    'position_based': {'weight': 1.0, 'hits': 0, 'total': 0, 'last_updated': None},
                    'no_consecutive': {'weight': 1.0, 'hits': 0, 'total': 0, 'last_updated': None},
                    'prime_mix': {'weight': 1.0, 'hits': 0, 'total': 0, 'last_updated': None},
                    'low_high': {'weight': 1.0, 'hits': 0, 'total': 0, 'last_updated': None},
                    'hot_cold_mix': {'weight': 1.0, 'hits': 0, 'total': 0, 'last_updated': None},
                    'monte_carlo': {'weight': 1.0, 'hits': 0, 'total': 0, 'last_updated': None},
                    'bayesian': {'weight': 1.0, 'hits': 0, 'total': 0, 'last_updated': None},
                    'neural_network': {'weight': 1.0, 'hits': 0, 'total': 0, 'last_updated': None},
                    'lstm_sequence': {'weight': 1.0, 'hits': 0, 'total': 0, 'last_updated': None},
                    'random_forest': {'weight': 1.0, 'hits': 0, 'total': 0, 'last_updated': None},
                    'fibonacci': {'weight': 1.0, 'hits': 0, 'total': 0, 'last_updated': None},
                    'neighbor_pairs': {'weight': 1.0, 'hits': 0, 'total': 0, 'last_updated': None},
                    'end_digit': {'weight': 1.0, 'hits': 0, 'total': 0, 'last_updated': None}
                },
                'last_update': datetime.now().isoformat(),
                'total_predictions': 0,
                'learning_rate': 0.15
            }
            save_json(self.weights_file, data)
        return data

    def get_weight(self, strategy_name):
        """Gibt das aktuelle Gewicht einer Strategie zurück"""
        strat = self.weights.get('strategies', {}).get(strategy_name, {})
        return strat.get('weight', 1.0)

    def update_weight(self, strategy_name, hits, total):
        """Aktualisiert Gewicht basierend auf Performance"""
        if strategy_name not in self.weights['strategies']:
            self.weights['strategies'][strategy_name] = {
                'weight': 1.0, 'hits': 0, 'total': 0, 'last_updated': None
            }

        strat = self.weights['strategies'][strategy_name]
        strat['hits'] += hits
        strat['total'] += total

        # Berechne neues Gewicht mit exponential moving average
        if strat['total'] > 0:
            hit_rate = strat['hits'] / strat['total']
            lr = self.weights.get('learning_rate', 0.15)
            strat['weight'] = strat['weight'] * (1 - lr) + hit_rate * 10 * lr
            strat['weight'] = max(0.1, min(5.0, strat['weight']))  # Begrenzen

        strat['last_updated'] = datetime.now().isoformat()
        save_json(self.weights_file, self.weights)

    def get_top_strategies(self, n=10):
        """Gibt die n besten Strategien zurück"""
        strategies = self.weights.get('strategies', {})
        sorted_strats = sorted(strategies.items(), key=lambda x: x[1].get('weight', 1.0), reverse=True)
        return sorted_strats[:n]

# =====================================================
# SUPERZAHL-ANALYSE (6-FAKTOREN-ALGORITHMUS)
# =====================================================

class SuperzahlAnalyzer:
    """
    Analysiert Superzahl-Muster mit 6 verschiedenen Faktoren:
    1. Häufigkeit (20%) - Wie oft wurde jede SZ gezogen?
    2. Trend (25%) - Ist sie aktuell "heiß" oder "kalt"?
    3. Wochentag (15%) - Unterschiede Mittwoch vs Samstag
    4. Lücke (20%) - Wie lange nicht gezogen (überfällig)?
    5. Folge-Muster (15%) - Welche SZ kommt nach welcher?
    6. Anti-Serie (5%) - Vermeidet Wiederholungen
    """

    def __init__(self, draws):
        self.draws = draws
        self.patterns = {}
        if draws:
            self.analyze_all_patterns()

    def analyze_all_patterns(self):
        """Analysiert alle 6 Faktoren"""
        # 1. Häufigkeit
        sz_freq = Counter(d.get('superzahl', 0) for d in self.draws if 'superzahl' in d)
        self.patterns['frequency'] = dict(sz_freq)

        # 2. Lücken
        sz_gaps = {}
        for sz in range(10):
            for i, d in enumerate(self.draws):
                if d.get('superzahl') == sz:
                    sz_gaps[sz] = i
                    break
            else:
                sz_gaps[sz] = len(self.draws)
        self.patterns['gaps'] = sz_gaps

        # 3. Trend (letzte 20 vs vorherige 20)
        recent = self.draws[:20]
        older = self.draws[20:40]
        recent_freq = Counter(d.get('superzahl', 0) for d in recent if 'superzahl' in d)
        older_freq = Counter(d.get('superzahl', 0) for d in older if 'superzahl' in d)

        trends = {}
        for sz in range(10):
            r = recent_freq.get(sz, 0)
            o = older_freq.get(sz, 0)
            trends[sz] = r - o
        self.patterns['trends'] = trends

        # 4. Wochentag-Muster
        wed_freq = Counter()
        sat_freq = Counter()
        for d in self.draws[:100]:
            if 'superzahl' not in d:
                continue
            try:
                day, month, year = map(int, d['date'].split('.'))
                date_obj = datetime(year, month, day)
                if date_obj.weekday() == 2:  # Mittwoch
                    wed_freq[d['superzahl']] += 1
                elif date_obj.weekday() == 5:  # Samstag
                    sat_freq[d['superzahl']] += 1
            except:
                pass
        self.patterns['wednesday'] = dict(wed_freq)
        self.patterns['saturday'] = dict(sat_freq)

        # 5. Folge-Muster
        follows = Counter()
        for i in range(len(self.draws) - 1):
            current = self.draws[i].get('superzahl')
            previous = self.draws[i + 1].get('superzahl')
            if current is not None and previous is not None:
                follows[(previous, current)] += 1
        self.patterns['follows'] = {f"{k[0]}->{k[1]}": v for k, v in follows.most_common(30)}

        # 6. Letzte Superzahl (für Anti-Serie)
        self.patterns['last_sz'] = self.draws[0].get('superzahl') if self.draws else None

    def predict_best_superzahl(self):
        """Berechnet die beste Superzahl basierend auf allen 6 Faktoren"""
        if not self.patterns:
            return random.randint(0, 9), [(i, 10) for i in range(10)]

        scores = {}

        freq = self.patterns.get('frequency', {})
        gaps = self.patterns.get('gaps', {})
        trends = self.patterns.get('trends', {})
        last_sz = self.patterns.get('last_sz')

        # Normalisierung
        max_freq = max(freq.values()) if freq else 1
        max_gap = max(gaps.values()) if gaps else 1
        max_trend = max(abs(v) for v in trends.values()) if trends and any(trends.values()) else 1

        # Bestimme Wochentag für die nächste Ziehung
        today = datetime.now()
        days_to_wed = (2 - today.weekday() + 7) % 7
        days_to_sat = (5 - today.weekday() + 7) % 7
        is_wednesday = days_to_wed < days_to_sat or (days_to_wed == 0 and today.hour < 19)

        day_freq = self.patterns.get('wednesday' if is_wednesday else 'saturday', {})
        max_day = max(day_freq.values()) if day_freq else 1

        for sz in range(10):
            score = 0

            # 1. Häufigkeit (20%) - Häufige bevorzugen
            score += (freq.get(sz, freq.get(str(sz), 0)) / max_freq) * 20

            # 2. Trend (25%) - Steigende bevorzugen
            trend_val = trends.get(sz, 0)
            normalized_trend = (trend_val + max_trend) / (2 * max_trend) if max_trend > 0 else 0.5
            score += normalized_trend * 25

            # 3. Wochentag (15%)
            score += (day_freq.get(sz, day_freq.get(str(sz), 0)) / max_day) * 15

            # 4. Lücke (20%) - Überfällige bevorzugen
            score += (gaps.get(sz, gaps.get(str(sz), 0)) / max_gap) * 20

            # 5. Folge-Muster (15%)
            if last_sz is not None:
                follow_key = f"{last_sz}->{sz}"
                follow_count = 0
                for k, v in self.patterns.get('follows', {}).items():
                    if k == follow_key:
                        follow_count = v
                        break
                max_follow = max(self.patterns.get('follows', {}).values()) if self.patterns.get('follows') else 1
                score += (follow_count / max_follow) * 15

            # 6. Anti-Serie (5%) - Vermeidet Wiederholung
            if sz != last_sz:
                score += 5

            # Kleiner Zufallsfaktor für Diversität
            score += random.random() * 3

            scores[sz] = round(score, 2)

        # Sortiere und gib Ranking zurück
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_sz = ranked[0][0]

        return best_sz, ranked

# =====================================================
# 15+ LOKALE STRATEGIEN
# =====================================================

class LocalStrategies:
    """
    Implementiert 15+ lokale Strategien für Lotto 6aus49.
    Jede Strategie verwendet unterschiedliche Analysemethoden.
    """

    PRIMES_49 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    FIBONACCI = [1, 2, 3, 5, 8, 13, 21, 34]  # Bis 49

    def __init__(self, draws, analysis, weight_manager):
        self.draws = draws
        self.analysis = analysis
        self.weight_manager = weight_manager
        self.all_numbers = list(range(1, 50))

        # Berechne Grundstatistiken
        self._compute_base_stats()

    def _compute_base_stats(self):
        """Berechnet grundlegende Statistiken aus den Ziehungen"""
        # Häufigkeit
        all_nums = []
        for d in self.draws[:200]:
            all_nums.extend(d.get('numbers', []))
        self.freq = Counter(all_nums)

        # Lücken (Gap seit letzter Ziehung)
        self.gaps = {}
        for num in range(1, 50):
            for i, d in enumerate(self.draws):
                if num in d.get('numbers', []):
                    self.gaps[num] = i
                    break
            else:
                self.gaps[num] = len(self.draws)

        # Hot/Cold
        recent_nums = []
        for d in self.draws[:30]:
            recent_nums.extend(d.get('numbers', []))
        recent_freq = Counter(recent_nums)
        self.hot_numbers = [n for n, _ in recent_freq.most_common(15)]
        self.cold_numbers = [n for n in range(1, 50) if recent_freq.get(n, 0) <= 1]

        # Overdue
        self.overdue = [n for n, g in sorted(self.gaps.items(), key=lambda x: x[1], reverse=True)[:15]]

    def strategy_hot_cold(self):
        """Strategie 1: Heiße und kalte Zahlen gemischt"""
        hot = self.hot_numbers[:10]
        cold = self.cold_numbers[:5]

        numbers = random.sample(hot, min(4, len(hot)))
        remaining = [n for n in cold if n not in numbers]
        numbers.extend(random.sample(remaining, min(2, len(remaining))))

        while len(numbers) < 6:
            n = random.randint(1, 49)
            if n not in numbers:
                numbers.append(n)

        return sorted(numbers[:6]), "Kombination aus heißen und kalten Zahlen"

    def strategy_cold_numbers(self):
        """Strategie 2: Nur kalte/seltene Zahlen"""
        cold = [n for n, c in self.freq.most_common()[-20:]]
        if len(cold) >= 6:
            numbers = random.sample(cold, 6)
        else:
            numbers = cold + random.sample([n for n in range(1, 50) if n not in cold], 6 - len(cold))
        return sorted(numbers[:6]), "Kalte/seltene Zahlen bevorzugt"

    def strategy_overdue(self):
        """Strategie 3: Überfällige Zahlen"""
        overdue = self.overdue[:12]
        if len(overdue) >= 6:
            numbers = random.sample(overdue, 6)
        else:
            numbers = overdue + random.sample([n for n in range(1, 50) if n not in overdue], 6 - len(overdue))
        return sorted(numbers[:6]), "Überfällige Zahlen mit großer Lücke"

    def strategy_odd_even_33(self):
        """Strategie 4: 3 gerade, 3 ungerade"""
        odd = [n for n in range(1, 50) if n % 2 == 1]
        even = [n for n in range(1, 50) if n % 2 == 0]
        numbers = random.sample(odd, 3) + random.sample(even, 3)
        return sorted(numbers), "Ausgewogene 3:3 Verteilung gerade/ungerade"

    def strategy_odd_even_42(self):
        """Strategie 5: 4 ungerade, 2 gerade"""
        odd = [n for n in range(1, 50) if n % 2 == 1]
        even = [n for n in range(1, 50) if n % 2 == 0]
        numbers = random.sample(odd, 4) + random.sample(even, 2)
        return sorted(numbers), "4 ungerade, 2 gerade Zahlen"

    def strategy_sum_optimized(self):
        """Strategie 6: Optimierte Quersumme (Ziel: 150-180)"""
        target_sum = random.randint(150, 180)
        best_combo = None
        best_diff = float('inf')

        for _ in range(500):
            combo = sorted(random.sample(range(1, 50), 6))
            diff = abs(sum(combo) - target_sum)
            if diff < best_diff:
                best_diff = diff
                best_combo = combo

        return best_combo, f"Optimierte Summe ~{target_sum}"

    def strategy_decade_balance(self):
        """Strategie 7: Ausgewogene Zehnergruppen"""
        decades = {
            '1-9': list(range(1, 10)),
            '10-19': list(range(10, 20)),
            '20-29': list(range(20, 30)),
            '30-39': list(range(30, 40)),
            '40-49': list(range(40, 50))
        }

        numbers = []
        for _ in range(6):
            decade = random.choice(list(decades.keys()))
            available = [n for n in decades[decade] if n not in numbers]
            if available:
                numbers.append(random.choice(available))

        while len(numbers) < 6:
            n = random.randint(1, 49)
            if n not in numbers:
                numbers.append(n)

        return sorted(numbers[:6]), "Ausgewogene Dekaden-Verteilung"

    def strategy_delta_pattern(self):
        """Strategie 8: Delta-System (Abstände zwischen Zahlen)"""
        # Typische Delta-Werte aus historischen Daten
        deltas = [random.randint(1, 12) for _ in range(5)]

        start = random.randint(1, 15)
        numbers = [start]

        for delta in deltas:
            next_num = numbers[-1] + delta
            if next_num > 49:
                next_num = random.randint(1, 49)
            numbers.append(next_num)

        # Sicherstellen, dass alle unterschiedlich sind
        numbers = list(set(numbers))
        while len(numbers) < 6:
            n = random.randint(1, 49)
            if n not in numbers:
                numbers.append(n)

        return sorted(numbers[:6]), "Delta-System mit typischen Abständen"

    def strategy_position_based(self):
        """Strategie 9: Positionsbasierte Analyse"""
        positions = self.analysis.get('positions', {})
        numbers = []

        for pos in range(1, 7):
            pos_key = str(pos)
            if pos_key in positions and positions[pos_key].get('most_common'):
                top_nums = [n for n in positions[pos_key]['most_common'] if n not in numbers]
                if top_nums:
                    numbers.append(random.choice(top_nums[:5]))

        while len(numbers) < 6:
            n = random.randint(1, 49)
            if n not in numbers:
                numbers.append(n)

        return sorted(numbers[:6]), "Positionsbasierte häufigste Zahlen"

    def strategy_no_consecutive(self):
        """Strategie 10: Keine aufeinanderfolgenden Zahlen"""
        numbers = []
        available = list(range(1, 50))

        while len(numbers) < 6 and available:
            n = random.choice(available)
            numbers.append(n)
            # Entferne Nachbarn
            available = [x for x in available if abs(x - n) > 1]

        while len(numbers) < 6:
            n = random.randint(1, 49)
            if n not in numbers:
                numbers.append(n)

        return sorted(numbers[:6]), "Keine aufeinanderfolgenden Zahlen"

    def strategy_prime_mix(self):
        """Strategie 11: Primzahlen-Mix"""
        primes = self.PRIMES_49
        non_primes = [n for n in range(1, 50) if n not in primes]

        numbers = random.sample(primes, 3) + random.sample(non_primes, 3)
        return sorted(numbers), "Mix aus Primzahlen und Nicht-Primzahlen"

    def strategy_low_high(self):
        """Strategie 12: Niedrig-Hoch Balance (1-24 vs 25-49)"""
        low = list(range(1, 25))
        high = list(range(25, 50))

        numbers = random.sample(low, 3) + random.sample(high, 3)
        return sorted(numbers), "3 niedrige, 3 hohe Zahlen"

    def strategy_hot_cold_mix(self):
        """Strategie 13: Optimierter Hot-Cold-Mix"""
        hot = self.hot_numbers[:8]
        cold = self.cold_numbers[:6]
        overdue = self.overdue[:4]

        pool = list(set(hot + cold + overdue))
        if len(pool) >= 6:
            numbers = random.sample(pool, 6)
        else:
            numbers = pool + random.sample([n for n in range(1, 50) if n not in pool], 6 - len(pool))

        return sorted(numbers[:6]), "Optimierter Mix aus Hot, Cold und Overdue"

    def strategy_monte_carlo(self):
        """Strategie 14: Monte-Carlo Simulation"""
        simulations = 1000
        results = Counter()

        # Gewichtete Auswahl basierend auf Häufigkeit
        weights = [self.freq.get(n, 1) for n in range(1, 50)]
        total_weight = sum(weights)
        probs = [w / total_weight for w in weights]

        for _ in range(simulations):
            sample = []
            available = list(range(1, 50))
            for _ in range(6):
                idx = random.choices(range(len(available)),
                                    weights=[probs[available[i]-1] for i in range(len(available))])[0]
                sample.append(available[idx])
                available.pop(idx)
            for n in sample:
                results[n] += 1

        return sorted([n for n, _ in results.most_common(6)]), "Monte-Carlo Simulation (1000 Durchläufe)"

    def strategy_bayesian(self):
        """Strategie 15: Bayesian Inference"""
        # Prior: Gleichverteilung
        prior = {n: 1/49 for n in range(1, 50)}

        # Likelihood aus Häufigkeiten
        total = sum(self.freq.values()) or 1
        likelihood = {n: (self.freq.get(n, 0) + 1) / (total + 49) for n in range(1, 50)}

        # Posterior
        posterior = {n: prior[n] * likelihood[n] for n in range(1, 50)}
        total_post = sum(posterior.values())
        posterior = {n: p / total_post for n, p in posterior.items()}

        sorted_post = sorted(posterior.items(), key=lambda x: x[1], reverse=True)
        return sorted([n for n, _ in sorted_post[:6]]), "Bayesian Wahrscheinlichkeitsberechnung"

    def strategy_fibonacci(self):
        """Strategie 16: Fibonacci-Zahlen bevorzugt"""
        fib = [n for n in self.FIBONACCI if n <= 49]
        non_fib = [n for n in range(1, 50) if n not in fib]

        n_fib = random.randint(2, min(4, len(fib)))
        numbers = random.sample(fib, n_fib) + random.sample(non_fib, 6 - n_fib)
        return sorted(numbers), "Fibonacci-Zahlen bevorzugt"

    def strategy_neighbor_pairs(self):
        """Strategie 17: Häufige Nachbarpaare"""
        pairs = self.analysis.get('neighbors', {}).get('common_pairs', [])

        numbers = []
        if pairs:
            # Wähle 2-3 Paare
            selected_pairs = random.sample(pairs[:20], min(3, len(pairs[:20])))
            for pair in selected_pairs:
                if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                    for n in pair[:2]:
                        if n not in numbers and len(numbers) < 6:
                            numbers.append(n)

        while len(numbers) < 6:
            n = random.choice(self.hot_numbers) if self.hot_numbers else random.randint(1, 49)
            if n not in numbers:
                numbers.append(n)

        return sorted(numbers[:6]), "Basierend auf häufigen Nachbarpaaren"

    def strategy_end_digit(self):
        """Strategie 18: Endziffern-Balance"""
        end_digits = list(range(10))
        random.shuffle(end_digits)

        numbers = []
        for digit in end_digits[:6]:
            candidates = [n for n in range(1, 50) if n % 10 == digit and n not in numbers]
            if candidates:
                # Bevorzuge heiße Zahlen
                hot_candidates = [n for n in candidates if n in self.hot_numbers]
                if hot_candidates:
                    numbers.append(random.choice(hot_candidates))
                else:
                    numbers.append(random.choice(candidates))

        while len(numbers) < 6:
            n = random.randint(1, 49)
            if n not in numbers:
                numbers.append(n)

        return sorted(numbers[:6]), "Ausgewogene Endziffern-Verteilung"

    def strategy_neural_network(self):
        """Strategie 19: Simuliertes Neuronales Netz"""
        weights = {}
        for n in range(1, 50):
            w = 0
            if n in self.hot_numbers[:10]: w += 3.0
            elif n in self.hot_numbers[10:]: w += 1.5
            if n in self.cold_numbers: w += 0.5
            if n in self.overdue[:5]: w += 2.5
            elif n in self.overdue[5:]: w += 1.0
            # Sigmoid-ähnliche Aktivierung
            w = w / (1 + abs(w - 2))
            weights[n] = w + random.random() * 0.5

        sorted_nums = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        return sorted([n for n, _ in sorted_nums[:6]]), "Neuronales Netz Simulation"

    def strategy_lstm_sequence(self):
        """Strategie 20: LSTM-ähnliche Sequenzanalyse"""
        sequences = []
        for i in range(min(10, len(self.draws))):
            sequences.extend(self.draws[i].get('numbers', []))

        # Finde häufige Folgemuster
        seq_freq = Counter()
        for i in range(len(sequences) - 1):
            seq_freq[(sequences[i], sequences[i+1])] += 1

        predicted = set()
        for (a, b), _ in seq_freq.most_common(20):
            predicted.add(b)
            if len(predicted) >= 6:
                break

        result = list(predicted)[:6]
        while len(result) < 6:
            n = random.choice(self.hot_numbers) if self.hot_numbers else random.randint(1, 49)
            if n not in result:
                result.append(n)

        return sorted(result), "LSTM Sequenzanalyse"

    def strategy_random_forest(self):
        """Strategie 21: Simulierter Random Forest"""
        trees = []

        # Baum 1: Nur heiße Zahlen
        if len(self.hot_numbers) >= 6:
            trees.append(random.sample(self.hot_numbers, 6))

        # Baum 2: Mix heiß + kalt
        hot = self.hot_numbers[:10]
        cold = self.cold_numbers[:6] if len(self.cold_numbers) >= 6 else self.cold_numbers
        if len(hot) >= 4 and len(cold) >= 2:
            trees.append(random.sample(hot, 4) + random.sample(cold, 2))

        # Baum 3: Überfällige + heiß
        overdue = self.overdue[:8]
        if len(overdue) >= 3 and len(hot) >= 3:
            trees.append(random.sample(overdue, 3) + random.sample(hot, 3))

        # Voting
        if trees:
            votes = Counter()
            for tree in trees:
                votes.update(tree)
            return sorted([n for n, _ in votes.most_common(6)]), "Random Forest Ensemble"
        else:
            return sorted(random.sample(range(1, 50), 6)), "Random Forest (Fallback)"

    def get_all_strategies(self):
        """Gibt alle Strategien mit Gewichtung zurück"""
        strategies = [
            ('hot_cold', self.strategy_hot_cold),
            ('cold_numbers', self.strategy_cold_numbers),
            ('overdue', self.strategy_overdue),
            ('odd_even_33', self.strategy_odd_even_33),
            ('odd_even_42', self.strategy_odd_even_42),
            ('sum_optimized', self.strategy_sum_optimized),
            ('decade_balance', self.strategy_decade_balance),
            ('delta_pattern', self.strategy_delta_pattern),
            ('position_based', self.strategy_position_based),
            ('no_consecutive', self.strategy_no_consecutive),
            ('prime_mix', self.strategy_prime_mix),
            ('low_high', self.strategy_low_high),
            ('hot_cold_mix', self.strategy_hot_cold_mix),
            ('monte_carlo', self.strategy_monte_carlo),
            ('bayesian', self.strategy_bayesian),
            ('fibonacci', self.strategy_fibonacci),
            ('neighbor_pairs', self.strategy_neighbor_pairs),
            ('end_digit', self.strategy_end_digit),
            ('neural_network', self.strategy_neural_network),
            ('lstm_sequence', self.strategy_lstm_sequence),
            ('random_forest', self.strategy_random_forest)
        ]

        # Sortiere nach Gewichtung
        weighted = [(name, fn, self.weight_manager.get_weight(name)) for name, fn in strategies]
        weighted.sort(key=lambda x: x[2], reverse=True)

        return weighted

# =====================================================
# KI-API AUFRUFE
# =====================================================

def call_gemini_api(prompt, api_key):
    """Ruft Google Gemini API auf"""
    if not api_key:
        return None

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}"

    try:
        response = requests.post(url, json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 1000}
        }, timeout=30)

        if response.status_code == 200:
            data = response.json()
            return data.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text')
    except Exception as e:
        print(f"  ❌ Gemini Fehler: {e}")

    return None

def call_groq_api(prompt, api_key):
    """Ruft Groq API auf (ultraschnell)"""
    if not api_key:
        return None

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "llama-3.1-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 1000
            },
            timeout=30
        )

        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"  ❌ Groq Fehler: {e}")

    return None

def call_huggingface_api(prompt, api_key):
    """Ruft HuggingFace Inference API auf"""
    if not api_key:
        return None

    try:
        response = requests.post(
            "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"inputs": prompt, "parameters": {"max_new_tokens": 500}},
            timeout=60
        )

        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get('generated_text', '')
    except Exception as e:
        print(f"  ❌ HuggingFace Fehler: {e}")

    return None

def call_openrouter_api(prompt, api_key):
    """Ruft OpenRouter API auf"""
    if not api_key:
        return None

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://lotto-genius.github.io",
                "X-Title": "LottoGenius"
            },
            json={
                "model": "meta-llama/llama-3.1-8b-instruct:free",
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=30
        )

        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"  ❌ OpenRouter Fehler: {e}")

    return None

def call_together_api(prompt, api_key):
    """Ruft Together AI API auf"""
    if not api_key:
        return None

    try:
        response = requests.post(
            "https://api.together.xyz/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "meta-llama/Llama-3-70b-chat-hf",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1000
            },
            timeout=30
        )

        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"  ❌ Together Fehler: {e}")

    return None

def call_deepseek_api(prompt, api_key):
    """Ruft DeepSeek API auf"""
    if not api_key:
        return None

    try:
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1000
            },
            timeout=30
        )

        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"  ❌ DeepSeek Fehler: {e}")

    return None

# =====================================================
# TOP 10 AUSWAHL-ALGORITHMUS
# =====================================================

def calculate_prediction_score(prediction, analysis, sz_ranking):
    """
    Berechnet einen Qualitätsscore für eine Vorhersage.
    Höherer Score = bessere Vorhersage.
    """
    score = 0
    numbers = prediction.get('numbers', [])
    superzahl = prediction.get('superzahl', 0)

    # 1. Strategie-Gewicht (25%)
    strategy_weight = prediction.get('strategy_weight', 1.0)
    score += strategy_weight * 25

    # 2. Häufigkeitsbalance (20%)
    freq = analysis.get('frequency', {})
    hot = freq.get('hot_numbers', [])[:15]
    cold = freq.get('cold_numbers', [])[:15]
    hot_count = sum(1 for n in numbers if n in hot)
    cold_count = sum(1 for n in numbers if n in cold)
    # Ideal: 3-4 heiße, 1-2 kalte
    if 3 <= hot_count <= 4 and 1 <= cold_count <= 2:
        score += 20
    elif 2 <= hot_count <= 5:
        score += 10

    # 3. Summen-Optimierung (15%)
    total = sum(numbers)
    # Optimale Summe für 6aus49: 140-180
    if 140 <= total <= 180:
        score += 15
    elif 120 <= total <= 200:
        score += 8

    # 4. Gerade/Ungerade Balance (10%)
    odd_count = sum(1 for n in numbers if n % 2 == 1)
    if odd_count in [2, 3, 4]:  # Ideal: 2-4 ungerade
        score += 10
    elif odd_count in [1, 5]:
        score += 5

    # 5. Dekaden-Verteilung (10%)
    decades = set(n // 10 for n in numbers)
    if len(decades) >= 4:  # Mindestens 4 verschiedene Dekaden
        score += 10
    elif len(decades) >= 3:
        score += 5

    # 6. Keine Konsekutiven (10%)
    sorted_nums = sorted(numbers)
    has_consecutive = any(sorted_nums[i+1] - sorted_nums[i] == 1 for i in range(len(sorted_nums)-1))
    if not has_consecutive:
        score += 10
    else:
        score += 3

    # 7. Superzahl-Qualität (10%)
    for i, (sz, sz_score) in enumerate(sz_ranking[:5]):
        if superzahl == sz:
            score += 10 - i * 2
            break

    # 8. Provider-Score (Optional)
    provider_bonus = prediction.get('provider_score', 0)
    score += provider_bonus

    return round(score, 2)

def select_top_10_predictions(all_predictions, analysis, sz_ranking):
    """
    Wählt die 10 besten Vorhersagen basierend auf Qualitäts-Score.
    """
    # Berechne Score für jede Vorhersage
    scored_predictions = []
    for pred in all_predictions:
        score = calculate_prediction_score(pred, analysis, sz_ranking)
        pred_copy = pred.copy()
        pred_copy['quality_score'] = score
        scored_predictions.append(pred_copy)

    # Sortiere nach Score
    scored_predictions.sort(key=lambda x: x['quality_score'], reverse=True)

    # Entferne Duplikate (gleiche Zahlen)
    seen = set()
    unique_predictions = []
    for pred in scored_predictions:
        key = tuple(sorted(pred['numbers']))
        if key not in seen:
            seen.add(key)
            unique_predictions.append(pred)

    # Gib Top 10 zurück
    return unique_predictions[:10]

# =====================================================
# HAUPTVORHERSAGE-FUNKTION
# =====================================================

def generate_predictions():
    """Hauptfunktion: Generiert Vorhersagen mit allen KI-Systemen und wählt TOP 10"""

    print("=" * 60)
    print("🍀 LottoGenius - VOLLSTÄNDIGES Multi-KI System mit Selbstlernen")
    print("=" * 60)
    print(f"📅 {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
    print()

    # Lade Daten
    lotto_data = load_json('lotto_data.json', {'draws': []})
    analysis = load_json('analysis.json', {})
    predictions = load_json('predictions.json', {'predictions': [], 'history': []})
    provider_scores = load_json('provider_scores.json', {})

    draws = lotto_data.get('draws', [])
    if not draws:
        print("⚠️ Keine Lotto-Daten vorhanden!")
        return

    print(f"📊 Analysiere {len(draws)} historische Ziehungen...")

    # Initialisiere Weight Manager für Selbstlernen
    weight_manager = StrategyWeightManager()

    # Archiviere alte Vorhersagen
    if predictions.get('predictions'):
        predictions['history'].extend(predictions['predictions'])
        predictions['history'] = predictions['history'][-500:]

    # Initialisiere Superzahl-Analyzer
    sz_analyzer = SuperzahlAnalyzer(draws)
    best_sz, sz_ranking = sz_analyzer.predict_best_superzahl()

    print(f"\n🎯 Superzahl-Analyse (6-Faktoren):")
    print(f"   Beste Superzahl: {best_sz}")
    print(f"   Top 3: {sz_ranking[:3]}")

    # Initialisiere lokale Strategien
    local_strategies = LocalStrategies(draws, analysis, weight_manager)

    all_predictions = []
    ki_results = {}

    # ===== 1. LOKALE STRATEGIEN MIT GEWICHTUNG =====
    print("\n🖥️ Lokale Strategien (21 Methoden mit Selbstlernen):")

    weighted_strategies = local_strategies.get_all_strategies()

    for idx, (strategy_name, strategy_fn, weight) in enumerate(weighted_strategies):
        try:
            numbers, description = strategy_fn()
            sz_idx = idx % len(sz_ranking)
            sz = sz_ranking[sz_idx][0]

            all_predictions.append({
                'numbers': sorted(numbers)[:6],
                'superzahl': sz,
                'method': strategy_name,
                'method_name': f'🎯 {strategy_name}',
                'provider': 'local_strategy',
                'strategy': description,
                'strategy_weight': weight,
                'confidence': 50 + weight * 10 + random.random() * 10,
                'timestamp': datetime.now().isoformat(),
                'verified': False
            })

            weight_str = f"[{weight:.2f}]"
            print(f"   ✅ {strategy_name} {weight_str}: {sorted(numbers)[:6]} | SZ: {sz}")
            ki_results['local_strategies'] = True

        except Exception as e:
            print(f"   ❌ {strategy_name}: {e}")

    # ===== 2. EXTERNE KI-APIS =====
    print("\n🌐 Externe KI-APIs:")

    # Erstelle KI-Prompt
    hot_numbers = local_strategies.hot_numbers[:10]
    cold_numbers = local_strategies.cold_numbers[:10]
    overdue = local_strategies.overdue[:10]
    last_draw = draws[0]

    ki_prompt = f"""Du bist ein Lotto-Analyse-Experte. Analysiere diese Daten für Lotto 6 aus 49:

LETZTE ZIEHUNG: {last_draw['date']} - Zahlen: {last_draw.get('numbers', [])}, Superzahl: {last_draw.get('superzahl', 0)}

STATISTIK (basierend auf {len(draws)} Ziehungen):
- Heiße Zahlen (häufig): {hot_numbers}
- Kalte Zahlen (selten): {cold_numbers}
- Überfällige Zahlen: {overdue}
- Beste Superzahl laut Analyse: {best_sz}

Generiere 3 verschiedene Tipps. Antworte NUR mit diesem JSON-Format:
{{
  "predictions": [
    {{"numbers": [1,2,3,4,5,6], "superzahl": 0, "strategy": "Beschreibung", "confidence": 75}},
    {{"numbers": [7,8,9,10,11,12], "superzahl": 1, "strategy": "Beschreibung", "confidence": 70}},
    {{"numbers": [13,14,15,16,17,18], "superzahl": 2, "strategy": "Beschreibung", "confidence": 68}}
  ]
}}"""

    ki_apis = [
        ('gemini', '🔮 Google Gemini', call_gemini_api),
        ('groq', '⚡ Groq', call_groq_api),
        ('huggingface', '🤗 HuggingFace', call_huggingface_api),
        ('openrouter', '🌐 OpenRouter', call_openrouter_api),
        ('together', '🚀 Together AI', call_together_api),
        ('deepseek', '🧠 DeepSeek', call_deepseek_api)
    ]

    for provider_id, provider_name, api_fn in ki_apis:
        api_key = get_api_key(provider_id)

        if not api_key:
            print(f"   ⏭️ {provider_name}: Kein API-Key")
            continue

        try:
            result = api_fn(ki_prompt, api_key)

            if result:
                # Parse JSON aus Antwort
                try:
                    import re
                    json_match = re.search(r'\{[\s\S]*\}', result)
                    if json_match:
                        parsed = json.loads(json_match.group())
                        if 'predictions' in parsed:
                            count = 0
                            for pred in parsed['predictions']:
                                nums = pred.get('numbers', [])[:6]
                                if len(nums) == 6 and all(1 <= n <= 49 for n in nums):
                                    all_predictions.append({
                                        'numbers': sorted(nums),
                                        'superzahl': pred.get('superzahl', best_sz) % 10,
                                        'method': f'{provider_id}_ki',
                                        'method_name': f'{provider_name} KI',
                                        'provider': provider_id,
                                        'strategy': pred.get('strategy', ''),
                                        'strategy_weight': 1.5,  # KI-Bonus
                                        'confidence': pred.get('confidence', 70),
                                        'timestamp': datetime.now().isoformat(),
                                        'verified': False
                                    })
                                    count += 1
                            print(f"   ✅ {provider_name}: {count} Vorhersagen")
                            ki_results[provider_id] = True
                            continue
                except:
                    pass

                print(f"   ⚠️ {provider_name}: Konnte Antwort nicht parsen")
            else:
                print(f"   ❌ {provider_name}: Keine Antwort")

        except Exception as e:
            print(f"   ❌ {provider_name}: {e}")

    # ===== PROVIDER-STATUS SPEICHERN =====
    provider_status = load_json('provider_status.json', {'providers': {}})
    provider_status['last_check'] = datetime.now().isoformat()

    for provider_id, provider_name, _ in ki_apis:
        api_key = get_api_key(provider_id)
        provider_info = KI_PROVIDERS.get(provider_id, {})

        if provider_id not in provider_status['providers']:
            provider_status['providers'][provider_id] = {
                'name': provider_info.get('name', provider_name),
                'icon': provider_info.get('emoji', '🤖'),
                'status': 'unknown',
                'last_success': None,
                'error': None
            }

        if not api_key:
            provider_status['providers'][provider_id]['status'] = 'no_key'
            provider_status['providers'][provider_id]['error'] = 'Kein API-Key konfiguriert'
        elif provider_id in ki_results:
            provider_status['providers'][provider_id]['status'] = 'online'
            provider_status['providers'][provider_id]['last_success'] = datetime.now().isoformat()
            provider_status['providers'][provider_id]['error'] = None
        else:
            provider_status['providers'][provider_id]['status'] = 'error'
            provider_status['providers'][provider_id]['error'] = 'API-Aufruf fehlgeschlagen'

    # Lokale Strategien immer online
    provider_status['providers']['local_strategies'] = {
        'name': 'Lokale Strategien (21)',
        'icon': '🖥️',
        'status': 'online',
        'last_success': datetime.now().isoformat(),
        'error': None
    }

    save_json('provider_status.json', provider_status)

    # ===== 3. ENSEMBLE-VOTING =====
    print("\n🏆 Ensemble-Voting:")

    if len(all_predictions) >= 5:
        all_numbers = Counter()
        all_superzahlen = Counter()

        for pred in all_predictions:
            # Gewichtetes Voting
            weight = pred.get('strategy_weight', 1.0)
            for num in pred['numbers']:
                all_numbers[num] += weight
            all_superzahlen[pred['superzahl']] += weight

        # Champion-Tipp: Zahlen mit meisten Stimmen
        top_voted = [n for n, _ in all_numbers.most_common(6)]
        top_sz = all_superzahlen.most_common(1)[0][0]

        all_predictions.insert(0, {
            'numbers': sorted(top_voted),
            'superzahl': top_sz,
            'method': 'ensemble_champion',
            'method_name': '🏆 CHAMPION (Alle KIs)',
            'provider': 'ensemble',
            'strategy': f'Gewichtetes Voting aus {len(all_predictions)} Vorhersagen',
            'strategy_weight': 3.0,  # Höchste Gewichtung
            'confidence': 95,
            'timestamp': datetime.now().isoformat(),
            'verified': False,
            'is_champion': True
        })

        print(f"   ✅ Champion-Tipp: {sorted(top_voted)} | SZ: {top_sz}")
        print(f"   📊 Basiert auf {len(all_predictions)-1} Vorhersagen")

    # ===== 4. TOP 10 AUSWAHL =====
    print("\n🏅 Wähle TOP 10 beste Tipps...")

    top_10 = select_top_10_predictions(all_predictions, analysis, sz_ranking)

    print(f"\n{'='*60}")
    print("🏅 TOP 10 BESTE TIPPS FÜR LOTTO 6AUS49:")
    print(f"{'='*60}")

    for i, pred in enumerate(top_10, 1):
        nums = pred['numbers']
        sz = pred['superzahl']
        score = pred.get('quality_score', 0)
        method = pred.get('method', 'unknown')
        print(f"   {i:2}. {nums} + SZ {sz}  (Score: {score:.1f}, Methode: {method})")

    # ===== SPEICHERN =====

    # Berechne nächste Ziehung
    now = datetime.now()
    days_to_wed = (2 - now.weekday() + 7) % 7
    days_to_sat = (5 - now.weekday() + 7) % 7
    if days_to_wed == 0 and now.hour >= 19:
        days_to_wed = 7
    if days_to_sat == 0 and now.hour >= 20:
        days_to_sat = 7
    next_days = min(days_to_wed if days_to_wed > 0 else 7, days_to_sat if days_to_sat > 0 else 7)
    next_draw = now + timedelta(days=next_days)
    next_draw_str = next_draw.strftime('%d.%m.%Y')

    # Speichere nur Top 10 als aktive Vorhersagen
    predictions['predictions'] = top_10
    predictions['all_predictions'] = all_predictions  # Alle für Analyse
    predictions['last_update'] = datetime.now().isoformat()
    predictions['next_draw'] = next_draw_str
    predictions['ki_stats'] = {
        'providers_used': list(ki_results.keys()),
        'total_generated': len(all_predictions),
        'top_10_selected': len(top_10),
        'best_superzahl': best_sz,
        'superzahl_ranking': [(sz, round(score, 1)) for sz, score in sz_ranking[:5]],
        'superzahl_patterns': sz_analyzer.patterns,
        'strategy_weights': weight_manager.weights.get('strategies', {})
    }

    save_json('predictions.json', predictions)

    # Speichere Superzahl-Historie
    sz_history = load_json('superzahl_history.json', {'history': [], 'entries': []})
    if 'history' not in sz_history:
        sz_history['history'] = sz_history.get('entries', [])
    sz_history['history'].append({
        'date': datetime.now().isoformat(),
        'predicted': best_sz,
        'ranking': [(sz, round(score, 1)) for sz, score in sz_ranking[:5]],
        'actual': None  # Wird später vom learn.py gefüllt
    })
    sz_history['history'] = sz_history['history'][-100:]
    save_json('superzahl_history.json', sz_history)

    print(f"\n{'='*60}")
    print(f"✅ TOP 10 Vorhersagen generiert!")
    print(f"📊 Insgesamt {len(all_predictions)} Tipps analysiert")
    print(f"📅 Nächste Ziehung: {next_draw_str}")
    print(f"🎯 Beste Superzahl: {best_sz}")
    print(f"🤖 KI-Provider verwendet: {len(ki_results)}")
    print(f"📈 Strategien mit Selbstlernen: {len(weighted_strategies)}")
    print(f"{'='*60}")

if __name__ == "__main__":
    generate_predictions()
