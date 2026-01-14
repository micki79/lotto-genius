#!/usr/bin/env python3
"""
🌟 LottoGenius - EUROJACKPOT Multi-KI Vorhersage-System

Integriert 7 kostenlose KI-APIs + 6 echte ML-Algorithmen + 15 lokale Strategien:

EXTERNE KI-APIs:
1. Google Gemini (1M Tokens/Tag)
2. Groq (ultraschnell)
3. HuggingFace (Mixtral-8x7B)
4. OpenRouter (50+ Modelle)
5. Together AI ($25 Startguthaben)
6. DeepSeek (komplett kostenlos)

ECHTE ML-ALGORITHMEN:
1. Neural Network (Backpropagation, 50→64→32→50)
2. Markov Chain (Übergangswahrscheinlichkeiten)
3. Bayesian Predictor (Thompson Sampling)
4. Reinforcement Learner (Q-Learning)
5. Eurozahl ML (Spezialisiert auf 2 aus 12)
6. Ensemble ML (Kombiniert alle Modelle)

SELBSTLERNENDES SYSTEM:
- Gewichtet Strategien basierend auf historischer Performance
- Q-Learning lernt optimale Strategie-Auswahl
- Eurozahl ML analysiert Paar-Muster
- Wählt die besten 8 Vorhersagen basierend auf Erfolgsquote
"""
import json
import os
import sys
from datetime import datetime, timedelta
import random
from collections import Counter
import requests
import math

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

# Importiere echte ML-Modelle (6 Algorithmen)
try:
    from ml_models import (
        get_eurojackpot_ml_predictions,
        train_eurojackpot_ml,
        EurojackpotEnsembleML,
        EurojackpotReinforcementLearner,
        EurozahlML
    )
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("⚠️ ML-Modelle nicht verfügbar")

# Primzahlen bis 50
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

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
    """Holt API-Key aus Environment Variable"""
    env_key = KI_PROVIDERS.get(provider, {}).get('env_key')
    if env_key:
        return os.environ.get(env_key)
    return None

# =====================================================
# EUROZAHL-ANALYSE (6-FAKTOREN-ALGORITHMUS)
# =====================================================

class EurozahlAnalyzer:
    """
    Analysiert Eurozahl-Muster mit 6 verschiedenen Faktoren:
    1. Häufigkeit (20%) - Wie oft wurde jede Eurozahl gezogen?
    2. Trend (25%) - Ist sie aktuell "heiß" oder "kalt"?
    3. Wochentag (15%) - Unterschiede Dienstag vs Freitag
    4. Lücke (20%) - Wie lange nicht gezogen (überfällig)?
    5. Folge-Muster (15%) - Welche Eurozahl-Paar kommt nach welchem?
    6. Anti-Serie (5%) - Vermeidet Wiederholungen
    """

    def __init__(self, draws):
        self.draws = draws
        self.patterns = {}
        if draws:
            self.analyze_all_patterns()

    def analyze_all_patterns(self):
        """Analysiert alle 6 Faktoren für Eurozahlen"""
        # 1. Häufigkeit
        euro_freq = Counter()
        for d in self.draws:
            euro_freq.update(d['eurozahlen'])
        self.patterns['frequency'] = dict(euro_freq)

        # 2. Lücken
        euro_gaps = {}
        for ez in range(1, 13):
            for i, d in enumerate(self.draws):
                if ez in d['eurozahlen']:
                    euro_gaps[ez] = i
                    break
            else:
                euro_gaps[ez] = len(self.draws)
        self.patterns['gaps'] = euro_gaps

        # 3. Trend (letzte 20 vs vorherige 20)
        recent = self.draws[:20]
        older = self.draws[20:40]
        recent_freq = Counter()
        older_freq = Counter()
        for d in recent:
            recent_freq.update(d['eurozahlen'])
        for d in older:
            older_freq.update(d['eurozahlen'])

        trends = {}
        for ez in range(1, 13):
            r = recent_freq.get(ez, 0)
            o = older_freq.get(ez, 0)
            trends[ez] = r - o
        self.patterns['trends'] = trends

        # 4. Wochentag-Muster
        tue_freq = Counter()
        fri_freq = Counter()
        for d in self.draws[:100]:
            try:
                day, month, year = map(int, d['date'].split('.'))
                date_obj = datetime(year, month, day)
                if date_obj.weekday() == 1:  # Dienstag
                    tue_freq.update(d['eurozahlen'])
                elif date_obj.weekday() == 4:  # Freitag
                    fri_freq.update(d['eurozahlen'])
            except:
                pass
        self.patterns['tuesday'] = dict(tue_freq)
        self.patterns['friday'] = dict(fri_freq)

        # 5. Folge-Muster (welches Paar folgt welchem)
        follows = Counter()
        for i in range(len(self.draws) - 1):
            current = tuple(sorted(self.draws[i]['eurozahlen']))
            previous = tuple(sorted(self.draws[i + 1]['eurozahlen']))
            follows[(previous, current)] += 1
        self.patterns['follows'] = {f"{k[0]}->{k[1]}": v for k, v in follows.most_common(30)}

        # 6. Letztes Eurozahlen-Paar (für Anti-Serie)
        self.patterns['last_euro'] = tuple(sorted(self.draws[0]['eurozahlen'])) if self.draws else None

    def predict_best_eurozahlen(self):
        """Berechnet die besten 2 Eurozahlen basierend auf allen 6 Faktoren"""
        if not self.patterns:
            return sorted(random.sample(range(1, 13), 2)), [(i, 10) for i in range(1, 13)]

        scores = {}

        freq = self.patterns.get('frequency', {})
        gaps = self.patterns.get('gaps', {})
        trends = self.patterns.get('trends', {})
        last_euro = self.patterns.get('last_euro')

        # Normalisierung
        max_freq = max(freq.values()) if freq else 1
        max_gap = max(gaps.values()) if gaps else 1
        max_trend = max(abs(v) for v in trends.values()) if trends and any(trends.values()) else 1

        # Bestimme Wochentag für die nächste Ziehung
        today = datetime.now()
        days_to_tue = (1 - today.weekday() + 7) % 7
        days_to_fri = (4 - today.weekday() + 7) % 7
        is_tuesday = days_to_tue < days_to_fri or (days_to_tue == 0 and today.hour < 21)

        day_freq = self.patterns.get('tuesday' if is_tuesday else 'friday', {})
        max_day = max(day_freq.values()) if day_freq else 1

        for ez in range(1, 13):
            score = 0

            # 1. Häufigkeit (20%)
            score += (freq.get(ez, 0) / max_freq) * 20

            # 2. Trend (25%)
            trend_val = trends.get(ez, 0)
            normalized_trend = (trend_val + max_trend) / (2 * max_trend) if max_trend > 0 else 0.5
            score += normalized_trend * 25

            # 3. Wochentag (15%)
            score += (day_freq.get(ez, 0) / max_day) * 15

            # 4. Lücke (20%)
            score += (gaps.get(ez, 0) / max_gap) * 20

            # 5. Folge-Muster (15%) - vereinfacht
            score += random.random() * 15  # Zufallskomponente für Diversität

            # 6. Anti-Serie (5%)
            if last_euro and ez not in last_euro:
                score += 5

            scores[ez] = round(score, 2)

        # Sortiere und gib Ranking zurück
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_two = sorted([ranked[0][0], ranked[1][0]])

        return best_two, ranked

# =====================================================
# SELBSTLERNENDES STRATEGIE-SYSTEM
# =====================================================

class StrategyWeightManager:
    """
    Verwaltet Strategie-Gewichte basierend auf historischer Performance.
    Strategien mit mehr Treffern bekommen höhere Gewichte.
    """

    def __init__(self):
        self.weights = load_json('eurojackpot_strategy_weights.json', {
            'strategies': {},
            'last_update': None,
            'total_predictions': 0
        })

    def get_weight(self, strategy_id):
        """Gibt das Gewicht einer Strategie zurück (Standard: 1.0)"""
        return self.weights['strategies'].get(strategy_id, {}).get('weight', 1.0)

    def update_weight(self, strategy_id, matches, euro_matches):
        """Aktualisiert das Gewicht basierend auf Treffern"""
        if strategy_id not in self.weights['strategies']:
            self.weights['strategies'][strategy_id] = {
                'weight': 1.0,
                'total_predictions': 0,
                'total_matches': 0,
                'total_euro_matches': 0,
                'wins_3plus': 0,
                'wins_4plus': 0,
                'wins_5': 0
            }

        s = self.weights['strategies'][strategy_id]
        s['total_predictions'] += 1
        s['total_matches'] += matches
        s['total_euro_matches'] += euro_matches

        if matches >= 3:
            s['wins_3plus'] += 1
        if matches >= 4:
            s['wins_4plus'] += 1
        if matches == 5:
            s['wins_5'] += 1

        # Berechne neues Gewicht basierend auf Durchschnitt
        avg_matches = s['total_matches'] / s['total_predictions']
        avg_euro = s['total_euro_matches'] / s['total_predictions']

        # Gewicht = Durchschnittliche Treffer + Euro-Bonus
        s['weight'] = round(0.5 + avg_matches * 0.3 + avg_euro * 0.2 + (s['wins_3plus'] / s['total_predictions']) * 0.5, 3)

        self.weights['last_update'] = datetime.now().isoformat()
        self.weights['total_predictions'] += 1

    def get_top_strategies(self, n=10):
        """Gibt die Top-N Strategien nach Gewicht zurück"""
        sorted_strategies = sorted(
            self.weights['strategies'].items(),
            key=lambda x: x[1].get('weight', 1.0),
            reverse=True
        )
        return sorted_strategies[:n]

    def save(self):
        """Speichert die Gewichte"""
        save_json('eurojackpot_strategy_weights.json', self.weights)

# =====================================================
# LOKALE ML-MODELLE (15 STRATEGIEN)
# =====================================================

class EurojackpotMLModels:
    """
    Implementiert 15 verschiedene ML-/Statistik-Strategien für Eurojackpot.
    Jede Strategie hat eine ID für das Selbstlernsystem.
    """

    def __init__(self, draws, analysis):
        self.draws = draws
        self.analysis = analysis
        self.freq = analysis.get('frequency', {})
        self.gaps = analysis.get('gaps', {})
        self.weight_manager = StrategyWeightManager()

    def get_hot_numbers(self, n=15):
        """Heißeste Hauptzahlen"""
        hot = self.freq.get('hot_numbers', [])
        if not hot:
            all_nums = []
            for d in self.draws[:50]:
                all_nums.extend(d['numbers'])
            hot = [num for num, _ in Counter(all_nums).most_common(n)]
        return hot[:n]

    def get_cold_numbers(self, n=10):
        """Kälteste Hauptzahlen"""
        cold = self.freq.get('cold_numbers', [])
        if not cold:
            all_nums = []
            for d in self.draws[:200]:
                all_nums.extend(d['numbers'])
            freq = Counter(all_nums)
            cold = [num for num, _ in freq.most_common()[-n:]]
        return cold[:n]

    def get_overdue_numbers(self, n=10):
        """Überfällige Zahlen"""
        overdue = self.gaps.get('overdue_numbers', [])
        if not overdue:
            gaps_dict = {}
            for num in range(1, 51):
                for i, d in enumerate(self.draws):
                    if num in d['numbers']:
                        gaps_dict[num] = i
                        break
                else:
                    gaps_dict[num] = len(self.draws)
            overdue = [n for n, _ in sorted(gaps_dict.items(), key=lambda x: x[1], reverse=True)[:n]]
        return overdue[:n]

    def select_unique(self, pool, count=5):
        """Wählt einzigartige Zahlen aus einem Pool"""
        pool = [n for n in pool if 1 <= n <= 50]
        if len(pool) < count:
            pool = list(range(1, 51))
        selected = random.sample(pool, min(count, len(pool)))
        while len(selected) < count:
            n = random.randint(1, 50)
            if n not in selected:
                selected.append(n)
        return sorted(selected)

    def select_eurozahlen(self, hot_euro=None):
        """Wählt 2 Eurozahlen"""
        if hot_euro and len(hot_euro) >= 2:
            return sorted(random.sample(hot_euro[:6], 2))
        return sorted(random.sample(range(1, 13), 2))

    # === STRATEGIE 1: Hot Numbers ===
    def strategy_hot_cold(self):
        """🔥 Hot Numbers - Häufigste Zahlen"""
        hot = self.get_hot_numbers(20)
        hot_euro = self.freq.get('hot_eurozahlen', list(range(1, 13)))

        return {
            'numbers': self.select_unique(hot, 5),
            'eurozahlen': self.select_eurozahlen(hot_euro),
            'strategy_id': 'ej_hot_cold',
            'strategy_name': '🔥 Hot Numbers',
            'description': 'Basiert auf den häufigsten Zahlen',
            'weight': self.weight_manager.get_weight('ej_hot_cold')
        }

    # === STRATEGIE 2: Cold Numbers ===
    def strategy_cold_numbers(self):
        """❄️ Cold Numbers - Seltene Zahlen"""
        cold = self.get_cold_numbers(15)
        cold_euro = self.freq.get('cold_eurozahlen', list(range(1, 13)))

        return {
            'numbers': self.select_unique(cold, 5),
            'eurozahlen': self.select_eurozahlen(cold_euro),
            'strategy_id': 'ej_cold',
            'strategy_name': '❄️ Cold Numbers',
            'description': 'Seltener gezogene Zahlen',
            'weight': self.weight_manager.get_weight('ej_cold')
        }

    # === STRATEGIE 3: Überfällige Zahlen ===
    def strategy_overdue(self):
        """⏰ Überfällige Zahlen"""
        overdue = self.get_overdue_numbers(15)
        overdue_euro = self.gaps.get('overdue_eurozahlen', list(range(1, 13)))

        return {
            'numbers': self.select_unique(overdue, 5),
            'eurozahlen': self.select_eurozahlen(overdue_euro),
            'strategy_id': 'ej_overdue',
            'strategy_name': '⏰ Überfällige Zahlen',
            'description': 'Zahlen die lange nicht kamen',
            'weight': self.weight_manager.get_weight('ej_overdue')
        }

    # === STRATEGIE 4: 3 Ungerade / 2 Gerade ===
    def strategy_odd_even_32(self):
        """⚖️ 3 Ungerade / 2 Gerade Balance"""
        odd_nums = [n for n in range(1, 51) if n % 2 == 1]
        even_nums = [n for n in range(1, 51) if n % 2 == 0]

        hot = set(self.get_hot_numbers(25))
        odd_hot = [n for n in odd_nums if n in hot]
        even_hot = [n for n in even_nums if n in hot]

        selected = random.sample(odd_hot or odd_nums, 3) + random.sample(even_hot or even_nums, 2)

        return {
            'numbers': sorted(selected),
            'eurozahlen': self.select_eurozahlen(),
            'strategy_id': 'ej_odd_even_32',
            'strategy_name': '⚖️ 3-Ungerade/2-Gerade',
            'description': 'Optimale Gerade/Ungerade-Balance',
            'weight': self.weight_manager.get_weight('ej_odd_even_32')
        }

    # === STRATEGIE 5: 2 Ungerade / 3 Gerade ===
    def strategy_odd_even_23(self):
        """⚖️ 2 Ungerade / 3 Gerade Balance"""
        odd_nums = [n for n in range(1, 51) if n % 2 == 1]
        even_nums = [n for n in range(1, 51) if n % 2 == 0]

        hot = set(self.get_hot_numbers(25))
        odd_hot = [n for n in odd_nums if n in hot]
        even_hot = [n for n in even_nums if n in hot]

        selected = random.sample(odd_hot or odd_nums, 2) + random.sample(even_hot or even_nums, 3)

        return {
            'numbers': sorted(selected),
            'eurozahlen': self.select_eurozahlen(),
            'strategy_id': 'ej_odd_even_23',
            'strategy_name': '⚖️ 2-Ungerade/3-Gerade',
            'description': 'Alternative Gerade/Ungerade-Balance',
            'weight': self.weight_manager.get_weight('ej_odd_even_23')
        }

    # === STRATEGIE 6: Summen-Optimierung ===
    def strategy_sum_optimized(self):
        """📊 Summen-Optimierung (Ziel: 95-160)"""
        target_sum = random.randint(110, 145)
        hot = self.get_hot_numbers(30)

        best_selection = None
        best_diff = float('inf')

        for _ in range(100):
            selection = random.sample(hot or list(range(1, 51)), 5)
            current_sum = sum(selection)
            diff = abs(current_sum - target_sum)

            if diff < best_diff:
                best_diff = diff
                best_selection = selection

        return {
            'numbers': sorted(best_selection or random.sample(range(1, 51), 5)),
            'eurozahlen': self.select_eurozahlen(),
            'strategy_id': 'ej_sum_optimized',
            'strategy_name': f'📊 Summen-Optimierung (Σ={sum(best_selection) if best_selection else 0})',
            'description': 'Optimiert auf typische Summen',
            'weight': self.weight_manager.get_weight('ej_sum_optimized')
        }

    # === STRATEGIE 7: Dekaden-Balance ===
    def strategy_decade_balance(self):
        """📈 Dekaden-Balance"""
        decades = {
            0: list(range(1, 11)),
            1: list(range(11, 21)),
            2: list(range(21, 31)),
            3: list(range(31, 41)),
            4: list(range(41, 51))
        }

        hot = set(self.get_hot_numbers(30))
        selected = []

        # Wähle aus verschiedenen Dekaden
        decade_picks = [0, 1, 2, 3, 4]
        random.shuffle(decade_picks)

        for decade in decade_picks[:4]:
            decade_nums = [n for n in decades[decade] if n in hot] or decades[decade]
            selected.append(random.choice(decade_nums))

        # 5. Zahl aus beliebiger Dekade
        remaining = [n for n in range(1, 51) if n not in selected]
        selected.append(random.choice([n for n in remaining if n in hot] or remaining))

        return {
            'numbers': sorted(selected[:5]),
            'eurozahlen': self.select_eurozahlen(),
            'strategy_id': 'ej_decade_balance',
            'strategy_name': '📈 Dekaden-Balance',
            'description': 'Verteilt über alle 10er-Gruppen',
            'weight': self.weight_manager.get_weight('ej_decade_balance')
        }

    # === STRATEGIE 8: Delta-Muster ===
    def strategy_delta_pattern(self):
        """🔢 Delta-Muster"""
        delta_info = self.analysis.get('delta', {})
        common_deltas = [d for d, _ in delta_info.get('common_deltas', [(8, 1), (6, 1), (10, 1)])]

        if not common_deltas:
            common_deltas = [5, 7, 9, 11, 13]

        # Starte mit einer Zahl und addiere Deltas
        start = random.randint(3, 15)
        selected = [start]

        for _ in range(4):
            delta = random.choice(common_deltas)
            next_num = selected[-1] + delta
            if 1 <= next_num <= 50 and next_num not in selected:
                selected.append(next_num)
            else:
                # Fallback
                available = [n for n in range(1, 51) if n not in selected]
                if available:
                    selected.append(random.choice(available))

        return {
            'numbers': sorted(selected[:5]),
            'eurozahlen': self.select_eurozahlen(),
            'strategy_id': 'ej_delta',
            'strategy_name': '🔢 Delta-Muster',
            'description': 'Basiert auf typischen Abständen',
            'weight': self.weight_manager.get_weight('ej_delta')
        }

    # === STRATEGIE 9: Positions-Optimierung ===
    def strategy_position_based(self):
        """📍 Positions-Optimierung"""
        pos_info = self.analysis.get('positions', {}).get('recommended_ranges', {})

        if not pos_info:
            pos_info = {1: [1, 15], 2: [8, 25], 3: [18, 35], 4: [28, 43], 5: [38, 50]}

        selected = []
        for pos in range(1, 6):
            range_info = pos_info.get(pos, pos_info.get(str(pos), [1, 50]))
            low, high = range_info
            available = [n for n in range(low, high + 1) if n not in selected]
            if available:
                selected.append(random.choice(available))

        # Falls weniger als 5, auffüllen
        while len(selected) < 5:
            n = random.randint(1, 50)
            if n not in selected:
                selected.append(n)

        return {
            'numbers': sorted(selected[:5]),
            'eurozahlen': self.select_eurozahlen(),
            'strategy_id': 'ej_position',
            'strategy_name': '📍 Positions-Optimierung',
            'description': 'Typische Bereiche pro Position',
            'weight': self.weight_manager.get_weight('ej_position')
        }

    # === STRATEGIE 10: Keine Konsekutiven ===
    def strategy_no_consecutive(self):
        """🔀 Keine Folge-Zahlen"""
        hot = self.get_hot_numbers(30)
        selected = []

        for num in hot:
            if not any(abs(num - s) == 1 for s in selected):
                selected.append(num)
                if len(selected) == 5:
                    break

        while len(selected) < 5:
            n = random.randint(1, 50)
            if n not in selected and not any(abs(n - s) == 1 for s in selected):
                selected.append(n)

        return {
            'numbers': sorted(selected[:5]),
            'eurozahlen': self.select_eurozahlen(),
            'strategy_id': 'ej_no_consecutive',
            'strategy_name': '🔀 Keine Folge-Zahlen',
            'description': 'Vermeidet aufeinanderfolgende',
            'weight': self.weight_manager.get_weight('ej_no_consecutive')
        }

    # === STRATEGIE 11: Primzahlen-Mix ===
    def strategy_prime_mix(self):
        """🔢 Primzahlen-Mix"""
        primes_in_range = [p for p in PRIMES if p <= 50]
        non_primes = [n for n in range(1, 51) if n not in primes_in_range]

        # 2 Primzahlen + 3 Nicht-Primzahlen
        selected = random.sample(primes_in_range, 2) + random.sample(non_primes, 3)

        return {
            'numbers': sorted(selected),
            'eurozahlen': self.select_eurozahlen(),
            'strategy_id': 'ej_prime_mix',
            'strategy_name': '🔢 Primzahlen-Mix',
            'description': '2 Primzahlen + 3 andere',
            'weight': self.weight_manager.get_weight('ej_prime_mix')
        }

    # === STRATEGIE 12: Low/High Balance ===
    def strategy_low_high(self):
        """📉📈 Low/High Balance (3:2)"""
        low = list(range(1, 26))
        high = list(range(26, 51))

        hot = set(self.get_hot_numbers(30))
        low_hot = [n for n in low if n in hot] or low
        high_hot = [n for n in high if n in hot] or high

        selected = random.sample(low_hot, 3) + random.sample(high_hot, 2)

        return {
            'numbers': sorted(selected),
            'eurozahlen': self.select_eurozahlen(),
            'strategy_id': 'ej_low_high',
            'strategy_name': '📉📈 Low/High Balance',
            'description': '3 niedrige + 2 hohe Zahlen',
            'weight': self.weight_manager.get_weight('ej_low_high')
        }

    # === STRATEGIE 13: Hot/Cold Mix ===
    def strategy_hot_cold_mix(self):
        """🔥❄️ Hot/Cold Mix"""
        hot = self.get_hot_numbers(15)
        cold = self.get_cold_numbers(10)

        selected = random.sample(hot, 3) + random.sample(cold, 2)

        return {
            'numbers': sorted(selected),
            'eurozahlen': self.select_eurozahlen(),
            'strategy_id': 'ej_hot_cold_mix',
            'strategy_name': '🔥❄️ Hot/Cold Mix',
            'description': '3 heiße + 2 kalte Zahlen',
            'weight': self.weight_manager.get_weight('ej_hot_cold_mix')
        }

    # === STRATEGIE 14: Monte-Carlo ===
    def strategy_monte_carlo(self, simulations=1000):
        """🎲 Monte-Carlo Simulation"""
        hot = self.get_hot_numbers(25)
        results = Counter()

        for _ in range(simulations):
            sample = random.choices(hot or list(range(1, 51)), k=5)
            sample = list(set(sample))
            while len(sample) < 5:
                sample.append(random.randint(1, 50))
            for n in sample[:5]:
                results[n] += 1

        top = [n for n, _ in results.most_common(5)]

        return {
            'numbers': sorted(top),
            'eurozahlen': self.select_eurozahlen(),
            'strategy_id': 'ej_monte_carlo',
            'strategy_name': '🎲 Monte-Carlo',
            'description': f'{simulations} Simulationen',
            'weight': self.weight_manager.get_weight('ej_monte_carlo')
        }

    # === STRATEGIE 15: Bayesian ===
    def strategy_bayesian(self):
        """📊 Bayesian Inference"""
        # Prior: Gleichverteilung
        prior = {n: 1/50 for n in range(1, 51)}

        # Likelihood basierend auf Häufigkeit
        all_nums = []
        for d in self.draws[:100]:
            all_nums.extend(d['numbers'])
        freq = Counter(all_nums)
        total = sum(freq.values()) or 1

        likelihood = {n: (freq.get(n, 1) / total) for n in range(1, 51)}

        # Posterior = Prior * Likelihood
        posterior = {n: prior[n] * likelihood[n] for n in range(1, 51)}

        # Normalisieren
        total_post = sum(posterior.values())
        posterior = {n: p / total_post for n, p in posterior.items()}

        # Top 5
        top = sorted(posterior.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            'numbers': sorted([n for n, _ in top]),
            'eurozahlen': self.select_eurozahlen(),
            'strategy_id': 'ej_bayesian',
            'strategy_name': '📊 Bayesian Inference',
            'description': 'Wahrscheinlichkeitsberechnung',
            'weight': self.weight_manager.get_weight('ej_bayesian')
        }

    def get_all_predictions(self):
        """Generiert alle 15 Strategie-Vorhersagen"""
        strategies = [
            self.strategy_hot_cold,
            self.strategy_cold_numbers,
            self.strategy_overdue,
            self.strategy_odd_even_32,
            self.strategy_odd_even_23,
            self.strategy_sum_optimized,
            self.strategy_decade_balance,
            self.strategy_delta_pattern,
            self.strategy_position_based,
            self.strategy_no_consecutive,
            self.strategy_prime_mix,
            self.strategy_low_high,
            self.strategy_hot_cold_mix,
            self.strategy_monte_carlo,
            self.strategy_bayesian
        ]

        predictions = []
        for strategy_fn in strategies:
            try:
                pred = strategy_fn()
                predictions.append(pred)
            except Exception as e:
                print(f"  ⚠️ Strategie-Fehler: {e}")

        return predictions

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
    """Ruft Groq API auf"""
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
    """Ruft HuggingFace API auf"""
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
# HAUPTVORHERSAGE-FUNKTION
# =====================================================

def generate_predictions():
    """Hauptfunktion: Generiert Eurojackpot-Vorhersagen mit allen KI-Systemen"""

    print("=" * 60)
    print("🌟 LottoGenius - EUROJACKPOT Multi-KI System")
    print("=" * 60)
    print(f"📅 {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
    print()

    # Lade Daten
    ej_data = load_json('eurojackpot_data.json', {'draws': []})
    analysis = load_json('eurojackpot_analysis.json', {})
    predictions = load_json('eurojackpot_predictions.json', {'predictions': [], 'history': []})

    draws = ej_data.get('draws', [])
    if not draws:
        print("⚠️ Keine Eurojackpot-Daten vorhanden!")
        return

    print(f"📊 Analysiere {len(draws)} historische Ziehungen...")

    # Archiviere alte Vorhersagen
    if predictions.get('predictions'):
        predictions['history'].extend(predictions['predictions'])
        predictions['history'] = predictions['history'][-500:]

    # Initialisiere Eurozahl-Analyzer
    euro_analyzer = EurozahlAnalyzer(draws)
    best_euro, euro_ranking = euro_analyzer.predict_best_eurozahlen()

    print(f"\n🌟 Eurozahlen-Analyse (6-Faktoren):")
    print(f"   Beste Eurozahlen: {best_euro}")
    print(f"   Top 5: {euro_ranking[:5]}")

    # Initialisiere ML-Modelle
    ml_models = EurojackpotMLModels(draws, analysis)

    # Erstelle KI-Prompt
    hot_numbers = analysis.get('frequency', {}).get('hot_numbers', [])[:10]
    cold_numbers = analysis.get('frequency', {}).get('cold_numbers', [])[:10]
    overdue = analysis.get('gaps', {}).get('overdue_numbers', [])[:10]
    last_draw = draws[0]

    ki_prompt = f"""Du bist ein Eurojackpot-Analyse-Experte. Analysiere diese Daten für Eurojackpot (5 aus 50 + 2 aus 12):

LETZTE ZIEHUNG: {last_draw['date']} - Zahlen: {last_draw['numbers']}, Eurozahlen: {last_draw['eurozahlen']}

STATISTIK (basierend auf {len(draws)} Ziehungen):
- Heiße Zahlen (häufig): {hot_numbers}
- Kalte Zahlen (selten): {cold_numbers}
- Überfällige Zahlen: {overdue}
- Beste Eurozahlen laut Analyse: {best_euro}

Generiere 2 verschiedene Tipps. Antworte NUR mit diesem JSON-Format:
{{
  "predictions": [
    {{"numbers": [1,2,3,4,5], "eurozahlen": [1,2], "strategy": "Beschreibung", "confidence": 75}},
    {{"numbers": [6,7,8,9,10], "eurozahlen": [3,4], "strategy": "Beschreibung", "confidence": 70}}
  ]
}}"""

    new_predictions = []
    ki_results = {}

    # ===== 0. ECHTE ML-MODELLE (6 Algorithmen) =====
    if ML_AVAILABLE:
        print("\n🧠 ECHTE ML-Modelle (6 Algorithmen):")
        print("   Neural Network | Markov | Bayesian | Q-Learning | EurozahlML | Ensemble")
        try:
            ml_predictions = get_eurojackpot_ml_predictions(draws)
            for pred in ml_predictions:
                pred['timestamp'] = datetime.now().isoformat()
                pred['verified'] = False
                new_predictions.append(pred)
                print(f"   ✅ {pred['method_name']}: {pred['numbers']} | Euro: {pred['eurozahlen']}")
            ki_results['ml_real'] = True
            print(f"   📊 {len(ml_predictions)} ML-Vorhersagen generiert")
        except Exception as e:
            print(f"   ❌ ML-Fehler: {e}")
    else:
        print("\n⚠️ Echte ML-Modelle nicht verfügbar")

    # ===== 1. LOKALE ML-MODELLE (15 Strategien) =====
    print("\n🖥️ Lokale Strategien (15 Methoden):")

    local_predictions = ml_models.get_all_predictions()

    for pred in local_predictions:
        try:
            new_predictions.append({
                'numbers': pred['numbers'],
                'eurozahlen': pred['eurozahlen'],
                'method': pred['strategy_id'],
                'method_name': pred['strategy_name'],
                'provider': 'local_ml',
                'strategy': pred['description'],
                'confidence': round(50 + pred['weight'] * 30, 1),
                'weight': pred['weight'],
                'timestamp': datetime.now().isoformat(),
                'verified': False
            })
            print(f"   ✅ {pred['strategy_name']}: {pred['numbers']} | Euro: {pred['eurozahlen']} (w={pred['weight']:.2f})")
            ki_results['local_ml'] = True
        except Exception as e:
            print(f"   ❌ Fehler: {e}")

    # ===== 2. EXTERNE KI-APIS =====
    print("\n🌐 Externe KI-APIs:")

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
                import re
                json_match = re.search(r'\{[\s\S]*\}', result)
                if json_match:
                    parsed = json.loads(json_match.group())
                    if 'predictions' in parsed:
                        for pred in parsed['predictions']:
                            nums = pred.get('numbers', [])[:5]
                            euros = pred.get('eurozahlen', [])[:2]
                            if len(nums) == 5 and all(1 <= n <= 50 for n in nums):
                                if len(euros) != 2 or not all(1 <= e <= 12 for e in euros):
                                    euros = best_euro
                                new_predictions.append({
                                    'numbers': sorted(nums),
                                    'eurozahlen': sorted(euros),
                                    'method': f'{provider_id}_ki',
                                    'method_name': f'{provider_name} KI',
                                    'provider': provider_id,
                                    'strategy': pred.get('strategy', ''),
                                    'confidence': pred.get('confidence', 70),
                                    'timestamp': datetime.now().isoformat(),
                                    'verified': False
                                })
                        print(f"   ✅ {provider_name}: {len(parsed['predictions'])} Vorhersagen")
                        ki_results[provider_id] = True
                        continue

                print(f"   ⚠️ {provider_name}: Konnte Antwort nicht parsen")
            else:
                print(f"   ❌ {provider_name}: Keine Antwort")

        except Exception as e:
            print(f"   ❌ {provider_name}: {e}")

    # ===== 3. ENSEMBLE-VOTING (KI wählt die Besten) =====
    print("\n🏆 Ensemble-Voting (KI wählt die Besten):")

    if len(new_predictions) >= 3:
        # Gewichtetes Voting basierend auf Strategie-Performance
        all_numbers = Counter()
        all_euro = Counter()

        for pred in new_predictions:
            weight = pred.get('weight', 1.0)
            for n in pred['numbers']:
                all_numbers[n] += weight
            for e in pred['eurozahlen']:
                all_euro[e] += weight

        # Champion-Tipp: Zahlen mit meisten gewichteten Stimmen
        top_voted = [n for n, _ in all_numbers.most_common(5)]
        top_euro = [e for e, _ in all_euro.most_common(2)]

        new_predictions.insert(0, {
            'numbers': sorted(top_voted),
            'eurozahlen': sorted(top_euro),
            'method': 'ensemble_champion',
            'method_name': '🏆 CHAMPION (Alle KIs)',
            'provider': 'ensemble',
            'strategy': f'Gewichtetes Voting aus {len(new_predictions)} KI-Vorhersagen',
            'confidence': 90,
            'timestamp': datetime.now().isoformat(),
            'verified': False,
            'is_champion': True
        })

        print(f"   ✅ Champion-Tipp: {sorted(top_voted)} | Euro: {sorted(top_euro)}")
        print(f"   📊 Basiert auf {len(new_predictions)-1} gewichteten Vorhersagen")

        # === TOP 10 nach Gewicht ===
        print("\n📊 TOP 10 Strategien nach Performance:")
        top_strategies = ml_models.weight_manager.get_top_strategies(10)
        for i, (strategy_id, stats) in enumerate(top_strategies, 1):
            print(f"   {i}. {strategy_id}: Gewicht={stats.get('weight', 1.0):.3f} | Treffer={stats.get('total_matches', 0)}")

    # ===== PROVIDER-STATUS SPEICHERN =====
    provider_status = load_json('eurojackpot_provider_status.json', {'providers': {}})
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
        elif provider_id in ki_results:
            provider_status['providers'][provider_id]['status'] = 'online'
            provider_status['providers'][provider_id]['last_success'] = datetime.now().isoformat()
        else:
            provider_status['providers'][provider_id]['status'] = 'error'

    provider_status['providers']['local_ml'] = {
        'name': 'Lokale ML-Modelle',
        'icon': '🖥️',
        'status': 'online',
        'last_success': datetime.now().isoformat()
    }

    save_json('eurojackpot_provider_status.json', provider_status)

    # ===== SPEICHERN =====

    # Berechne nächste Ziehung (Dienstag oder Freitag)
    now = datetime.now()
    days_to_tue = (1 - now.weekday() + 7) % 7
    days_to_fri = (4 - now.weekday() + 7) % 7
    if days_to_tue == 0 and now.hour >= 21:
        days_to_tue = 7
    if days_to_fri == 0 and now.hour >= 21:
        days_to_fri = 7
    next_days = min(days_to_tue if days_to_tue > 0 else 7, days_to_fri if days_to_fri > 0 else 7)
    next_draw = now + timedelta(days=next_days)
    next_draw_str = next_draw.strftime('%d.%m.%Y')

    # ===== NUR DIE BESTEN 8 TIPPS BEHALTEN =====
    # Sortiere nach: 1. is_champion, 2. confidence, 3. weight
    def sort_key(pred):
        is_champ = 1 if pred.get('is_champion') else 0
        conf = pred.get('confidence', 50)
        weight = pred.get('weight', 1.0)
        # ML-Modelle bekommen Bonus
        is_ml = 1 if 'ml_' in pred.get('method', '') or '_real' in pred.get('method', '') else 0
        return (is_champ, is_ml, conf, weight)

    sorted_predictions = sorted(new_predictions, key=sort_key, reverse=True)
    top_8_predictions = sorted_predictions[:8]

    print(f"\n🏆 TOP 8 BESTE TIPPS (aus {len(new_predictions)} analysiert):")
    for i, pred in enumerate(top_8_predictions, 1):
        nums = pred.get('numbers', [])
        euro = pred.get('eurozahlen', [])
        method = pred.get('method_name', pred.get('method', 'unknown'))
        conf = pred.get('confidence', 0)
        champ = " 👑" if pred.get('is_champion') else ""
        print(f"   {i}. {nums} | Euro: {euro} - {method} ({conf:.0f}%){champ}")

    predictions['predictions'] = top_8_predictions
    predictions['all_predictions_count'] = len(new_predictions)
    predictions['last_update'] = datetime.now().isoformat()
    predictions['next_draw'] = next_draw_str
    predictions['ki_stats'] = {
        'providers_used': list(ki_results.keys()),
        'total_analyzed': len(new_predictions),
        'top_8_saved': len(top_8_predictions),
        'best_eurozahlen': best_euro,
        'eurozahl_ranking': [(ez, round(score, 1)) for ez, score in euro_ranking[:5]],
        'top_strategies': [(s, stats.get('weight', 1.0)) for s, stats in ml_models.weight_manager.get_top_strategies(5)]
    }

    save_json('eurojackpot_predictions.json', predictions)
    ml_models.weight_manager.save()

    print("\n" + "=" * 60)
    print(f"✅ TOP 8 Eurojackpot-Tipps gespeichert! (aus {len(new_predictions)} analysiert)")
    print(f"📅 Nächste Ziehung: {next_draw_str}")
    print(f"🌟 Beste Eurozahlen: {best_euro}")
    print(f"🤖 KI-Provider verwendet: {len(ki_results)}")
    print("=" * 60)

if __name__ == "__main__":
    generate_predictions()
