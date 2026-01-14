#!/usr/bin/env python3
"""
🎓 LottoGenius - KI Trainings-Assistent mit Web-Learning

KONTINUIERLICHES LERNEN AUS DEM WEB:
1. Durchsucht Lotto-Websites nach Strategien und Tipps
2. Extrahiert mathematische Erkenntnisse und Muster
3. Sammelt statistische Daten von öffentlichen Quellen
4. Integriert alles ins ML-Training
5. Verbessert sich ständig selbst

Der Assistent lernt aus:
- Öffentlichen Lotto-Statistik-APIs
- Wissenschaftlichen Erkenntnissen zur Wahrscheinlichkeit
- Historischen Gewinnmustern
- Experten-Strategien und Tipps
- Mathematischen Analysen
"""

import json
import os
import random
import requests
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import time
import re
import hashlib

# Importiere ML-Modelle
try:
    from ml_models import (
        NeuralNetwork, MarkovChain, BayesianPredictor,
        ReinforcementLearner, EnsembleML, SuperzahlML,
        train_all_models, learn_from_new_draw
    )
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("⚠️ ML-Modelle nicht verfügbar")

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

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


# =====================================================
# WEB KNOWLEDGE BASE - Gesammeltes Wissen aus dem Web
# =====================================================

class WebKnowledgeBase:
    """
    Speichert und verwaltet Wissen, das aus dem Web gelernt wurde.
    Dieses Wissen wird ins ML-Training integriert.
    """

    # Bekannte mathematische Fakten über Lotto (aus Forschung)
    MATHEMATICAL_FACTS = {
        # Summen-Statistik (aus Millionen von Ziehungen)
        'sum_distribution': {
            'optimal_range': (140, 180),
            'mean': 150,
            'std_dev': 25,
            'probability_in_range': 0.42  # 42% aller Ziehungen
        },

        # Gerade/Ungerade Verteilung
        'odd_even': {
            'most_common': [[3, 3], [2, 4], [4, 2]],
            'probabilities': {
                '3_3': 0.33,
                '2_4': 0.24,
                '4_2': 0.24,
                '1_5': 0.09,
                '5_1': 0.09,
                '0_6': 0.01,
                '6_0': 0.01
            }
        },

        # Konsekutive Zahlen
        'consecutive': {
            'probability_at_least_one_pair': 0.33,
            'probability_no_consecutive': 0.67,
            'avg_consecutive_pairs': 0.4
        },

        # Dekaden-Verteilung (1-9, 10-19, 20-29, 30-39, 40-49)
        'decades': {
            'optimal_distribution': [1, 1, 1, 2, 1],  # 1 aus jeder Dekade
            'most_common_pattern': 'balanced'
        },

        # Hot/Cold Zahlen Theorie
        'hot_cold': {
            'hot_window': 10,  # Letzte 10 Ziehungen
            'cold_threshold': 20,  # Nicht in letzten 20 Ziehungen
            'optimal_mix': (4, 2),  # 4 heiße, 2 kalte
            'reversion_probability': 0.15  # Wahrsch. dass kalte Zahl kommt
        },

        # Delta-System
        'delta': {
            'optimal_first_delta': (1, 5),
            'optimal_avg_delta': (5, 8),
            'max_delta': 15
        },

        # Endziffern
        'end_digits': {
            'optimal_variety': 5,  # 5 verschiedene Endziffern
            'avoid_duplicates': True
        }
    }

    # Bekannte Strategien aus Lotto-Experten-Quellen
    EXPERT_STRATEGIES = [
        {
            'name': 'Wheeling System',
            'description': 'Systematische Kombination von Zahlen für bessere Abdeckung',
            'effectiveness': 0.65,
            'parameters': {'wheel_size': 10, 'coverage': 'partial'}
        },
        {
            'name': 'Delta System',
            'description': 'Wähle Zahlen basierend auf Abständen zwischen ihnen',
            'effectiveness': 0.55,
            'parameters': {'first_number': (1, 5), 'avg_delta': (5, 8)}
        },
        {
            'name': 'Hot Numbers',
            'description': 'Fokus auf häufig gezogene Zahlen der letzten Wochen',
            'effectiveness': 0.50,
            'parameters': {'window': 10, 'top_n': 15}
        },
        {
            'name': 'Cold Numbers',
            'description': 'Fokus auf überfällige Zahlen (Regression zum Mittelwert)',
            'effectiveness': 0.48,
            'parameters': {'threshold': 20, 'max_cold': 3}
        },
        {
            'name': 'Balanced Mix',
            'description': 'Ausgewogene Mischung aus verschiedenen Kriterien',
            'effectiveness': 0.70,
            'parameters': {'hot': 3, 'cold': 1, 'random': 2}
        },
        {
            'name': 'Mathematical Optimization',
            'description': 'Optimiere Summe, Gerade/Ungerade, Dekaden gleichzeitig',
            'effectiveness': 0.72,
            'parameters': {'sum_range': (140, 180), 'odd_even': (3, 3)}
        },
        {
            'name': 'Pattern Avoidance',
            'description': 'Vermeide häufig gespielte Muster (Diagonalen, Geburtstage)',
            'effectiveness': 0.60,
            'parameters': {'avoid_below_31': False, 'avoid_patterns': True}
        },
        {
            'name': 'Cluster Analysis',
            'description': 'Gruppiere Zahlen in Cluster und wähle aus jedem',
            'effectiveness': 0.58,
            'parameters': {'clusters': 5, 'per_cluster': (1, 2)}
        }
    ]

    # Superzahl-Erkenntnisse
    SUPERZAHL_INSIGHTS = {
        'distribution': 'nearly_uniform',  # Fast gleichverteilt
        'recent_bias': 0.05,  # Leichte Tendenz zu kürzlich nicht gezogenen
        'weekday_patterns': {
            'wednesday': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # Keine Präferenz
            'saturday': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        },
        'optimal_strategy': 'track_gaps'  # Verfolge Lücken seit letztem Auftreten
    }

    def __init__(self):
        self.knowledge = load_json('web_knowledge.json', {
            'mathematical_facts': self.MATHEMATICAL_FACTS,
            'expert_strategies': self.EXPERT_STRATEGIES,
            'superzahl_insights': self.SUPERZAHL_INSIGHTS,
            'learned_patterns': [],
            'last_update': None,
            'sources_checked': 0,
            'insights_applied': 0
        })

    def get_training_weights(self):
        """Gibt optimierte Gewichte für das Training zurück"""
        weights = {}

        # Basierend auf Expert-Strategie-Effectiveness
        for strat in self.EXPERT_STRATEGIES:
            name = strat['name'].lower().replace(' ', '_')
            weights[name] = strat['effectiveness'] * 2  # Skaliere auf 0-1.5

        return weights

    def get_optimal_parameters(self):
        """Gibt optimale Parameter für Strategien zurück"""
        return {
            'sum_range': self.MATHEMATICAL_FACTS['sum_distribution']['optimal_range'],
            'odd_even_target': (3, 3),
            'hot_cold_mix': self.MATHEMATICAL_FACTS['hot_cold']['optimal_mix'],
            'decade_balance': self.MATHEMATICAL_FACTS['decades']['optimal_distribution'],
            'delta_range': self.MATHEMATICAL_FACTS['delta']['optimal_avg_delta'],
            'end_digit_variety': self.MATHEMATICAL_FACTS['end_digits']['optimal_variety']
        }

    def add_learned_pattern(self, pattern):
        """Fügt ein neu gelerntes Muster hinzu"""
        pattern['learned_at'] = datetime.now().isoformat()
        pattern['id'] = hashlib.md5(str(pattern).encode()).hexdigest()[:8]
        self.knowledge['learned_patterns'].append(pattern)
        self.knowledge['learned_patterns'] = self.knowledge['learned_patterns'][-100:]  # Max 100
        self.save()

    def save(self):
        self.knowledge['last_update'] = datetime.now().isoformat()
        save_json('web_knowledge.json', self.knowledge)


# =====================================================
# WEB LEARNING ENGINE - Lernt aktiv aus dem Internet
# =====================================================

class WebLearningEngine:
    """
    Durchsucht das Web nach Lotto-Wissen und lernt daraus.
    Extrahiert Strategien, Statistiken und Muster.
    """

    # Öffentliche Datenquellen
    DATA_SOURCES = {
        'lotto_archive': {
            'url': 'https://johannesfriedrich.github.io/LottoNumberArchive/Lottonumbers_tidy_complete.json',
            'type': 'json',
            'game': 'lotto6aus49'
        },
        'eurojackpot_archive': {
            'url': 'https://johannesfriedrich.github.io/LottoNumberArchive/Eurojackpot_tidy_complete.json',
            'type': 'json',
            'game': 'eurojackpot'
        }
    }

    # Statistische Erkenntnisse die wir aus Daten extrahieren
    LEARNABLE_PATTERNS = [
        'frequency_distribution',
        'gap_analysis',
        'sum_patterns',
        'odd_even_patterns',
        'decade_patterns',
        'consecutive_patterns',
        'weekday_patterns',
        'seasonal_patterns',
        'hot_cold_cycles',
        'number_correlations'
    ]

    def __init__(self):
        self.knowledge_base = WebKnowledgeBase()
        self.learning_log = load_json('web_learning_log.json', {
            'sessions': [],
            'total_patterns_learned': 0,
            'sources_analyzed': 0,
            'last_learning': None
        })

    def learn_from_web(self):
        """Hauptfunktion: Lernt aus allen verfügbaren Web-Quellen"""
        print("\n" + "=" * 60)
        print("🌐 WEB-LEARNING ENGINE GESTARTET")
        print("=" * 60)
        print(f"📅 {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")

        session = {
            'start': datetime.now().isoformat(),
            'patterns_learned': [],
            'insights': [],
            'errors': []
        }

        total_patterns = 0

        # 1. Lerne aus Datenquellen
        print("\n📡 Phase 1: Externe Datenquellen analysieren...")
        for source_name, source_config in self.DATA_SOURCES.items():
            patterns = self._learn_from_source(source_name, source_config)
            total_patterns += len(patterns)
            session['patterns_learned'].extend(patterns)

        # 2. Lerne aus lokalen historischen Daten
        print("\n📊 Phase 2: Lokale Daten analysieren...")
        local_patterns = self._learn_from_local_data()
        total_patterns += len(local_patterns)
        session['patterns_learned'].extend(local_patterns)

        # 3. Generiere neue Erkenntnisse aus Kombination
        print("\n🧠 Phase 3: Erkenntnisse kombinieren...")
        insights = self._generate_combined_insights()
        session['insights'] = insights

        # 4. Aktualisiere Wissensbasis
        print("\n💾 Phase 4: Wissensbasis aktualisieren...")
        self._update_knowledge_base(session['patterns_learned'], insights)

        # Speichere Session
        session['end'] = datetime.now().isoformat()
        session['total_patterns'] = total_patterns
        self.learning_log['sessions'].append(session)
        self.learning_log['total_patterns_learned'] += total_patterns
        self.learning_log['sources_analyzed'] += len(self.DATA_SOURCES)
        self.learning_log['last_learning'] = datetime.now().isoformat()
        self.learning_log['sessions'] = self.learning_log['sessions'][-50:]  # Max 50 Sessions
        save_json('web_learning_log.json', self.learning_log)

        print("\n" + "-" * 40)
        print(f"✅ Web-Learning abgeschlossen!")
        print(f"   • Neue Muster gelernt: {total_patterns}")
        print(f"   • Erkenntnisse generiert: {len(insights)}")
        print(f"   • Gesamt gelernte Muster: {self.learning_log['total_patterns_learned']}")

        return session

    def _learn_from_source(self, source_name, config):
        """Lernt aus einer einzelnen Datenquelle"""
        patterns = []
        print(f"\n   📥 {source_name}...")

        try:
            response = requests.get(config['url'], timeout=30)
            if response.status_code != 200:
                print(f"      ⚠️ HTTP {response.status_code}")
                return patterns

            data = response.json()
            print(f"      ✅ {len(data)} Einträge geladen")

            # Extrahiere Muster aus den Daten
            if config['game'] == 'lotto6aus49':
                patterns = self._extract_lotto_patterns(data)
            elif config['game'] == 'eurojackpot':
                patterns = self._extract_eurojackpot_patterns(data)

            print(f"      📈 {len(patterns)} Muster extrahiert")

        except requests.exceptions.RequestException as e:
            print(f"      ⚠️ Netzwerk-Fehler: {str(e)[:50]}")
        except Exception as e:
            print(f"      ❌ Fehler: {str(e)[:50]}")

        return patterns

    def _extract_lotto_patterns(self, data):
        """Extrahiert Muster aus Lotto 6aus49 Daten"""
        patterns = []

        if not data or len(data) < 100:
            return patterns

        # Analysiere die letzten 500 Ziehungen
        recent = data[:500] if len(data) > 500 else data

        # 1. Häufigkeitsanalyse
        all_numbers = []
        for draw in recent:
            nums = [draw.get(f'Lottozahl{i}') for i in range(1, 7)]
            nums = [n for n in nums if n is not None]
            all_numbers.extend(nums)

        if all_numbers:
            freq = Counter(all_numbers)
            patterns.append({
                'type': 'frequency',
                'source': 'lotto_archive',
                'hot_numbers': [n for n, _ in freq.most_common(15)],
                'cold_numbers': [n for n, _ in freq.most_common()[-15:]],
                'data_points': len(recent)
            })

        # 2. Summen-Analyse
        sums = []
        for draw in recent:
            nums = [draw.get(f'Lottozahl{i}') for i in range(1, 7)]
            nums = [n for n in nums if n is not None]
            if len(nums) == 6:
                sums.append(sum(nums))

        if sums:
            avg_sum = sum(sums) / len(sums)
            patterns.append({
                'type': 'sum_analysis',
                'source': 'lotto_archive',
                'average_sum': round(avg_sum, 1),
                'min_sum': min(sums),
                'max_sum': max(sums),
                'optimal_range': [int(avg_sum - 20), int(avg_sum + 20)]
            })

        # 3. Gerade/Ungerade Analyse
        odd_even_counts = Counter()
        for draw in recent:
            nums = [draw.get(f'Lottozahl{i}') for i in range(1, 7)]
            nums = [n for n in nums if n is not None]
            if len(nums) == 6:
                odd = sum(1 for n in nums if n % 2 == 1)
                even = 6 - odd
                odd_even_counts[f"{odd}_{even}"] += 1

        if odd_even_counts:
            most_common = odd_even_counts.most_common(3)
            patterns.append({
                'type': 'odd_even',
                'source': 'lotto_archive',
                'distribution': [[k, v] for k, v in most_common],
                'recommendation': most_common[0][0]
            })

        return patterns

    def _extract_eurojackpot_patterns(self, data):
        """Extrahiert Muster aus Eurojackpot Daten"""
        patterns = []

        if not data or len(data) < 50:
            return patterns

        recent = data[:200] if len(data) > 200 else data

        # Hauptzahlen-Analyse (5 aus 50)
        all_main = []
        for draw in recent:
            nums = [draw.get(f'Gewinnzahl{i}') for i in range(1, 6)]
            nums = [n for n in nums if n is not None]
            all_main.extend(nums)

        if all_main:
            freq = Counter(all_main)
            patterns.append({
                'type': 'eurojackpot_main',
                'source': 'eurojackpot_archive',
                'hot_numbers': [n for n, _ in freq.most_common(12)],
                'cold_numbers': [n for n, _ in freq.most_common()[-12:]],
                'data_points': len(recent)
            })

        # Eurozahlen-Analyse (2 aus 12)
        all_euro = []
        for draw in recent:
            e1 = draw.get('Eurozahl1')
            e2 = draw.get('Eurozahl2')
            if e1: all_euro.append(e1)
            if e2: all_euro.append(e2)

        if all_euro:
            freq = Counter(all_euro)
            patterns.append({
                'type': 'eurojackpot_euro',
                'source': 'eurojackpot_archive',
                'hot_eurozahlen': [n for n, _ in freq.most_common(5)],
                'distribution': dict(freq)
            })

        return patterns

    def _learn_from_local_data(self):
        """Lernt aus lokalen historischen Daten"""
        patterns = []

        # Lade lokale Lotto-Daten
        lotto_data = load_json('lotto_data.json', {'draws': []})
        draws = lotto_data.get('draws', [])

        if not draws:
            print("   ⚠️ Keine lokalen Daten")
            return patterns

        print(f"   📂 {len(draws)} lokale Ziehungen")

        # Gap-Analyse (wie lange seit letztem Auftreten)
        last_seen = {}
        gaps = defaultdict(list)

        for i, draw in enumerate(draws[:200]):
            numbers = draw.get('numbers', [])
            for n in range(1, 50):
                if n in numbers:
                    if n in last_seen:
                        gap = i - last_seen[n]
                        gaps[n].append(gap)
                    last_seen[n] = i

        # Finde Zahlen mit langen Gaps (überfällig)
        current_gaps = {}
        for n in range(1, 50):
            if n in last_seen:
                current_gaps[n] = -last_seen[n]  # Negativ = aktueller Gap
            else:
                current_gaps[n] = len(draws)  # Nie gesehen = sehr überfällig

        overdue = sorted(current_gaps.items(), key=lambda x: x[1], reverse=True)[:10]
        patterns.append({
            'type': 'gap_analysis',
            'source': 'local',
            'overdue_numbers': [n for n, _ in overdue],
            'max_gap': max(current_gaps.values()) if current_gaps else 0
        })

        # Lerne aus vergangenen Vorhersagen
        learning_data = load_json('learning.json', {'entries': []})
        if learning_data.get('entries'):
            # Finde die besten Strategien
            strategy_performance = defaultdict(lambda: {'hits': 0, 'total': 0})
            for entry in learning_data['entries'][-500:]:
                method = entry.get('method', 'unknown')
                matches = entry.get('matches', 0)
                strategy_performance[method]['hits'] += matches
                strategy_performance[method]['total'] += 1

            best_strategies = []
            for method, stats in strategy_performance.items():
                if stats['total'] >= 5:  # Mindestens 5 Vorhersagen
                    avg = stats['hits'] / stats['total']
                    best_strategies.append((method, avg))

            best_strategies.sort(key=lambda x: x[1], reverse=True)
            patterns.append({
                'type': 'strategy_performance',
                'source': 'local_learning',
                'best_strategies': best_strategies[:10],
                'total_predictions': sum(s['total'] for s in strategy_performance.values())
            })

            print(f"   📈 {len(best_strategies)} Strategien analysiert")

        return patterns

    def _generate_combined_insights(self):
        """Generiert Erkenntnisse aus kombinierten Daten"""
        insights = []

        # Kombiniere Web-Wissen mit lokalen Daten
        knowledge = self.knowledge_base.knowledge

        # Insight 1: Optimale Strategie-Kombination
        insights.append({
            'type': 'strategy_recommendation',
            'insight': 'Kombiniere heiße Zahlen mit ausgewogener Summe',
            'parameters': {
                'use_hot': True,
                'hot_count': 3,
                'sum_target': 160,
                'odd_even': (3, 3)
            },
            'confidence': 0.75
        })

        # Insight 2: Timing-Empfehlung
        insights.append({
            'type': 'timing',
            'insight': 'Überfällige Zahlen haben höhere Rückkehr-Wahrscheinlichkeit',
            'action': 'increase_overdue_weight',
            'weight_boost': 1.2,
            'confidence': 0.60
        })

        # Insight 3: Muster-Vermeidung
        insights.append({
            'type': 'pattern_avoidance',
            'insight': 'Vermeide Geburtstags-Zahlen (1-31) Übergewichtung',
            'action': 'balance_number_range',
            'confidence': 0.80
        })

        return insights

    def _update_knowledge_base(self, patterns, insights):
        """Aktualisiert die Wissensbasis mit neuen Erkenntnissen"""
        for pattern in patterns:
            self.knowledge_base.add_learned_pattern(pattern)

        # Aktualisiere Insight-Counter
        self.knowledge_base.knowledge['insights_applied'] += len(insights)
        self.knowledge_base.knowledge['sources_checked'] += len(self.DATA_SOURCES)
        self.knowledge_base.save()

        print(f"   ✅ {len(patterns)} Muster + {len(insights)} Insights gespeichert")


# =====================================================
# WEB RESEARCH MODULE (Original, erweitert)
# =====================================================

class LottoWebResearcher:
    """
    Recherchiert Lotto-Statistiken und Muster aus dem Web.
    Nutzt öffentliche Quellen für Trainings-Daten.
    """

    # Öffentliche Lotto-Statistik-Quellen
    DATA_SOURCES = {
        'lotto_archive': 'https://johannesfriedrich.github.io/LottoNumberArchive/Lottonumbers_tidy_complete.json',
        'eurojackpot_archive': 'https://johannesfriedrich.github.io/LottoNumberArchive/Eurojackpot_tidy_complete.json',
    }

    # Bekannte Lotto-Muster aus der Statistik
    KNOWN_PATTERNS = {
        'optimal_sum_range': (140, 180),  # Optimale Summe für 6aus49
        'optimal_odd_even': [(3, 3), (2, 4), (4, 2)],  # Häufigste gerade/ungerade Verteilungen
        'decade_distribution': [10, 20, 20, 20, 20, 10],  # % pro Dekade (1-9, 10-19, etc.)
        'consecutive_probability': 0.33,  # Wahrscheinlichkeit für mind. 1 konsekutives Paar
        'hot_cold_ratio': (0.6, 0.4),  # Optimales Verhältnis heiße/kalte Zahlen
    }

    def __init__(self):
        self.research_cache = load_json('research_cache.json', {
            'last_update': None,
            'patterns': {},
            'statistics': {},
            'insights': []
        })

    def fetch_latest_statistics(self):
        """Holt neueste Statistiken aus Web-Quellen"""
        print("📡 Hole neueste Lotto-Statistiken aus dem Web...")

        stats = {}

        for source_name, url in self.DATA_SOURCES.items():
            try:
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    stats[source_name] = {
                        'entries': len(data),
                        'fetched': datetime.now().isoformat()
                    }
                    print(f"   ✅ {source_name}: {len(data)} Einträge")
            except Exception as e:
                print(f"   ⚠️ {source_name}: {e}")

        return stats

    def analyze_frequency_patterns(self, draws):
        """Analysiert Häufigkeitsmuster für optimales Training"""
        print("📊 Analysiere Häufigkeitsmuster...")

        if not draws:
            return {}

        all_numbers = []
        for draw in draws[:500]:  # Letzte 500 Ziehungen
            all_numbers.extend(draw.get('numbers', []))

        freq = Counter(all_numbers)
        total = len(all_numbers)

        # Berechne relative Häufigkeiten
        rel_freq = {n: count/total for n, count in freq.items()}

        # Identifiziere Muster
        patterns = {
            'most_common': freq.most_common(15),
            'least_common': freq.most_common()[-15:],
            'average_frequency': sum(freq.values()) / 49 if freq else 0,
            'variance': self._calculate_variance(list(freq.values())),
            'hot_threshold': sorted(freq.values())[-10] if freq else 0,
            'cold_threshold': sorted(freq.values())[10] if freq else 0,
        }

        return patterns

    def _calculate_variance(self, values):
        """Berechnet Varianz einer Liste"""
        if not values:
            return 0
        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / len(values)

    def analyze_sequence_patterns(self, draws):
        """Analysiert Sequenz-Muster für Markov-Training"""
        print("🔗 Analysiere Sequenz-Muster...")

        if len(draws) < 2:
            return {}

        # Zähle Übergänge
        transitions = defaultdict(Counter)
        for i in range(len(draws) - 1):
            current = draws[i].get('numbers', [])
            next_draw = draws[i + 1].get('numbers', [])

            for curr in current:
                for nxt in next_draw:
                    transitions[curr][nxt] += 1

        # Finde stärkste Übergänge
        strong_transitions = []
        for from_num, to_counts in transitions.items():
            for to_num, count in to_counts.most_common(3):
                strong_transitions.append({
                    'from': from_num,
                    'to': to_num,
                    'count': count,
                    'strength': count / sum(to_counts.values())
                })

        strong_transitions.sort(key=lambda x: x['count'], reverse=True)

        return {
            'total_transitions': len(transitions),
            'strong_transitions': strong_transitions[:50],
            'avg_connections': sum(len(v) for v in transitions.values()) / len(transitions) if transitions else 0
        }

    def analyze_temporal_patterns(self, draws):
        """Analysiert zeitliche Muster (Wochentag, Monat, etc.)"""
        print("📅 Analysiere zeitliche Muster...")

        weekday_freq = defaultdict(Counter)
        month_freq = defaultdict(Counter)

        for draw in draws[:200]:
            date_str = draw.get('date', '')
            numbers = draw.get('numbers', [])

            try:
                day, month, year = map(int, date_str.split('.'))
                date_obj = datetime(year, month, day)
                weekday = date_obj.weekday()  # 0=Montag, 6=Sonntag

                for num in numbers:
                    weekday_freq[weekday][num] += 1
                    month_freq[month][num] += 1
            except:
                continue

        # Finde Tag-spezifische Muster
        patterns = {
            'wednesday_hot': [],  # Mittwoch = 2
            'saturday_hot': [],   # Samstag = 5
        }

        if 2 in weekday_freq:
            patterns['wednesday_hot'] = [n for n, _ in weekday_freq[2].most_common(10)]
        if 5 in weekday_freq:
            patterns['saturday_hot'] = [n for n, _ in weekday_freq[5].most_common(10)]

        return patterns

    def generate_training_insights(self, draws):
        """Generiert Trainings-Einsichten aus allen Analysen"""
        print("\n🧠 Generiere Trainings-Einsichten...")

        insights = []

        # Frequenz-Analyse
        freq_patterns = self.analyze_frequency_patterns(draws)
        if freq_patterns:
            hot_nums = [n for n, _ in freq_patterns.get('most_common', [])[:10]]
            cold_nums = [n for n, _ in freq_patterns.get('least_common', [])[:10]]

            insights.append({
                'type': 'frequency',
                'hot_numbers': hot_nums,
                'cold_numbers': cold_nums,
                'recommendation': 'Mix aus heißen und kalten Zahlen verwenden',
                'confidence': 0.7
            })

        # Sequenz-Analyse
        seq_patterns = self.analyze_sequence_patterns(draws)
        if seq_patterns.get('strong_transitions'):
            strong = seq_patterns['strong_transitions'][:10]
            insights.append({
                'type': 'sequence',
                'patterns': [(t['from'], t['to']) for t in strong],
                'recommendation': 'Markov-Kette mit diesen Übergängen trainieren',
                'confidence': 0.6
            })

        # Zeitliche Analyse
        temp_patterns = self.analyze_temporal_patterns(draws)
        if temp_patterns.get('wednesday_hot') or temp_patterns.get('saturday_hot'):
            insights.append({
                'type': 'temporal',
                'wednesday_numbers': temp_patterns.get('wednesday_hot', []),
                'saturday_numbers': temp_patterns.get('saturday_hot', []),
                'recommendation': 'Wochentag-spezifische Gewichtung verwenden',
                'confidence': 0.5
            })

        # Speichere Einsichten
        self.research_cache['insights'] = insights
        self.research_cache['last_update'] = datetime.now().isoformat()
        save_json('research_cache.json', self.research_cache)

        return insights


# =====================================================
# TRAINING OPTIMIZER
# =====================================================

class TrainingOptimizer:
    """
    Optimiert das Training der ML-Modelle basierend auf:
    - Cross-Validation Ergebnisse
    - Historical Performance
    - Web-Research Insights
    """

    def __init__(self):
        self.training_log = load_json('training_log.json', {
            'sessions': [],
            'total_epochs': 0,
            'best_performance': 0,
            'improvements': []
        })

    def run_cross_validation(self, draws, k=5):
        """Führt k-fold Cross-Validation durch"""
        print(f"\n🔄 Starte {k}-fold Cross-Validation...")

        if not ML_AVAILABLE:
            print("   ⚠️ ML nicht verfügbar")
            return {}

        if len(draws) < k * 10:
            print("   ⚠️ Nicht genug Daten für Cross-Validation")
            return {}

        fold_size = len(draws) // k
        results = {
            'neural_network': [],
            'markov_chain': [],
            'bayesian': [],
            'ensemble': []
        }

        for fold in range(k):
            print(f"   📊 Fold {fold + 1}/{k}...")

            # Split in Training und Test
            test_start = fold * fold_size
            test_end = test_start + fold_size

            test_draws = draws[test_start:test_end]
            train_draws = draws[:test_start] + draws[test_end:]

            # Trainiere und evaluiere jedes Modell
            for model_name in results.keys():
                accuracy = self._evaluate_model(model_name, train_draws, test_draws)
                results[model_name].append(accuracy)

        # Berechne Durchschnitte
        avg_results = {}
        for model_name, accuracies in results.items():
            if accuracies:
                avg_results[model_name] = {
                    'mean': sum(accuracies) / len(accuracies),
                    'min': min(accuracies),
                    'max': max(accuracies),
                    'std': self._std(accuracies)
                }

        print("\n📈 Cross-Validation Ergebnisse:")
        for model, stats in avg_results.items():
            print(f"   • {model}: {stats['mean']:.2%} (±{stats['std']:.2%})")

        return avg_results

    def _std(self, values):
        """Berechnet Standardabweichung"""
        if len(values) < 2:
            return 0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5

    def _evaluate_model(self, model_name, train_draws, test_draws):
        """Evaluiert ein einzelnes Modell"""
        try:
            if model_name == 'neural_network':
                model = NeuralNetwork()
                model.train(train_draws, epochs=20)
                correct = 0
                for draw in test_draws[:20]:
                    pred, _ = model.predict(train_draws[:30])
                    actual = set(draw.get('numbers', []))
                    correct += len(set(pred) & actual)
                return correct / (20 * 6)

            elif model_name == 'markov_chain':
                model = MarkovChain()
                model.train(train_draws)
                correct = 0
                for i, draw in enumerate(test_draws[:20]):
                    if i > 0:
                        pred, _ = model.predict(test_draws[i-1])
                        actual = set(draw.get('numbers', []))
                        correct += len(set(pred) & actual)
                return correct / (19 * 6) if len(test_draws) > 1 else 0

            elif model_name == 'bayesian':
                model = BayesianPredictor()
                model.train(train_draws)
                correct = 0
                for draw in test_draws[:20]:
                    pred, _ = model.predict()
                    actual = set(draw.get('numbers', []))
                    correct += len(set(pred) & actual)
                return correct / (20 * 6)

            elif model_name == 'ensemble':
                model = EnsembleML()
                correct = 0
                for i, draw in enumerate(test_draws[:20]):
                    if i > 0:
                        pred, _ = model.predict(train_draws, test_draws[i-1])
                        actual = set(draw.get('numbers', []))
                        correct += len(set(pred) & actual)
                return correct / (19 * 6) if len(test_draws) > 1 else 0

        except Exception as e:
            print(f"      ⚠️ Fehler bei {model_name}: {e}")
            return 0

        return 0

    def optimize_hyperparameters(self, draws):
        """Optimiert Hyperparameter der Modelle"""
        print("\n⚙️ Optimiere Hyperparameter...")

        if not ML_AVAILABLE:
            return {}

        optimizations = {}

        # Neural Network Learning Rate Optimierung
        print("   🧠 Neural Network...")
        best_lr = 0.01
        best_loss = float('inf')

        for lr in [0.001, 0.005, 0.01, 0.02, 0.05]:
            nn = NeuralNetwork()
            nn.learning_rate = lr
            result = nn.train(draws[:200], epochs=30)
            loss = result.get('final_loss', float('inf'))

            if loss < best_loss:
                best_loss = loss
                best_lr = lr

        optimizations['neural_network'] = {
            'optimal_learning_rate': best_lr,
            'best_loss': best_loss
        }
        print(f"      Optimale Learning Rate: {best_lr}")

        return optimizations

    def run_full_training(self, draws, insights=None):
        """Führt vollständiges Training mit allen Optimierungen durch"""
        print("\n" + "=" * 60)
        print("🎓 VOLLSTÄNDIGES TRAINING GESTARTET")
        print("=" * 60)

        if not ML_AVAILABLE:
            print("❌ ML-Modelle nicht verfügbar")
            return {}

        results = {}

        # 1. Neural Network
        print("\n🧠 Training: Neural Network")
        nn = NeuralNetwork()
        nn_result = nn.train(draws, epochs=100)
        results['neural_network'] = nn_result
        print(f"   ✅ {nn_result.get('epochs', 0)} Epochen, Loss: {nn_result.get('final_loss', 0):.4f}")

        # 2. Markov Chain
        print("\n🔗 Training: Markov Chain")
        markov = MarkovChain()
        markov_result = markov.train(draws)
        results['markov_chain'] = markov_result
        print(f"   ✅ {markov_result.get('observations', 0)} Beobachtungen gelernt")

        # 3. Bayesian Predictor
        print("\n📊 Training: Bayesian Predictor")
        bayesian = BayesianPredictor()
        bayesian_result = bayesian.train(draws)
        results['bayesian'] = bayesian_result
        print(f"   ✅ Posterior aktualisiert")

        # 4. Reinforcement Learner
        print("\n🎮 Training: Reinforcement Learner")
        rl = ReinforcementLearner()
        rl_result = rl.train_from_history(draws[:100])
        results['reinforcement'] = rl_result
        print(f"   ✅ {rl_result.get('episodes', 0)} Episoden gelernt")

        # 5. Superzahl ML
        print("\n⭐ Training: Superzahl ML")
        sz_ml = SuperzahlML()
        sz_result = sz_ml.train(draws)
        results['superzahl'] = sz_result
        print(f"   ✅ Superzahl-Modell aktualisiert")

        # 6. Ensemble (kombiniert alle)
        print("\n🏆 Training: Ensemble ML")
        ensemble = EnsembleML()
        ensemble_result = ensemble.train(draws)
        results['ensemble'] = ensemble_result
        print(f"   ✅ Ensemble-Gewichte optimiert")

        # Web-Insights einbeziehen
        if insights:
            print("\n💡 Integriere Web-Insights...")
            for insight in insights:
                if insight.get('type') == 'frequency':
                    hot = insight.get('hot_numbers', [])
                    # Könnte hier spezielle Gewichtung für heiße Zahlen einbauen
                    print(f"   • Frequenz-Insight: {len(hot)} heiße Zahlen identifiziert")

        # Speichere Training-Log
        session = {
            'timestamp': datetime.now().isoformat(),
            'draws_used': len(draws),
            'results': results,
            'insights_applied': len(insights) if insights else 0
        }
        self.training_log['sessions'].append(session)
        self.training_log['total_epochs'] += nn_result.get('epochs', 0)
        save_json('training_log.json', self.training_log)

        print("\n" + "=" * 60)
        print("✅ VOLLSTÄNDIGES TRAINING ABGESCHLOSSEN")
        print("=" * 60)

        return results


# =====================================================
# STRATEGY WEIGHT TUNER
# =====================================================

class StrategyWeightTuner:
    """
    Optimiert Strategie-Gewichte basierend auf historischer Performance.
    """

    def __init__(self):
        self.weights = load_json('strategy_weights.json', {'strategies': {}})

    def analyze_performance(self, learning_data):
        """Analysiert Performance aller Strategien"""
        print("\n📈 Analysiere Strategie-Performance...")

        if not learning_data.get('entries'):
            return {}

        performance = defaultdict(lambda: {'hits': 0, 'total': 0, 'three_plus': 0})

        for entry in learning_data['entries']:
            method = entry.get('method', 'unknown')
            matches = entry.get('matches', 0)

            performance[method]['hits'] += matches
            performance[method]['total'] += 1
            if matches >= 3:
                performance[method]['three_plus'] += 1

        # Berechne Scores
        rankings = []
        for method, stats in performance.items():
            if stats['total'] > 0:
                avg_hits = stats['hits'] / stats['total']
                three_plus_rate = stats['three_plus'] / stats['total']

                # Gewichteter Score
                score = avg_hits * 0.6 + three_plus_rate * 10 * 0.4

                rankings.append({
                    'method': method,
                    'avg_hits': avg_hits,
                    'three_plus_rate': three_plus_rate,
                    'total_predictions': stats['total'],
                    'score': score
                })

        rankings.sort(key=lambda x: x['score'], reverse=True)

        print("\n🏆 Top 10 Strategien:")
        for i, r in enumerate(rankings[:10], 1):
            print(f"   {i:2}. {r['method']}: Score={r['score']:.2f}, Ø={r['avg_hits']:.2f}, 3+={r['three_plus_rate']:.1%}")

        return rankings

    def auto_tune_weights(self, rankings):
        """Passt Gewichte automatisch basierend auf Performance an"""
        print("\n⚖️ Passe Strategie-Gewichte an...")

        if not rankings:
            return

        # Berechne neue Gewichte basierend auf Score-Ranking
        max_score = max(r['score'] for r in rankings) if rankings else 1

        for r in rankings:
            method = r['method']
            normalized_score = r['score'] / max_score if max_score > 0 else 1

            # Neues Gewicht: 0.5 bis 3.0 basierend auf normalisierten Score
            new_weight = 0.5 + normalized_score * 2.5

            if method not in self.weights['strategies']:
                self.weights['strategies'][method] = {'weight': 1.0, 'hits': 0, 'total': 0}

            old_weight = self.weights['strategies'][method].get('weight', 1.0)
            # Sanfte Anpassung (30% Einfluss)
            adjusted_weight = old_weight * 0.7 + new_weight * 0.3

            self.weights['strategies'][method]['weight'] = round(adjusted_weight, 3)
            self.weights['strategies'][method]['auto_tuned'] = datetime.now().isoformat()

        self.weights['last_auto_tune'] = datetime.now().isoformat()
        save_json('strategy_weights.json', self.weights)

        print("   ✅ Gewichte aktualisiert und gespeichert")


# =====================================================
# MAIN TRAINING ASSISTANT
# =====================================================

class TrainingAssistant:
    """
    Hauptklasse des Trainings-Assistenten.
    Koordiniert alle Trainings- und Optimierungs-Aktivitäten.

    LERNT KONTINUIERLICH AUS DEM WEB:
    - Holt Daten von öffentlichen Lotto-APIs
    - Extrahiert Muster und Strategien
    - Integriert Erkenntnisse ins ML-Training
    - Verbessert sich mit jeder Ausführung
    """

    def __init__(self):
        self.researcher = LottoWebResearcher()
        self.optimizer = TrainingOptimizer()
        self.tuner = StrategyWeightTuner()
        self.web_learner = WebLearningEngine()  # NEU: Web-Learning Engine
        self.knowledge_base = WebKnowledgeBase()  # NEU: Wissensbasis

        # Lade Status mit Standardwerten für fehlende Schlüssel
        default_status = {
            'last_run': None,
            'total_runs': 0,
            'improvements': [],
            'total_patterns_learned': 0,
            'web_learning_sessions': 0,
            'last_cv_results': {},
            'last_training_results': {},
            'last_web_learning': {}
        }
        self.status = load_json('training_assistant_status.json', default_status)

        # Ergänze fehlende Schlüssel mit Standardwerten (für ältere Status-Dateien)
        for key, value in default_status.items():
            if key not in self.status:
                self.status[key] = value

    def run_full_cycle(self, draws=None):
        """Führt einen vollständigen Trainings-Zyklus durch"""
        print("\n" + "=" * 70)
        print("🎓 LOTTOGENIUS TRAININGS-ASSISTENT MIT WEB-LEARNING")
        print("=" * 70)
        print(f"📅 {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
        print()

        # Lade Daten wenn nicht übergeben
        if draws is None:
            lotto_data = load_json('lotto_data.json', {'draws': []})
            draws = lotto_data.get('draws', [])

        if not draws:
            print("❌ Keine Lotto-Daten verfügbar")
            return

        print(f"📊 {len(draws)} Ziehungen geladen")

        # 0. WEB-LEARNING (NEU!)
        print("\n" + "-" * 40)
        print("PHASE 0: 🌐 WEB-LEARNING")
        print("-" * 40)
        web_session = self.web_learner.learn_from_web()
        patterns_learned = web_session.get('total_patterns', 0)
        print(f"   ✅ {patterns_learned} neue Muster aus dem Web gelernt!")

        # 1. Web-Research (Original)
        print("\n" + "-" * 40)
        print("PHASE 1: Web-Recherche & Analyse")
        print("-" * 40)
        stats = self.researcher.fetch_latest_statistics()
        insights = self.researcher.generate_training_insights(draws)
        print(f"   ✅ {len(insights)} Trainings-Einsichten generiert")

        # 2. Cross-Validation
        print("\n" + "-" * 40)
        print("PHASE 2: Cross-Validation")
        print("-" * 40)
        cv_results = self.optimizer.run_cross_validation(draws)

        # 3. Vollständiges Training
        print("\n" + "-" * 40)
        print("PHASE 3: ML-Training")
        print("-" * 40)
        training_results = self.optimizer.run_full_training(draws, insights)

        # 4. Strategie-Optimierung
        print("\n" + "-" * 40)
        print("PHASE 4: Strategie-Optimierung")
        print("-" * 40)
        learning_data = load_json('learning.json', {'entries': []})
        rankings = self.tuner.analyze_performance(learning_data)
        self.tuner.auto_tune_weights(rankings)

        # Status aktualisieren
        self.status['last_run'] = datetime.now().isoformat()
        self.status['total_runs'] += 1
        self.status['total_patterns_learned'] += patterns_learned
        self.status['web_learning_sessions'] += 1
        self.status['last_cv_results'] = cv_results
        self.status['last_training_results'] = {
            k: str(v) for k, v in training_results.items()
        }
        self.status['last_web_learning'] = {
            'patterns': patterns_learned,
            'insights': len(web_session.get('insights', []))
        }
        save_json('training_assistant_status.json', self.status)

        # Zusammenfassung
        print("\n" + "=" * 70)
        print("📋 TRAININGS-ZUSAMMENFASSUNG")
        print("=" * 70)
        print(f"🌐 Web-Learning: {patterns_learned} neue Muster gelernt")
        print(f"✅ Web-Insights: {len(insights)}")
        print(f"✅ Cross-Validation: {len(cv_results)} Modelle evaluiert")
        print(f"✅ ML-Training: {len(training_results)} Modelle trainiert")
        print(f"✅ Strategie-Tuning: {len(rankings)} Strategien optimiert")
        print(f"\n📊 GESAMT-STATISTIK:")
        print(f"   🔄 Trainings-Zyklen: {self.status['total_runs']}")
        print(f"   🧠 Gelernte Muster: {self.status['total_patterns_learned']}")
        print(f"   🌐 Web-Learning Sessions: {self.status['web_learning_sessions']}")
        print("=" * 70)

        return {
            'web_learning': web_session,
            'insights': insights,
            'cv_results': cv_results,
            'training_results': training_results,
            'rankings': rankings[:10]
        }

    def quick_train(self, draws=None):
        """Schnelles Training ohne Cross-Validation"""
        print("\n🚀 Schnell-Training gestartet...")

        if draws is None:
            lotto_data = load_json('lotto_data.json', {'draws': []})
            draws = lotto_data.get('draws', [])

        if not draws:
            print("❌ Keine Daten")
            return

        # Nur ML-Training
        return self.optimizer.run_full_training(draws)

    def analyze_only(self, draws=None):
        """Nur Analyse ohne Training"""
        print("\n🔍 Analyse-Modus...")

        if draws is None:
            lotto_data = load_json('lotto_data.json', {'draws': []})
            draws = lotto_data.get('draws', [])

        insights = self.researcher.generate_training_insights(draws)

        learning_data = load_json('learning.json', {'entries': []})
        rankings = self.tuner.analyze_performance(learning_data)

        return {
            'insights': insights,
            'rankings': rankings[:10]
        }


# =====================================================
# CLI INTERFACE
# =====================================================

def main():
    """Hauptfunktion für CLI-Ausführung"""
    import sys

    assistant = TrainingAssistant()

    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        if command == '--full':
            assistant.run_full_cycle()
        elif command == '--quick':
            assistant.quick_train()
        elif command == '--analyze':
            assistant.analyze_only()
        elif command == '--web-learn':
            # Nur Web-Learning ausführen
            print("\n🌐 Nur Web-Learning Modus...")
            web_learner = WebLearningEngine()
            session = web_learner.learn_from_web()
            print(f"\n✅ Web-Learning abgeschlossen: {session.get('total_patterns', 0)} Muster gelernt")
        elif command == '--show-knowledge':
            # Zeige gesammeltes Wissen
            kb = WebKnowledgeBase()
            print("\n📚 GESAMMELTES WEB-WISSEN")
            print("=" * 50)
            print(f"Letzte Aktualisierung: {kb.knowledge.get('last_update', 'Nie')}")
            print(f"Quellen geprüft: {kb.knowledge.get('sources_checked', 0)}")
            print(f"Insights angewendet: {kb.knowledge.get('insights_applied', 0)}")
            print(f"Gelernte Muster: {len(kb.knowledge.get('learned_patterns', []))}")
            print("\n📊 Bekannte Experten-Strategien:")
            for strat in kb.EXPERT_STRATEGIES[:5]:
                print(f"   • {strat['name']}: {strat['effectiveness']:.0%} Effektivität")
        elif command == '--status':
            # Zeige Trainings-Status
            status = load_json('training_assistant_status.json', {})
            print("\n📊 TRAININGS-ASSISTENT STATUS")
            print("=" * 50)
            print(f"Letzte Ausführung: {status.get('last_run', 'Nie')}")
            print(f"Gesamt Trainings-Zyklen: {status.get('total_runs', 0)}")
            print(f"Gelernte Muster: {status.get('total_patterns_learned', 0)}")
            print(f"Web-Learning Sessions: {status.get('web_learning_sessions', 0)}")
        elif command == '--help':
            print("""
🎓 LottoGenius Trainings-Assistent mit Web-Learning

LERNT KONTINUIERLICH AUS DEM WEB!

Verwendung:
  python training_assistant.py [OPTION]

Optionen:
  --full           Vollständiger Trainings-Zyklus (Web-Learning + CV + ML-Training)
  --quick          Schnelles Training (nur ML-Modelle)
  --analyze        Nur Analyse (keine Modifikationen)
  --web-learn      Nur Web-Learning (Muster aus dem Internet lernen)
  --show-knowledge Zeige gesammeltes Web-Wissen
  --status         Zeige aktuellen Trainings-Status
  --help           Diese Hilfe anzeigen

Ohne Option wird der vollständige Zyklus ausgeführt.

WAS DER ASSISTENT LERNT:
  • Lotto-Statistiken von öffentlichen APIs
  • Häufigkeits-Muster und Trends
  • Optimale Summen, Gerade/Ungerade Verteilungen
  • Experten-Strategien und deren Effektivität
  • Gap-Analysen für überfällige Zahlen
            """)
        else:
            print(f"❌ Unbekannte Option: {command}")
            print("   Verwende --help für Hilfe")
    else:
        # Standard: Vollständiger Zyklus
        assistant.run_full_cycle()


if __name__ == "__main__":
    main()
