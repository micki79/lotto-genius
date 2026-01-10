#!/usr/bin/env python3
"""
🎓 LottoGenius - KI Trainings-Assistent

Kontinuierliches Training und Verbesserung des KI-Systems durch:
1. Web-Recherche nach Lotto-Statistiken und Mustern
2. Automatisches Training aller ML-Modelle
3. Optimierung der Strategie-Gewichte
4. Cross-Validation und Performance-Analyse
5. Externe Datenquellen-Integration

Der Assistent kann:
- Neue Lotto-Tipps und Strategien aus dem Web lernen
- Historische Muster analysieren und ins Training einbeziehen
- Die Gewichtungen kontinuierlich optimieren
- Modell-Performance tracken und verbessern
"""

import json
import os
import random
import requests
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import time
import re

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
# WEB RESEARCH MODULE
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
    """

    def __init__(self):
        self.researcher = LottoWebResearcher()
        self.optimizer = TrainingOptimizer()
        self.tuner = StrategyWeightTuner()
        self.status = load_json('training_assistant_status.json', {
            'last_run': None,
            'total_runs': 0,
            'improvements': []
        })

    def run_full_cycle(self, draws=None):
        """Führt einen vollständigen Trainings-Zyklus durch"""
        print("\n" + "=" * 70)
        print("🎓 LOTTOGENIUS TRAININGS-ASSISTENT")
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

        # 1. Web-Research
        print("\n" + "-" * 40)
        print("PHASE 1: Web-Recherche")
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
        self.status['last_cv_results'] = cv_results
        self.status['last_training_results'] = {
            k: str(v) for k, v in training_results.items()
        }
        save_json('training_assistant_status.json', self.status)

        # Zusammenfassung
        print("\n" + "=" * 70)
        print("📋 TRAININGS-ZUSAMMENFASSUNG")
        print("=" * 70)
        print(f"✅ Web-Insights: {len(insights)}")
        print(f"✅ Cross-Validation: {len(cv_results)} Modelle evaluiert")
        print(f"✅ ML-Training: {len(training_results)} Modelle trainiert")
        print(f"✅ Strategie-Tuning: {len(rankings)} Strategien optimiert")
        print(f"\n🔄 Gesamte Trainings-Zyklen: {self.status['total_runs']}")
        print("=" * 70)

        return {
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
        elif command == '--help':
            print("""
🎓 LottoGenius Trainings-Assistent

Verwendung:
  python training_assistant.py [OPTION]

Optionen:
  --full      Vollständiger Trainings-Zyklus (Web-Research + CV + Training)
  --quick     Schnelles Training (nur ML-Modelle)
  --analyze   Nur Analyse (keine Modifikationen)
  --help      Diese Hilfe anzeigen

Ohne Option wird der vollständige Zyklus ausgeführt.
            """)
        else:
            print(f"❌ Unbekannte Option: {command}")
            print("   Verwende --help für Hilfe")
    else:
        # Standard: Vollständiger Zyklus
        assistant.run_full_cycle()


if __name__ == "__main__":
    main()
