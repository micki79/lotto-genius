#!/usr/bin/env python3
"""
🧠 LottoGenius - MASTER AI SYSTEM

Zentrale KI, die alle 5 Spiele verwaltet und automatisch lernt:

1. ÜBERWACHT alle Spiele gleichzeitig
2. LERNT kontinuierlich aus jeder Ziehung
3. WÄHLT automatisch die besten Strategien aus
4. OPTIMIERT Modell-Gewichte über alle Spiele
5. TRACKT Performance global

Das System wird nach jeder Ziehung automatisch besser!

Verwendung:
    python master_ai.py                    # Vorhersagen für alle Spiele
    python master_ai.py --learn            # Lernen aus neuen Ziehungen
    python master_ai.py --train            # Vollständiges Training
    python master_ai.py --status           # Zeige KI-Status
    python master_ai.py --auto             # Automatischer Modus (Vorhersagen + Lernen)
"""
import json
import os
import sys
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import random

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

# =====================================================
# KONFIGURATION
# =====================================================

GAMES = {
    'lotto_6aus49': {
        'name': 'Lotto 6aus49',
        'emoji': '🎱',
        'data_file': 'lotto_data.json',
        'predictions_file': 'predictions.json',
        'type': 'numbers',  # 6 aus 49
        'draw_days': [2, 5],  # Mittwoch, Samstag
    },
    'eurojackpot': {
        'name': 'Eurojackpot',
        'emoji': '🌟',
        'data_file': 'eurojackpot_data.json',
        'predictions_file': 'eurojackpot_predictions.json',
        'type': 'eurojackpot',  # 5 aus 50 + 2 aus 12
        'draw_days': [1, 4],  # Dienstag, Freitag
    },
    'spiel77': {
        'name': 'Spiel 77',
        'emoji': '🎰',
        'data_file': 'spiel77_data.json',
        'predictions_file': 'spiel77_predictions.json',
        'type': 'digits',
        'num_digits': 7,
        'draw_days': [2, 5],
    },
    'super6': {
        'name': 'Super 6',
        'emoji': '🎲',
        'data_file': 'super6_data.json',
        'predictions_file': 'super6_predictions.json',
        'type': 'digits',
        'num_digits': 6,
        'draw_days': [2, 5],
    },
    'gluecksspirale': {
        'name': 'Glücksspirale',
        'emoji': '🌀',
        'data_file': 'gluecksspirale_data.json',
        'predictions_file': 'gluecksspirale_predictions.json',
        'type': 'digits',
        'num_digits': 7,
        'draw_days': [5],  # Nur Samstag
    }
}


def load_json(filename, default=None):
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except:
            pass
    return default if default else {}


def save_json(filename, data):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, filename), 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


class MasterAI:
    """
    Zentrale KI, die alle Spiele verwaltet und optimiert.

    Features:
    - Globales Strategie-Ranking über alle Spiele
    - Automatische Modell-Optimierung
    - Cross-Game Learning (lernt Muster die für alle Spiele gelten)
    - Intelligente Gewichtung basierend auf Performance
    """

    STATUS_FILE = 'master_ai_status.json'

    def __init__(self):
        self.status = self.load_status()
        self.games = GAMES

    def load_status(self):
        """Lädt den Master-AI Status"""
        default = {
            'initialized': datetime.now().isoformat(),
            'last_update': None,
            'total_predictions': 0,
            'total_correct': 0,
            'games': {},
            'global_strategy_weights': {},
            'model_performance': {
                'neural_network': {'score': 1.0, 'predictions': 0, 'hits': 0},
                'markov': {'score': 1.0, 'predictions': 0, 'hits': 0},
                'bayesian': {'score': 1.0, 'predictions': 0, 'hits': 0},
                'ensemble': {'score': 1.0, 'predictions': 0, 'hits': 0},
            },
            'learning_rate': 0.1,
            'best_strategies': [],
            'version': '2.0'
        }
        return load_json(self.STATUS_FILE, default)

    def save_status(self):
        """Speichert den Master-AI Status"""
        self.status['last_update'] = datetime.now().isoformat()
        save_json(self.STATUS_FILE, self.status)

    def get_next_draws(self):
        """Findet die nächsten Ziehungstermine für alle Spiele"""
        today = datetime.now()
        next_draws = {}

        for game_id, config in self.games.items():
            draw_days = config['draw_days']

            # Finde nächsten Ziehungstag
            days_ahead = float('inf')
            for draw_day in draw_days:
                diff = (draw_day - today.weekday() + 7) % 7
                if diff == 0 and today.hour >= 20:  # Nach 20 Uhr = nächste Woche
                    diff = 7
                if diff < days_ahead:
                    days_ahead = diff

            next_draw = today + timedelta(days=days_ahead)
            next_draws[game_id] = {
                'date': next_draw.strftime('%d.%m.%Y'),
                'days_until': days_ahead
            }

        return next_draws

    def generate_all_predictions(self):
        """Generiert Vorhersagen für alle Spiele mit der Master-AI"""

        print("=" * 70)
        print("🧠 LOTTOGENIUS MASTER AI - ZENTRALE VORHERSAGEN")
        print("=" * 70)
        print(f"📅 {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
        print()

        next_draws = self.get_next_draws()
        all_predictions = {}

        # Importiere ML-Module
        try:
            from ml_models import (
                get_ml_predictions,
                get_eurojackpot_ml_predictions,
                get_digit_game_ml_predictions
            )
            ML_AVAILABLE = True
        except ImportError as e:
            print(f"⚠️ ML-Module nicht verfügbar: {e}")
            ML_AVAILABLE = False

        for game_id, config in self.games.items():
            print(f"\n{'='*60}")
            print(f"{config['emoji']} {config['name']}")
            print(f"   📅 Nächste Ziehung: {next_draws[game_id]['date']} ({next_draws[game_id]['days_until']} Tage)")
            print('='*60)

            # Lade Daten
            data = load_json(config['data_file'], {'draws': []})
            draws = data.get('draws', [])

            if not draws:
                print(f"   ⚠️ Keine Daten vorhanden")
                continue

            print(f"   📊 {len(draws)} historische Ziehungen")

            predictions = []

            if ML_AVAILABLE:
                try:
                    if config['type'] == 'numbers':
                        # Lotto 6aus49
                        ml_preds = get_ml_predictions(draws)
                        for p in ml_preds:
                            predictions.append({
                                'numbers': p.get('numbers', []),
                                'superzahl': p.get('superzahl'),
                                'method': p.get('method', 'unknown'),
                                'confidence': p.get('confidence', 50),
                                'is_ml': True
                            })

                    elif config['type'] == 'eurojackpot':
                        ml_preds = get_eurojackpot_ml_predictions(draws)
                        for p in ml_preds:
                            predictions.append({
                                'numbers': p.get('numbers', []),
                                'eurozahlen': p.get('eurozahlen', []),
                                'method': p.get('method', 'unknown'),
                                'confidence': p.get('confidence', 50),
                                'is_ml': True
                            })

                    elif config['type'] == 'digits':
                        ml_preds = get_digit_game_ml_predictions(
                            game_id,
                            config['num_digits'],
                            draws
                        )
                        for p in ml_preds:
                            predictions.append({
                                'number': p.get('number', ''),
                                'method': p.get('method', 'unknown'),
                                'confidence': p.get('confidence', 50),
                                'is_ml': True
                            })

                    print(f"   ✅ {len(predictions)} ML-Vorhersagen generiert")

                except Exception as e:
                    print(f"   ❌ ML-Fehler: {e}")

            # Wähle beste Vorhersage basierend auf Master-AI Gewichten
            if predictions:
                best = self._select_best_prediction(predictions, game_id)
                all_predictions[game_id] = {
                    'best': best,
                    'all': predictions,
                    'next_draw': next_draws[game_id]['date']
                }

                # Zeige beste Vorhersage
                print(f"\n   🏆 MASTER AI EMPFEHLUNG:")
                if 'numbers' in best:
                    nums = best['numbers']
                    if 'eurozahlen' in best:
                        print(f"      Zahlen: {nums} | Euro: {best['eurozahlen']}")
                    elif 'superzahl' in best:
                        print(f"      Zahlen: {nums} | SZ: {best['superzahl']}")
                    else:
                        print(f"      Zahlen: {nums}")
                elif 'number' in best:
                    print(f"      Zahl: {best['number']}")
                print(f"      Methode: {best.get('method', 'ensemble')}")
                print(f"      Konfidenz: {best.get('confidence', 0):.1f}%")

        # Speichere alle Vorhersagen
        master_predictions = {
            'timestamp': datetime.now().isoformat(),
            'predictions': all_predictions,
            'next_draws': next_draws
        }
        save_json('master_ai_predictions.json', master_predictions)

        # Update Status
        self.status['total_predictions'] += sum(len(p.get('all', [])) for p in all_predictions.values())
        self.save_status()

        print("\n" + "=" * 70)
        print("✅ MASTER AI VORHERSAGEN ABGESCHLOSSEN")
        print("=" * 70)
        print(f"📊 Gesamt: {sum(len(p.get('all', [])) for p in all_predictions.values())} Vorhersagen")
        print(f"💾 Gespeichert in: data/master_ai_predictions.json")
        print()

        return all_predictions

    def _select_best_prediction(self, predictions, game_id):
        """Wählt die beste Vorhersage basierend auf globalen Gewichten"""

        if not predictions:
            return {}

        # Hole globale Strategie-Gewichte
        weights = self.status.get('global_strategy_weights', {})
        model_perf = self.status.get('model_performance', {})

        best_score = -1
        best_pred = predictions[0]

        for pred in predictions:
            method = pred.get('method', 'unknown')
            base_conf = pred.get('confidence', 50)

            # Berechne Score basierend auf:
            # 1. Basis-Konfidenz
            # 2. Historische Performance der Methode
            # 3. Modell-spezifische Performance

            strategy_weight = weights.get(method, 1.0)

            # Prüfe welches ML-Modell verwendet wird
            model_weight = 1.0
            for model_name in ['neural_network', 'markov', 'bayesian', 'ensemble']:
                if model_name in method.lower():
                    model_weight = model_perf.get(model_name, {}).get('score', 1.0)
                    break

            score = base_conf * strategy_weight * model_weight

            if score > best_score:
                best_score = score
                best_pred = pred

        best_pred['master_ai_score'] = best_score
        return best_pred

    def learn_from_all_games(self):
        """Lernt aus allen Spielen gleichzeitig"""

        print("=" * 70)
        print("🧠 MASTER AI - GLOBALES LERNEN")
        print("=" * 70)
        print(f"📅 {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
        print()

        total_learned = 0
        learning_results = {}

        for game_id, config in self.games.items():
            print(f"\n{config['emoji']} {config['name']}:")

            # Lade Daten und Vorhersagen
            data = load_json(config['data_file'], {'draws': []})
            predictions = load_json(config['predictions_file'], {'predictions': []})

            draws = data.get('draws', [])
            preds = predictions.get('predictions', [])

            if not draws:
                print(f"   ⚠️ Keine Daten")
                continue

            last_draw = draws[0]
            unverified = [p for p in preds if not p.get('verified')]

            if not unverified:
                print(f"   ℹ️ Keine unverifizierten Vorhersagen")
                continue

            # Lerne aus jeder Vorhersage
            learned = 0
            for pred in unverified:
                result = self._evaluate_prediction(pred, last_draw, config['type'])
                if result:
                    self._update_global_weights(pred, result)
                    learned += 1
                    pred['verified'] = True
                    pred['result'] = result

            # Speichere aktualisierte Vorhersagen
            save_json(config['predictions_file'], predictions)

            learning_results[game_id] = {
                'learned': learned,
                'draw_date': last_draw.get('date', 'unknown')
            }
            total_learned += learned

            print(f"   ✅ {learned} Vorhersagen gelernt")
            print(f"   📊 Letzte Ziehung: {last_draw.get('date', 'unbekannt')}")

        # Update Model Performance
        self._update_model_performance()

        # Finde beste Strategien global
        self._update_best_strategies()

        self.save_status()

        print("\n" + "=" * 70)
        print("📊 GLOBALES LEARNING ZUSAMMENFASSUNG")
        print("=" * 70)
        print(f"✅ Gesamt gelernt: {total_learned} Vorhersagen")
        print()

        print("🏆 Top 5 Globale Strategien:")
        for i, strat in enumerate(self.status.get('best_strategies', [])[:5], 1):
            print(f"   {i}. {strat['name']}: Score={strat['score']:.3f}")

        print()
        print("🤖 Modell-Performance:")
        for model, perf in self.status.get('model_performance', {}).items():
            score = perf.get('score', 1.0)
            preds = perf.get('predictions', 0)
            print(f"   • {model}: {score:.3f} ({preds} Vorhersagen)")

        print()

        return learning_results

    def _evaluate_prediction(self, pred, actual_draw, game_type):
        """Bewertet eine Vorhersage gegen das echte Ergebnis"""

        if game_type == 'numbers':
            # Lotto 6aus49
            pred_nums = set(pred.get('numbers', []))
            actual_nums = set(actual_draw.get('numbers', []))
            matches = len(pred_nums & actual_nums)

            sz_match = pred.get('superzahl') == actual_draw.get('superzahl')

            return {
                'matches': matches,
                'superzahl_match': sz_match,
                'score': matches + (0.5 if sz_match else 0)
            }

        elif game_type == 'eurojackpot':
            pred_nums = set(pred.get('numbers', []))
            actual_nums = set(actual_draw.get('numbers', []))
            main_matches = len(pred_nums & actual_nums)

            pred_euro = set(pred.get('eurozahlen', []))
            actual_euro = set(actual_draw.get('eurozahlen', []))
            euro_matches = len(pred_euro & actual_euro)

            return {
                'main_matches': main_matches,
                'euro_matches': euro_matches,
                'score': main_matches + euro_matches * 0.5
            }

        elif game_type == 'digits':
            pred_num = str(pred.get('number', '')).zfill(7)
            actual_num = str(actual_draw.get('number', '')).zfill(7)

            # Zähle Endziffern von rechts
            end_matches = 0
            for p, a in zip(pred_num[::-1], actual_num[::-1]):
                if p == a:
                    end_matches += 1
                else:
                    break

            return {
                'end_matches': end_matches,
                'score': end_matches
            }

        return None

    def _update_global_weights(self, pred, result):
        """Aktualisiert globale Strategie-Gewichte"""

        method = pred.get('method', 'unknown')
        score = result.get('score', 0)

        weights = self.status.get('global_strategy_weights', {})

        if method not in weights:
            weights[method] = 1.0

        # Exponentieller gleitender Durchschnitt
        lr = self.status.get('learning_rate', 0.1)
        old_weight = weights[method]

        # Score normalisieren (0-1)
        normalized_score = min(score / 6, 1.0)  # Max 6 Treffer

        # Neues Gewicht = alter Wert + Learning Rate * (Score - alter Wert)
        new_weight = old_weight + lr * (normalized_score - old_weight + 0.5)
        new_weight = max(0.1, min(3.0, new_weight))

        weights[method] = new_weight
        self.status['global_strategy_weights'] = weights

        # Update Model Performance
        model_perf = self.status.get('model_performance', {})
        for model_name in ['neural_network', 'markov', 'bayesian', 'ensemble']:
            if model_name in method.lower():
                if model_name not in model_perf:
                    model_perf[model_name] = {'score': 1.0, 'predictions': 0, 'hits': 0}

                model_perf[model_name]['predictions'] += 1
                model_perf[model_name]['hits'] += score

                # Update Score
                total = model_perf[model_name]['predictions']
                hits = model_perf[model_name]['hits']
                model_perf[model_name]['score'] = 0.5 + (hits / max(total, 1)) / 6
                break

        self.status['model_performance'] = model_perf

    def _update_model_performance(self):
        """Aktualisiert die globale Modell-Performance"""
        pass  # Bereits in _update_global_weights

    def _update_best_strategies(self):
        """Findet die besten globalen Strategien"""

        weights = self.status.get('global_strategy_weights', {})

        sorted_strategies = sorted(
            weights.items(),
            key=lambda x: x[1],
            reverse=True
        )

        self.status['best_strategies'] = [
            {'name': name, 'score': score}
            for name, score in sorted_strategies[:20]
        ]

    def train_all_models(self):
        """Trainiert alle ML-Modelle für alle Spiele"""

        print("=" * 70)
        print("🧠 MASTER AI - VOLLSTÄNDIGES ML-TRAINING")
        print("=" * 70)

        try:
            from ml_models import train_all_models
            from ml_models import train_eurojackpot_ml, train_digit_game_ml
        except ImportError as e:
            print(f"❌ ML-Module nicht verfügbar: {e}")
            return

        # Lotto 6aus49
        print(f"\n🎱 Lotto 6aus49...")
        data = load_json('lotto_data.json', {'draws': []})
        if data.get('draws'):
            train_all_models(data['draws'])

        # Eurojackpot
        print(f"\n🌟 Eurojackpot...")
        data = load_json('eurojackpot_data.json', {'draws': []})
        if data.get('draws'):
            train_eurojackpot_ml(data['draws'])

        # Spiel 77
        print(f"\n🎰 Spiel 77...")
        data = load_json('spiel77_data.json', {'draws': []})
        if data.get('draws'):
            train_digit_game_ml('spiel77', 7, data['draws'])

        # Super 6
        print(f"\n🎲 Super 6...")
        data = load_json('super6_data.json', {'draws': []})
        if data.get('draws'):
            train_digit_game_ml('super6', 6, data['draws'])

        # Glücksspirale
        print(f"\n🌀 Glücksspirale...")
        data = load_json('gluecksspirale_data.json', {'draws': []})
        if data.get('draws'):
            train_digit_game_ml('gluecksspirale', 7, data['draws'])

        self.save_status()

        print("\n" + "=" * 70)
        print("✅ VOLLSTÄNDIGES TRAINING ABGESCHLOSSEN")
        print("=" * 70)

    def show_status(self):
        """Zeigt den aktuellen Master-AI Status"""

        print("=" * 70)
        print("🧠 MASTER AI STATUS")
        print("=" * 70)
        print()

        print(f"📅 Initialisiert: {self.status.get('initialized', 'unbekannt')}")
        print(f"📅 Letztes Update: {self.status.get('last_update', 'nie')}")
        print(f"📊 Gesamt Vorhersagen: {self.status.get('total_predictions', 0)}")
        print()

        print("🎮 Spiele:")
        for game_id, config in self.games.items():
            data = load_json(config['data_file'], {'draws': []})
            count = len(data.get('draws', []))
            print(f"   {config['emoji']} {config['name']}: {count} Ziehungen")

        print()
        print("🤖 Modell-Performance:")
        for model, perf in self.status.get('model_performance', {}).items():
            score = perf.get('score', 1.0)
            preds = perf.get('predictions', 0)
            bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
            print(f"   {model:15} [{bar}] {score:.3f} ({preds})")

        print()
        print("🏆 Top 10 Globale Strategien:")
        for i, strat in enumerate(self.status.get('best_strategies', [])[:10], 1):
            print(f"   {i:2}. {strat['name']}: {strat['score']:.3f}")

        print()

        # Nächste Ziehungen
        next_draws = self.get_next_draws()
        print("📅 Nächste Ziehungen:")
        for game_id, info in sorted(next_draws.items(), key=lambda x: x[1]['days_until']):
            config = self.games[game_id]
            days = info['days_until']
            if days == 0:
                days_str = "HEUTE!"
            elif days == 1:
                days_str = "morgen"
            else:
                days_str = f"in {days} Tagen"
            print(f"   {config['emoji']} {config['name']}: {info['date']} ({days_str})")

        print()

    def auto_mode(self):
        """Automatischer Modus: Vorhersagen + Lernen"""

        print("=" * 70)
        print("🤖 MASTER AI - AUTOMATISCHER MODUS")
        print("=" * 70)
        print()

        # 1. Zeige Status
        self.show_status()

        # 2. Lerne aus neuen Ziehungen
        print("\n" + "=" * 70)
        print("📚 PHASE 1: LERNEN")
        print("=" * 70)
        self.learn_from_all_games()

        # 3. Generiere neue Vorhersagen
        print("\n" + "=" * 70)
        print("🎯 PHASE 2: VORHERSAGEN")
        print("=" * 70)
        self.generate_all_predictions()

        print("\n" + "=" * 70)
        print("✅ AUTOMATISCHER MODUS ABGESCHLOSSEN")
        print("=" * 70)
        print()
        print("💡 Die Master-AI hat gelernt und neue Vorhersagen generiert.")
        print("   Die besten Strategien werden automatisch bevorzugt!")
        print()


def main():
    """Hauptfunktion"""

    ai = MasterAI()

    if '--status' in sys.argv:
        ai.show_status()
    elif '--learn' in sys.argv:
        ai.learn_from_all_games()
    elif '--train' in sys.argv:
        ai.train_all_models()
    elif '--auto' in sys.argv:
        ai.auto_mode()
    else:
        # Standard: Vorhersagen generieren
        ai.generate_all_predictions()


if __name__ == "__main__":
    main()
