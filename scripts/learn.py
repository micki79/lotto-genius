#!/usr/bin/env python3
"""
🧠 LottoGenius - Kontinuierliches Lern-System mit ECHTEM ML

Lernt aus jeder Ziehung:
1. Vergleicht Vorhersagen mit echten Zahlen
2. Aktualisiert Provider-Scores (welche KI am besten ist)
3. Trackt Superzahl-Erfolge separat
4. Berechnet Methoden-Rankings
5. AKTUALISIERT STRATEGIE-GEWICHTE (Selbstlernen!)
6. Erkennt Gewinnklassen
7. Speichert alles für langfristiges Lernen
8. *** NEU: TRAINIERT ECHTE ML-MODELLE ***
   - Neuronales Netz mit Backpropagation
   - Markov-Ketten
   - Bayesian Learning
   - Reinforcement Learning
"""
import json
import os
import sys
from datetime import datetime
from collections import Counter

# Füge scripts-Verzeichnis zum Pfad hinzu für ML-Modelle Import
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Importiere echte ML-Modelle
try:
    from ml_models import (
        NeuralNetwork, MarkovChain, BayesianPredictor,
        ReinforcementLearner, EnsembleML, SuperzahlML,
        train_all_models, learn_from_new_draw, get_ml_predictions
    )
    ML_AVAILABLE = True
except ImportError as e:
    ML_AVAILABLE = False
    print(f"⚠️ ML-Modelle nicht verfügbar: {e}")

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

# Lotto 6aus49 Gewinnklassen
GEWINNKLASSEN = {
    1: {'matches': 6, 'superzahl': True, 'name': 'Jackpot (6+SZ)', 'weight_boost': 5.0},
    2: {'matches': 6, 'superzahl': False, 'name': '6 Richtige', 'weight_boost': 4.0},
    3: {'matches': 5, 'superzahl': True, 'name': '5+SZ', 'weight_boost': 3.0},
    4: {'matches': 5, 'superzahl': False, 'name': '5 Richtige', 'weight_boost': 2.5},
    5: {'matches': 4, 'superzahl': True, 'name': '4+SZ', 'weight_boost': 2.0},
    6: {'matches': 4, 'superzahl': False, 'name': '4 Richtige', 'weight_boost': 1.5},
    7: {'matches': 3, 'superzahl': True, 'name': '3+SZ', 'weight_boost': 1.3},
    8: {'matches': 3, 'superzahl': False, 'name': '3 Richtige', 'weight_boost': 1.2},
    9: {'matches': 2, 'superzahl': True, 'name': '2+SZ', 'weight_boost': 1.1}
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

def determine_gewinnklasse(matches, sz_match):
    """Bestimmt die Gewinnklasse basierend auf Treffern und Superzahl"""
    for klasse, criteria in GEWINNKLASSEN.items():
        if matches == criteria['matches'] and sz_match == criteria['superzahl']:
            return klasse, criteria
    return None, None

def calculate_new_weight(old_weight, hits, total, learning_rate=0.15):
    """
    Berechnet neues Gewicht mit Exponential Moving Average.
    Strategien mit mehr Treffern bekommen höhere Gewichte.
    """
    if total == 0:
        return old_weight

    # Treffer-Rate (0-1)
    hit_rate = hits / total

    # Exponential Moving Average
    new_weight = old_weight * (1 - learning_rate) + hit_rate * 10 * learning_rate

    # Begrenzen auf sinnvolle Werte
    return max(0.1, min(5.0, new_weight))

class StrategyWeightUpdater:
    """Aktualisiert Strategie-Gewichte basierend auf Performance"""

    def __init__(self):
        self.weights_file = 'strategy_weights.json'
        self.weights = self.load_weights()

    def load_weights(self):
        """Lädt Gewichtungen"""
        data = load_json(self.weights_file, {})
        if not data.get('strategies'):
            # Initialisiere mit Standardwerten
            data = {
                'strategies': {},
                'last_update': datetime.now().isoformat(),
                'total_predictions': 0,
                'learning_rate': 0.15
            }
        return data

    def update_strategy(self, strategy_name, matches, sz_match, gewinnklasse=None):
        """Aktualisiert eine Strategie basierend auf Ergebnis"""
        if strategy_name not in self.weights['strategies']:
            self.weights['strategies'][strategy_name] = {
                'weight': 1.0,
                'hits': 0,
                'total': 0,
                'three_plus': 0,
                'four_plus': 0,
                'sz_correct': 0,
                'gewinnklassen': {},
                'last_updated': None
            }

        strat = self.weights['strategies'][strategy_name]

        # Stelle sicher dass alle Felder existieren
        if 'total' not in strat:
            strat['total'] = 0
        if 'hits' not in strat:
            strat['hits'] = 0
        if 'three_plus' not in strat:
            strat['three_plus'] = 0
        if 'four_plus' not in strat:
            strat['four_plus'] = 0
        if 'sz_correct' not in strat:
            strat['sz_correct'] = 0
        if 'gewinnklassen' not in strat:
            strat['gewinnklassen'] = {}

        strat['total'] += 1
        strat['hits'] += matches

        if matches >= 3:
            strat['three_plus'] += 1
        if matches >= 4:
            strat['four_plus'] += 1
        if sz_match:
            strat['sz_correct'] += 1

        # Gewinnklasse tracken
        if gewinnklasse:
            klasse_key = str(gewinnklasse)
            if klasse_key not in strat['gewinnklassen']:
                strat['gewinnklassen'][klasse_key] = 0
            strat['gewinnklassen'][klasse_key] += 1

        # Berechne neues Gewicht
        old_weight = strat['weight']
        hit_rate = strat['hits'] / (strat['total'] * 6) if strat['total'] > 0 else 0

        # Basis-Gewicht aus Hit-Rate
        lr = self.weights.get('learning_rate', 0.15)
        new_weight = old_weight * (1 - lr) + hit_rate * 30 * lr

        # Bonus für Gewinnklassen
        if gewinnklasse:
            boost = GEWINNKLASSEN.get(gewinnklasse, {}).get('weight_boost', 1.0)
            new_weight *= boost

        # Begrenzen
        strat['weight'] = max(0.1, min(5.0, new_weight))
        strat['last_updated'] = datetime.now().isoformat()

        return strat['weight']

    def save(self):
        """Speichert Gewichtungen"""
        self.weights['last_update'] = datetime.now().isoformat()
        self.weights['total_predictions'] = sum(
            s.get('total', 0) for s in self.weights['strategies'].values()
        )
        save_json(self.weights_file, self.weights)

    def get_ranking(self, top_n=10):
        """Gibt Ranking der besten Strategien zurück"""
        strategies = self.weights.get('strategies', {})
        sorted_strats = sorted(
            strategies.items(),
            key=lambda x: x[1].get('weight', 1.0),
            reverse=True
        )
        return sorted_strats[:top_n]

def learn_from_results():
    """Hauptfunktion: Lernt aus vergangenen Vorhersagen"""

    print("=" * 60)
    print("🧠 LottoGenius - Lern-System mit Selbstlernen")
    print("=" * 60)
    print(f"📅 {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
    print()

    # Lade alle Daten
    lotto_data = load_json('lotto_data.json', {'draws': []})
    predictions = load_json('predictions.json', {'predictions': [], 'history': []})
    learning = load_json('learning.json', {'entries': [], 'stats': {}, 'by_method': {}, 'by_provider': {}})
    provider_scores = load_json('provider_scores.json', {})
    superzahl_history = load_json('superzahl_history.json', {'entries': [], 'stats': {}})

    # Initialisiere Weight Updater
    weight_updater = StrategyWeightUpdater()

    draws = lotto_data.get('draws', [])
    if not draws:
        print("⚠️ Keine Lotto-Daten vorhanden")
        return

    last_draw = draws[0]
    actual_numbers = set(last_draw.get('numbers', []))
    actual_sz = last_draw.get('superzahl', 0)
    draw_date = last_draw.get('date', 'unbekannt')

    print(f"📅 Letzte Ziehung: {draw_date}")
    print(f"🎱 Zahlen: {sorted(actual_numbers)} | Superzahl: {actual_sz}")
    print()

    # Finde unverarbeitete Vorhersagen
    unverified = [p for p in predictions.get('predictions', []) if not p.get('verified')]

    if not unverified:
        print("ℹ️ Keine unverifizierten Vorhersagen vorhanden")
        # Schaue auch in history
        recent_history = [p for p in predictions.get('history', [])[-50:] if not p.get('verified')]
        if recent_history:
            unverified = recent_history
            print(f"📚 {len(unverified)} Vorhersagen aus Historie gefunden")
        else:
            print("📚 Auch keine unverifizierten in Historie")
            return

    print(f"📊 Verarbeite {len(unverified)} Vorhersagen...")
    print("-" * 40)

    new_entries = []
    method_updates = {}
    provider_updates = {}
    sz_entries = []
    gewinnklassen_found = []

    for pred in unverified:
        pred_numbers = set(pred.get('numbers', []))
        pred_sz = pred.get('superzahl')
        method = pred.get('method', 'unknown')
        provider = pred.get('provider', method)

        # Berechne Treffer
        matches = len(pred_numbers & actual_numbers)
        sz_match = pred_sz == actual_sz

        # Bestimme Gewinnklasse
        gewinnklasse, gk_info = determine_gewinnklasse(matches, sz_match)

        # Aktualisiere Strategie-Gewicht (SELBSTLERNEN!)
        if method and method != 'unknown':
            new_weight = weight_updater.update_strategy(method, matches, sz_match, gewinnklasse)
        else:
            new_weight = 1.0

        # Erstelle Lern-Eintrag
        entry = {
            'id': f"{draw_date}_{method}_{datetime.now().timestamp()}",
            'date': datetime.now().isoformat(),
            'draw_date': draw_date,
            'predicted_numbers': list(pred_numbers),
            'actual_numbers': list(actual_numbers),
            'matches': matches,
            'predicted_sz': pred_sz,
            'actual_sz': actual_sz,
            'sz_match': sz_match,
            'method': method,
            'provider': provider,
            'confidence': pred.get('confidence', 50),
            'strategy': pred.get('strategy', ''),
            'gewinnklasse': gewinnklasse,
            'new_weight': new_weight
        }
        new_entries.append(entry)

        # Gewinnklasse tracking
        if gewinnklasse:
            gewinnklassen_found.append({
                'klasse': gewinnklasse,
                'name': gk_info['name'],
                'method': method,
                'numbers': list(pred_numbers),
                'matches': matches,
                'sz_match': sz_match
            })

        # Superzahl-Eintrag
        sz_entries.append({
            'date': draw_date,
            'predicted': pred_sz,
            'actual': actual_sz,
            'correct': sz_match,
            'method': method
        })

        # Methoden-Statistik
        if method not in method_updates:
            method_updates[method] = {
                'predictions': 0,
                'total_matches': 0,
                'sz_correct': 0,
                'three_plus': 0,
                'four_plus': 0,
                'gewinnklassen': {}
            }
        method_updates[method]['predictions'] += 1
        method_updates[method]['total_matches'] += matches
        if sz_match:
            method_updates[method]['sz_correct'] += 1
        if matches >= 3:
            method_updates[method]['three_plus'] += 1
        if matches >= 4:
            method_updates[method]['four_plus'] += 1
        if gewinnklasse:
            gk_key = str(gewinnklasse)
            if gk_key not in method_updates[method]['gewinnklassen']:
                method_updates[method]['gewinnklassen'][gk_key] = 0
            method_updates[method]['gewinnklassen'][gk_key] += 1

        # Provider-Statistik
        if provider not in provider_updates:
            provider_updates[provider] = {
                'predictions': 0,
                'total_matches': 0,
                'sz_correct': 0,
                'gewinnklassen': {}
            }
        provider_updates[provider]['predictions'] += 1
        provider_updates[provider]['total_matches'] += matches
        if sz_match:
            provider_updates[provider]['sz_correct'] += 1
        if gewinnklasse:
            gk_key = str(gewinnklasse)
            if gk_key not in provider_updates[provider]['gewinnklassen']:
                provider_updates[provider]['gewinnklassen'][gk_key] = 0
            provider_updates[provider]['gewinnklassen'][gk_key] += 1

        # Markiere als verifiziert
        pred['verified'] = True
        pred['result'] = {
            'matches': matches,
            'sz_match': sz_match,
            'draw_date': draw_date,
            'gewinnklasse': gewinnklasse
        }

        # Output
        if gewinnklasse:
            match_indicator = f"🏆 KLASSE {gewinnklasse}"
        elif matches >= 3:
            match_indicator = "🎯"
        elif matches >= 2:
            match_indicator = "✓"
        else:
            match_indicator = "•"

        sz_indicator = "✓SZ" if sz_match else ""
        weight_str = f"[W:{new_weight:.2f}]" if method != 'unknown' else ""
        print(f"  {match_indicator} {method}: {matches}/6 {sz_indicator} {weight_str}")

    print("-" * 40)

    # Speichere Weight Updates
    weight_updater.save()

    # Zeige gefundene Gewinnklassen
    if gewinnklassen_found:
        print()
        print("🏆 GEWINNKLASSEN GEFUNDEN:")
        for gk in gewinnklassen_found:
            print(f"   • Klasse {gk['klasse']} ({gk['name']}): {gk['method']}")
            print(f"     Zahlen: {gk['numbers']} ({gk['matches']} Treffer)")

    # Speichere neue Lern-Einträge
    learning['entries'].extend(new_entries)
    learning['entries'] = learning['entries'][-2000:]  # Behalte letzte 2000

    # Aktualisiere Methoden-Statistiken
    for method, update in method_updates.items():
        if method not in learning['by_method']:
            learning['by_method'][method] = {
                'total_predictions': 0,
                'total_matches': 0,
                'total_possible': 0,
                'sz_correct': 0,
                'three_plus': 0,
                'four_plus': 0,
                'gewinnklassen': {}
            }

        m = learning['by_method'][method]
        m['total_predictions'] += update['predictions']
        m['total_matches'] += update['total_matches']
        m['total_possible'] += update['predictions'] * 6
        m['sz_correct'] += update['sz_correct']
        m['three_plus'] += update['three_plus']
        m['four_plus'] += update['four_plus']

        # Gewinnklassen
        for gk_key, count in update['gewinnklassen'].items():
            if gk_key not in m['gewinnklassen']:
                m['gewinnklassen'][gk_key] = 0
            m['gewinnklassen'][gk_key] += count

        # Berechne Accuracy
        if m['total_possible'] > 0:
            m['accuracy'] = round((m['total_matches'] / m['total_possible']) * 100, 2)
        if m['total_predictions'] > 0:
            m['sz_accuracy'] = round((m['sz_correct'] / m['total_predictions']) * 100, 2)
            m['three_plus_rate'] = round((m['three_plus'] / m['total_predictions']) * 100, 2)

    # Aktualisiere Provider-Scores
    for provider, update in provider_updates.items():
        if provider not in provider_scores:
            provider_scores[provider] = {
                'total_predictions': 0,
                'total_matches': 0,
                'sz_correct': 0,
                'accuracy': 0,
                'sz_accuracy': 0,
                'gewinnklassen': {}
            }

        p = provider_scores[provider]
        p['total_predictions'] += update['predictions']
        p['total_matches'] += update['total_matches']
        p['sz_correct'] += update['sz_correct']

        # Gewinnklassen
        for gk_key, count in update.get('gewinnklassen', {}).items():
            if gk_key not in p.get('gewinnklassen', {}):
                if 'gewinnklassen' not in p:
                    p['gewinnklassen'] = {}
                p['gewinnklassen'][gk_key] = 0
            p['gewinnklassen'][gk_key] += count

        if p['total_predictions'] > 0:
            p['accuracy'] = round((p['total_matches'] / (p['total_predictions'] * 6)) * 100, 2)
            p['sz_accuracy'] = round((p['sz_correct'] / p['total_predictions']) * 100, 2)

    # Superzahl-Historie
    superzahl_history['entries'].extend(sz_entries)
    superzahl_history['entries'] = superzahl_history['entries'][-500:]

    # Superzahl-Statistik
    sz_total = len(superzahl_history['entries'])
    sz_correct = sum(1 for e in superzahl_history['entries'] if e.get('correct'))
    superzahl_history['stats'] = {
        'total': sz_total,
        'correct': sz_correct,
        'accuracy': round((sz_correct / sz_total * 100), 2) if sz_total > 0 else 0,
        'last_update': datetime.now().isoformat()
    }

    # Gesamt-Statistik
    total_entries = len(learning['entries'])
    if total_entries > 0:
        total_matches = sum(e.get('matches', 0) for e in learning['entries'])
        total_sz_correct = sum(1 for e in learning['entries'] if e.get('sz_match'))
        three_plus = sum(1 for e in learning['entries'] if e.get('matches', 0) >= 3)
        four_plus = sum(1 for e in learning['entries'] if e.get('matches', 0) >= 4)

        # Zähle Gewinnklassen
        gewinnklassen_total = {}
        for e in learning['entries']:
            gk = e.get('gewinnklasse')
            if gk:
                gk_key = str(gk)
                if gk_key not in gewinnklassen_total:
                    gewinnklassen_total[gk_key] = 0
                gewinnklassen_total[gk_key] += 1

        learning['stats'] = {
            'total_entries': total_entries,
            'avg_matches': round(total_matches / total_entries, 2),
            'sz_accuracy': round((total_sz_correct / total_entries) * 100, 2),
            'three_plus_rate': round((three_plus / total_entries) * 100, 2),
            'four_plus_rate': round((four_plus / total_entries) * 100, 2),
            'gewinnklassen_total': gewinnklassen_total,
            'last_update': datetime.now().isoformat(),
            'last_draw': draw_date
        }

    # Speichere alles
    save_json('predictions.json', predictions)
    save_json('learning.json', learning)
    save_json('provider_scores.json', provider_scores)
    save_json('superzahl_history.json', superzahl_history)

    # Output Zusammenfassung
    print()
    print(f"✅ {len(new_entries)} Vorhersagen gelernt!")
    print()
    print("📊 Gesamt-Statistik:")
    print(f"   • Einträge: {learning['stats'].get('total_entries', 0)}")
    print(f"   • Ø Treffer: {learning['stats'].get('avg_matches', 0):.2f}")
    print(f"   • SZ-Quote: {learning['stats'].get('sz_accuracy', 0):.1f}%")
    print(f"   • 3+ Treffer: {learning['stats'].get('three_plus_rate', 0):.1f}%")
    print(f"   • 4+ Treffer: {learning['stats'].get('four_plus_rate', 0):.1f}%")

    # Gewinnklassen-Übersicht
    gk_total = learning['stats'].get('gewinnklassen_total', {})
    if gk_total:
        print()
        print("🏆 Gewinnklassen-Statistik:")
        for klasse in sorted(gk_total.keys(), key=int):
            count = gk_total[klasse]
            name = GEWINNKLASSEN.get(int(klasse), {}).get('name', f'Klasse {klasse}')
            print(f"   • Klasse {klasse} ({name}): {count}x")

    print()

    # Strategie-Ranking (mit Gewichten)
    print("📈 Top 10 Strategien (nach Gewicht):")
    top_strategies = weight_updater.get_ranking(10)
    for i, (strat_name, strat_data) in enumerate(top_strategies, 1):
        weight = strat_data.get('weight', 1.0)
        total = strat_data.get('total', 0)
        hits = strat_data.get('hits', 0)
        avg = hits / (total * 6) * 100 if total > 0 else 0
        print(f"   {i:2}. {strat_name}: W={weight:.2f} | {total} Vorhersagen | {avg:.1f}% Treffer")

    print()

    # Provider-Ranking
    print("🏆 Provider-Ranking:")
    ranked_providers = sorted(
        provider_scores.items(),
        key=lambda x: x[1].get('accuracy', 0),
        reverse=True
    )
    for i, (prov, stats) in enumerate(ranked_providers[:5], 1):
        gk_count = sum(stats.get('gewinnklassen', {}).values())
        gk_str = f" | {gk_count} GK" if gk_count > 0 else ""
        print(f"   {i}. {prov}: {stats.get('accuracy', 0):.1f}% | SZ: {stats.get('sz_accuracy', 0):.0f}%{gk_str}")

    print()

    # =====================================================
    # *** ECHTES ML-TRAINING ***
    # =====================================================
    if ML_AVAILABLE:
        print("=" * 60)
        print("🧠 ECHTES ML-TRAINING STARTET")
        print("=" * 60)

        try:
            # Hole vorherige Ziehung für Markov-Kette
            previous_draw = draws[1] if len(draws) > 1 else {}

            # Trainiere alle ML-Modelle inkrementell mit der neuen Ziehung
            print("\n📚 Inkrementelles Training mit neuer Ziehung...")
            ml_results = learn_from_new_draw(draws, last_draw, previous_draw)

            print(f"   ✅ Neural Network: Trainiert auf {last_draw.get('date', 'unbekannt')}")
            print(f"   ✅ Markov-Kette: Übergänge aktualisiert")
            print(f"   ✅ Bayesian: Posterior aktualisiert")
            print(f"   ✅ Superzahl-ML: Modell aktualisiert")

            # Reinforcement Learning - Lerne aus Vorhersage-Ergebnissen
            print("\n🎮 Reinforcement Learning Update...")
            rl = ReinforcementLearner()

            # Erstelle Feature-Vektor für RL
            recent_nums = []
            for d in draws[:30]:
                recent_nums.extend(d.get('numbers', []))
            freq = Counter(recent_nums)
            features = [freq.get(i, 0) / 30 for i in range(1, 11)]  # Top 10 Features

            rl_updates = 0
            for pred in unverified:
                result = rl.learn_from_result(pred, last_draw, features)
                rl_updates += 1

            print(f"   ✅ RL: {rl_updates} Vorhersagen bewertet")
            print(f"   📊 Gesamte Rewards: {rl.total_rewards:.1f}")
            print(f"   🎯 Beste Strategien:")
            for strat in rl.get_best_strategies(3):
                print(f"      • {strat['strategy']}: {strat['value']:.2f}")

            # Ensemble Gewichte aktualisieren
            print("\n🏆 Ensemble-Gewichte aktualisieren...")
            ensemble = EnsembleML()

            # Sammle ML-Vorhersagen für Gewichts-Update
            model_predictions = {}

            # Neural Network
            nn = NeuralNetwork()
            nn_nums, _ = nn.predict(draws)
            model_predictions['neural_network'] = {'numbers': nn_nums}

            # Markov
            markov = MarkovChain()
            markov_nums, _ = markov.predict(previous_draw)
            model_predictions['markov_chain'] = {'numbers': markov_nums}

            # Bayesian
            bayesian = BayesianPredictor()
            bayes_nums, _ = bayesian.predict()
            model_predictions['bayesian'] = {'numbers': bayes_nums}

            # Update Ensemble-Gewichte basierend auf echten Ergebnissen
            ensemble.update_weights_from_result(model_predictions, last_draw)

            print(f"   ✅ Neue Modell-Gewichte:")
            for model, weight in ensemble.model_weights.items():
                print(f"      • {model}: {weight:.3f}")

            # Speichere ML-Status
            ml_status = {
                'last_training': datetime.now().isoformat(),
                'training_draw': draw_date,
                'models': {
                    'neural_network': {
                        'epochs_trained': nn.epochs_trained,
                        'status': 'active'
                    },
                    'markov_chain': {
                        'observations': markov.observations,
                        'status': 'active'
                    },
                    'bayesian': {
                        'observations': bayesian.observations,
                        'status': 'active'
                    },
                    'reinforcement': {
                        'episodes': rl.episodes,
                        'total_rewards': rl.total_rewards,
                        'status': 'active'
                    }
                },
                'ensemble_weights': ensemble.model_weights
            }
            save_json('ml_status.json', ml_status)

            print("\n" + "=" * 60)
            print("✅ ECHTES ML-TRAINING ABGESCHLOSSEN!")
            print("=" * 60)

        except Exception as e:
            print(f"\n❌ ML-Training Fehler: {e}")
            import traceback
            traceback.print_exc()

    else:
        print("\n⚠️ ML-Modelle nicht verfügbar - überspringe ML-Training")

    print()
    print("=" * 60)
    print("✅ Lernen mit Selbstlernen + ML abgeschlossen!")
    print("=" * 60)


def train_ml_models_full():
    """Vollständiges Training aller ML-Modelle (für initiales Setup oder Retraining)"""
    if not ML_AVAILABLE:
        print("❌ ML-Modelle nicht verfügbar")
        return

    print("\n" + "=" * 60)
    print("🧠 VOLLSTÄNDIGES ML-TRAINING")
    print("=" * 60)

    lotto_data = load_json('lotto_data.json', {'draws': []})
    draws = lotto_data.get('draws', [])

    if not draws:
        print("❌ Keine Trainingsdaten vorhanden")
        return

    print(f"📊 Trainiere mit {len(draws)} historischen Ziehungen...")

    results = train_all_models(draws)

    print("\n📈 Training-Ergebnisse:")
    for model, result in results.items():
        print(f"   • {model}: {result}")

    print("\n✅ Vollständiges ML-Training abgeschlossen!")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--full-train':
        # Vollständiges Training
        train_ml_models_full()
    else:
        # Normales inkrementelles Lernen
        learn_from_results()
