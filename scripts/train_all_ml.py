#!/usr/bin/env python3
"""
🧠 LottoGenius - ALLE ML-MODELLE TRAINIEREN

Trainiert die echten ML-Modelle für alle 5 Spiele:
- Lotto 6aus49 (Neural Network, Markov, Bayesian, RL, Ensemble)
- Eurojackpot (Neural Network, Markov, Bayesian)
- Spiel 77 (Neural Network, Markov, Bayesian)
- Super 6 (Neural Network, Markov, Bayesian)
- Glücksspirale (Neural Network, Markov, Bayesian)

Verwendung:
    python train_all_ml.py           # Trainiert alle Modelle
    python train_all_ml.py --quick   # Schnelles Training (weniger Epochen)
"""
import json
import os
import sys
from datetime import datetime

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
        json.dump(data, f, indent=2, ensure_ascii=False)

def train_all():
    """Trainiert alle ML-Modelle für alle Spiele"""

    print("=" * 70)
    print("🧠 LOTTOGENIUS - VOLLSTÄNDIGES ML-TRAINING FÜR ALLE SPIELE")
    print("=" * 70)
    print(f"📅 {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
    print()

    quick_mode = '--quick' in sys.argv
    if quick_mode:
        print("⚡ SCHNELLMODUS: Weniger Epochen")
        print()

    results = {}

    # =====================================================
    # 1. LOTTO 6aus49
    # =====================================================
    print("=" * 70)
    print("🎱 1/5: LOTTO 6aus49 ML-TRAINING")
    print("=" * 70)

    try:
        from ml_models import train_all_models

        lotto_data = load_json('lotto_data.json', {'draws': []})
        draws = lotto_data.get('draws', [])

        if draws:
            print(f"📊 {len(draws)} Ziehungen gefunden")
            result = train_all_models(draws)
            results['lotto_6aus49'] = {'status': 'success', 'draws': len(draws)}
            print("✅ Lotto 6aus49 ML-Training abgeschlossen!")
        else:
            print("⚠️ Keine Lotto 6aus49 Daten gefunden")
            results['lotto_6aus49'] = {'status': 'no_data'}
    except Exception as e:
        print(f"❌ Fehler: {e}")
        results['lotto_6aus49'] = {'status': 'error', 'error': str(e)}

    print()

    # =====================================================
    # 2. EUROJACKPOT
    # =====================================================
    print("=" * 70)
    print("🌟 2/5: EUROJACKPOT ML-TRAINING")
    print("=" * 70)

    try:
        from ml_models import train_eurojackpot_ml

        ej_data = load_json('eurojackpot_data.json', {'draws': []})
        draws = ej_data.get('draws', [])

        if draws:
            print(f"📊 {len(draws)} Ziehungen gefunden")
            train_eurojackpot_ml(draws)
            results['eurojackpot'] = {'status': 'success', 'draws': len(draws)}
            print("✅ Eurojackpot ML-Training abgeschlossen!")
        else:
            print("⚠️ Keine Eurojackpot Daten gefunden")
            results['eurojackpot'] = {'status': 'no_data'}
    except Exception as e:
        print(f"❌ Fehler: {e}")
        results['eurojackpot'] = {'status': 'error', 'error': str(e)}

    print()

    # =====================================================
    # 3. SPIEL 77
    # =====================================================
    print("=" * 70)
    print("🎰 3/5: SPIEL 77 ML-TRAINING")
    print("=" * 70)

    try:
        from ml_models import train_digit_game_ml

        sp77_data = load_json('spiel77_data.json', {'draws': []})
        draws = sp77_data.get('draws', [])

        if draws:
            print(f"📊 {len(draws)} Ziehungen gefunden")
            train_digit_game_ml('spiel77', 7, draws)
            results['spiel77'] = {'status': 'success', 'draws': len(draws)}
            print("✅ Spiel 77 ML-Training abgeschlossen!")
        else:
            print("⚠️ Keine Spiel 77 Daten gefunden")
            results['spiel77'] = {'status': 'no_data'}
    except Exception as e:
        print(f"❌ Fehler: {e}")
        results['spiel77'] = {'status': 'error', 'error': str(e)}

    print()

    # =====================================================
    # 4. SUPER 6
    # =====================================================
    print("=" * 70)
    print("🎲 4/5: SUPER 6 ML-TRAINING")
    print("=" * 70)

    try:
        from ml_models import train_digit_game_ml

        s6_data = load_json('super6_data.json', {'draws': []})
        draws = s6_data.get('draws', [])

        if draws:
            print(f"📊 {len(draws)} Ziehungen gefunden")
            train_digit_game_ml('super6', 6, draws)
            results['super6'] = {'status': 'success', 'draws': len(draws)}
            print("✅ Super 6 ML-Training abgeschlossen!")
        else:
            print("⚠️ Keine Super 6 Daten gefunden")
            results['super6'] = {'status': 'no_data'}
    except Exception as e:
        print(f"❌ Fehler: {e}")
        results['super6'] = {'status': 'error', 'error': str(e)}

    print()

    # =====================================================
    # 5. GLÜCKSSPIRALE
    # =====================================================
    print("=" * 70)
    print("🌀 5/5: GLÜCKSSPIRALE ML-TRAINING")
    print("=" * 70)

    try:
        from ml_models import train_digit_game_ml

        gs_data = load_json('gluecksspirale_data.json', {'draws': []})
        draws = gs_data.get('draws', [])

        if draws:
            print(f"📊 {len(draws)} Ziehungen gefunden")
            train_digit_game_ml('gluecksspirale', 7, draws)
            results['gluecksspirale'] = {'status': 'success', 'draws': len(draws)}
            print("✅ Glücksspirale ML-Training abgeschlossen!")
        else:
            print("⚠️ Keine Glücksspirale Daten gefunden")
            results['gluecksspirale'] = {'status': 'no_data'}
    except Exception as e:
        print(f"❌ Fehler: {e}")
        results['gluecksspirale'] = {'status': 'error', 'error': str(e)}

    print()

    # =====================================================
    # ZUSAMMENFASSUNG
    # =====================================================
    print("=" * 70)
    print("📊 TRAINING-ZUSAMMENFASSUNG")
    print("=" * 70)

    success_count = sum(1 for r in results.values() if r.get('status') == 'success')
    total_draws = sum(r.get('draws', 0) for r in results.values())

    for game, result in results.items():
        status = result.get('status')
        if status == 'success':
            print(f"   ✅ {game}: {result.get('draws', 0)} Ziehungen trainiert")
        elif status == 'no_data':
            print(f"   ⚠️ {game}: Keine Daten")
        else:
            print(f"   ❌ {game}: Fehler - {result.get('error', 'Unbekannt')}")

    print()
    print(f"🎯 {success_count}/5 Spiele erfolgreich trainiert")
    print(f"📈 Gesamt: {total_draws} Ziehungen verarbeitet")

    # Speichere Training-Status
    training_status = {
        'last_full_training': datetime.now().isoformat(),
        'results': results,
        'success_count': success_count,
        'total_draws': total_draws
    }
    save_json('ml_training_status.json', training_status)

    print()
    print("=" * 70)
    print("✅ VOLLSTÄNDIGES ML-TRAINING ABGESCHLOSSEN!")
    print("=" * 70)
    print()
    print("💡 Die ML-Modelle werden jetzt bei jeder Vorhersage verwendet.")
    print("   Führe predict_*.py aus, um Vorhersagen mit ML zu erhalten.")
    print()


def learn_all():
    """Lernt aus den letzten Ziehungen für alle Spiele"""

    print("=" * 70)
    print("🧠 LOTTOGENIUS - LERNEN AUS ALLEN SPIELEN")
    print("=" * 70)
    print(f"📅 {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
    print()

    # Import learn scripts
    import subprocess

    scripts = [
        ('learn.py', 'Lotto 6aus49'),
        ('learn_eurojackpot.py', 'Eurojackpot'),
        ('learn_spiel77.py', 'Spiel 77'),
        ('learn_super6.py', 'Super 6'),
        ('learn_gluecksspirale.py', 'Glücksspirale')
    ]

    for script, name in scripts:
        print(f"\n{'='*60}")
        print(f"🎯 {name}")
        print('='*60)

        script_path = os.path.join(os.path.dirname(__file__), script)
        if os.path.exists(script_path):
            try:
                result = subprocess.run(
                    ['python3', script_path],
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                print(result.stdout)
                if result.stderr:
                    print(f"⚠️ {result.stderr}")
            except subprocess.TimeoutExpired:
                print(f"⏰ Timeout bei {name}")
            except Exception as e:
                print(f"❌ Fehler: {e}")
        else:
            print(f"⚠️ Skript nicht gefunden: {script}")

    print()
    print("=" * 70)
    print("✅ LERNEN FÜR ALLE SPIELE ABGESCHLOSSEN!")
    print("=" * 70)


if __name__ == "__main__":
    if '--learn' in sys.argv:
        learn_all()
    else:
        train_all()
