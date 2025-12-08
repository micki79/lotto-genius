#!/usr/bin/env python3
"""
🧠 LottoGenius - Kontinuierliches Lern-System

Lernt aus jeder Ziehung:
1. Vergleicht Vorhersagen mit echten Zahlen
2. Aktualisiert Provider-Scores (welche KI am besten ist)
3. Trackt Superzahl-Erfolge separat
4. Berechnet Methoden-Rankings
5. Speichert alles für langfristiges Lernen
"""
import json
import os
from datetime import datetime
from collections import Counter

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

def learn_from_results():
    """Hauptfunktion: Lernt aus vergangenen Vorhersagen"""
    
    print("=" * 60)
    print("🧠 LottoGenius - Lern-System")
    print("=" * 60)
    print(f"📅 {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
    print()
    
    # Lade alle Daten
    lotto_data = load_json('lotto_data.json', {'draws': []})
    predictions = load_json('predictions.json', {'predictions': [], 'history': []})
    learning = load_json('learning.json', {'entries': [], 'stats': {}, 'by_method': {}, 'by_provider': {}})
    provider_scores = load_json('provider_scores.json', {})
    superzahl_history = load_json('superzahl_history.json', {'entries': [], 'stats': {}})
    
    draws = lotto_data.get('draws', [])
    if not draws:
        print("⚠️ Keine Lotto-Daten vorhanden")
        return
    
    last_draw = draws[0]
    actual_numbers = set(last_draw['numbers'])
    actual_sz = last_draw['superzahl']
    draw_date = last_draw['date']
    
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
    
    for pred in unverified:
        pred_numbers = set(pred.get('numbers', []))
        pred_sz = pred.get('superzahl')
        method = pred.get('method', 'unknown')
        provider = pred.get('provider', method)
        
        # Berechne Treffer
        matches = len(pred_numbers & actual_numbers)
        sz_match = pred_sz == actual_sz
        
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
            'strategy': pred.get('strategy', '')
        }
        new_entries.append(entry)
        
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
                'four_plus': 0
            }
        method_updates[method]['predictions'] += 1
        method_updates[method]['total_matches'] += matches
        if sz_match:
            method_updates[method]['sz_correct'] += 1
        if matches >= 3:
            method_updates[method]['three_plus'] += 1
        if matches >= 4:
            method_updates[method]['four_plus'] += 1
        
        # Provider-Statistik
        if provider not in provider_updates:
            provider_updates[provider] = {
                'predictions': 0,
                'total_matches': 0,
                'sz_correct': 0
            }
        provider_updates[provider]['predictions'] += 1
        provider_updates[provider]['total_matches'] += matches
        if sz_match:
            provider_updates[provider]['sz_correct'] += 1
        
        # Markiere als verifiziert
        pred['verified'] = True
        pred['result'] = {
            'matches': matches,
            'sz_match': sz_match,
            'draw_date': draw_date
        }
        
        # Output
        match_indicator = "🎯" if matches >= 3 else "✓" if matches >= 2 else "•"
        sz_indicator = "✓ SZ" if sz_match else ""
        print(f"  {match_indicator} {method}: {matches}/6 Treffer {sz_indicator}")
    
    print("-" * 40)
    
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
                'four_plus': 0
            }
        
        m = learning['by_method'][method]
        m['total_predictions'] += update['predictions']
        m['total_matches'] += update['total_matches']
        m['total_possible'] += update['predictions'] * 6
        m['sz_correct'] += update['sz_correct']
        m['three_plus'] += update['three_plus']
        m['four_plus'] += update['four_plus']
        
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
                'sz_accuracy': 0
            }
        
        p = provider_scores[provider]
        p['total_predictions'] += update['predictions']
        p['total_matches'] += update['total_matches']
        p['sz_correct'] += update['sz_correct']
        
        if p['total_predictions'] > 0:
            p['accuracy'] = round((p['total_matches'] / (p['total_predictions'] * 6)) * 100, 2)
            p['sz_accuracy'] = round((p['sz_correct'] / p['total_predictions']) * 100, 2)
    
    # Superzahl-Historie
    superzahl_history['entries'].extend(sz_entries)
    superzahl_history['entries'] = superzahl_history['entries'][-500:]
    
    # Superzahl-Statistik
    sz_total = len(superzahl_history['entries'])
    sz_correct = sum(1 for e in superzahl_history['entries'] if e['correct'])
    superzahl_history['stats'] = {
        'total': sz_total,
        'correct': sz_correct,
        'accuracy': round((sz_correct / sz_total * 100), 2) if sz_total > 0 else 0,
        'last_update': datetime.now().isoformat()
    }
    
    # Gesamt-Statistik
    total_entries = len(learning['entries'])
    if total_entries > 0:
        total_matches = sum(e['matches'] for e in learning['entries'])
        total_sz_correct = sum(1 for e in learning['entries'] if e['sz_match'])
        three_plus = sum(1 for e in learning['entries'] if e['matches'] >= 3)
        
        learning['stats'] = {
            'total_entries': total_entries,
            'avg_matches': round(total_matches / total_entries, 2),
            'sz_accuracy': round((total_sz_correct / total_entries) * 100, 2),
            'three_plus_rate': round((three_plus / total_entries) * 100, 2),
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
    print()
    
    # Provider-Ranking
    print("🏆 Provider-Ranking:")
    ranked_providers = sorted(
        provider_scores.items(),
        key=lambda x: x[1].get('accuracy', 0),
        reverse=True
    )
    for i, (prov, stats) in enumerate(ranked_providers[:5], 1):
        print(f"   {i}. {prov}: {stats.get('accuracy', 0):.1f}% | SZ: {stats.get('sz_accuracy', 0):.0f}%")
    
    print()
    print("=" * 60)
    print("✅ Lernen abgeschlossen!")
    print("=" * 60)

if __name__ == "__main__":
    learn_from_results()
