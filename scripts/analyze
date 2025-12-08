#!/usr/bin/env python3
"""
📊 LottoGenius - KI-Analyse-System
Analysiert Lotto-Daten mit verschiedenen statistischen Methoden
"""
import json
import os
from datetime import datetime
from collections import Counter

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

def load_json(filename):
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {}

def save_json(filename, data):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, filename), 'w') as f:
        json.dump(data, f, indent=2)

def analyze_frequency(draws):
    """Häufigkeitsanalyse aller Zahlen"""
    all_numbers = []
    all_superzahlen = []
    
    for draw in draws:
        all_numbers.extend(draw['numbers'])
        all_superzahlen.append(draw['superzahl'])
    
    number_freq = Counter(all_numbers)
    sz_freq = Counter(all_superzahlen)
    
    return {
        'numbers': dict(number_freq.most_common()),
        'superzahlen': dict(sz_freq.most_common()),
        'hot_numbers': [n for n, _ in number_freq.most_common(15)],
        'cold_numbers': [n for n, _ in number_freq.most_common()[-15:]],
        'hot_superzahlen': [n for n, _ in sz_freq.most_common(5)]
    }

def analyze_gaps(draws):
    """Lücken-Analyse: Wie lange wurde jede Zahl nicht gezogen?"""
    number_gaps = {}
    sz_gaps = {}
    
    for num in range(1, 50):
        for i, draw in enumerate(draws):
            if num in draw['numbers']:
                number_gaps[num] = i
                break
        else:
            number_gaps[num] = len(draws)
    
    for sz in range(10):
        for i, draw in enumerate(draws):
            if draw['superzahl'] == sz:
                sz_gaps[sz] = i
                break
        else:
            sz_gaps[sz] = len(draws)
    
    overdue = sorted(number_gaps.items(), key=lambda x: x[1], reverse=True)[:15]
    overdue_sz = sorted(sz_gaps.items(), key=lambda x: x[1], reverse=True)[:5]
    
    return {
        'number_gaps': number_gaps,
        'sz_gaps': sz_gaps,
        'overdue_numbers': [n for n, _ in overdue],
        'overdue_superzahlen': [n for n, _ in overdue_sz]
    }

def analyze_trends(draws, window=20):
    """Trend-Analyse: Steigende vs fallende Zahlen"""
    recent = draws[:window]
    older = draws[window:window*2]
    
    recent_freq = Counter()
    older_freq = Counter()
    
    for draw in recent:
        recent_freq.update(draw['numbers'])
    for draw in older:
        older_freq.update(draw['numbers'])
    
    trends = {}
    for num in range(1, 50):
        r = recent_freq.get(num, 0)
        o = older_freq.get(num, 0)
        if o > 0:
            trend = round((r - o) / o * 100, 1)
        else:
            trend = r * 100 if r > 0 else 0
        trends[num] = trend
    
    rising = sorted(trends.items(), key=lambda x: x[1], reverse=True)[:15]
    falling = sorted(trends.items(), key=lambda x: x[1])[:15]
    
    return {
        'trends': trends,
        'rising_numbers': [n for n, _ in rising],
        'falling_numbers': [n for n, _ in falling]
    }

def analyze_pairs(draws, limit=100):
    """Paar-Analyse: Welche Zahlen kommen oft zusammen?"""
    pairs = Counter()
    triplets = Counter()
    
    for draw in draws[:limit]:
        nums = draw['numbers']
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                pairs[tuple(sorted([nums[i], nums[j]]))] += 1
                for k in range(j + 1, len(nums)):
                    triplets[tuple(sorted([nums[i], nums[j], nums[k]]))] += 1
    
    return {
        'top_pairs': [list(p) for p, _ in pairs.most_common(30)],
        'top_triplets': [list(t) for t, _ in triplets.most_common(20)]
    }

def analyze_superzahl_patterns(draws):
    """Superzahl-Muster-Analyse"""
    # Folge-Muster
    follows = Counter()
    for i in range(len(draws) - 1):
        current = draws[i]['superzahl']
        previous = draws[i + 1]['superzahl']
        follows[(previous, current)] += 1
    
    # Wochentag-Muster
    wed_freq = Counter()
    sat_freq = Counter()
    
    for draw in draws[:200]:
        try:
            day, month, year = map(int, draw['date'].split('.'))
            date = datetime(year, month, day)
            if date.weekday() == 2:
                wed_freq[draw['superzahl']] += 1
            elif date.weekday() == 5:
                sat_freq[draw['superzahl']] += 1
        except:
            pass
    
    return {
        'follow_patterns': {f"{k[0]}->{k[1]}": v for k, v in follows.most_common(30)},
        'wednesday_freq': dict(wed_freq),
        'saturday_freq': dict(sat_freq)
    }

def run_analysis():
    print("=" * 60)
    print("📊 LottoGenius - KI-Analyse")
    print("=" * 60)
    print(f"📅 {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
    print()
    
    data = load_json('lotto_data.json')
    draws = data.get('draws', [])
    
    if not draws:
        print("⚠️ Keine Lotto-Daten vorhanden")
        return
    
    print(f"📊 Analysiere {len(draws)} Ziehungen...")
    print()
    
    analysis = {
        'timestamp': datetime.now().isoformat(),
        'total_draws': len(draws),
        'last_draw': draws[0] if draws else None,
        'frequency': analyze_frequency(draws),
        'gaps': analyze_gaps(draws),
        'trends': analyze_trends(draws),
        'pairs': analyze_pairs(draws),
        'superzahl_patterns': analyze_superzahl_patterns(draws)
    }
    
    save_json('analysis.json', analysis)
    
    print("✅ Analyse abgeschlossen!")
    print()
    print(f"🔥 Heiße Zahlen: {analysis['frequency']['hot_numbers'][:5]}")
    print(f"❄️ Kalte Zahlen: {analysis['frequency']['cold_numbers'][:5]}")
    print(f"⏰ Überfällig: {analysis['gaps']['overdue_numbers'][:5]}")
    print(f"📈 Steigend: {analysis['trends']['rising_numbers'][:5]}")
    print()
    print("=" * 60)
    
    return analysis

if __name__ == "__main__":
    run_analysis()
