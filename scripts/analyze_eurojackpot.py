#!/usr/bin/env python3
"""
📊 LottoGenius - Eurojackpot KI-Analyse-System
Analysiert Eurojackpot-Daten mit ALLEN wissenschaftlichen Methoden

Methoden:
1. Häufigkeitsanalyse (absolut + relativ)
2. Gap-Analyse (überfällige Zahlen)
3. Trend-Analyse (steigend/fallend)
4. Delta-System (Differenzen)
5. Positionsanalyse (Bereiche pro Position)
6. Summen-Analyse (optimaler Bereich)
7. Gerade/Ungerade-Verteilung
8. Hoch/Tief-Verteilung
9. Dekaden-Balance
10. Primzahlen-Analyse
11. Konsekutiv-Analyse (aufeinanderfolgende)
12. Zahlenpaare & Dreiergruppen
13. Wochentags-Muster (Di vs Fr)
14. Eurozahlen-Analyse (separat)
"""
import json
import os
from datetime import datetime
from collections import Counter
import math

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

# Primzahlen bis 50
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

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

# =====================================================
# ANALYSE-METHODEN
# =====================================================

def analyze_frequency(draws):
    """
    1. Häufigkeitsanalyse - Absolute und relative Häufigkeit
    """
    all_numbers = []
    all_eurozahlen = []

    for draw in draws:
        all_numbers.extend(draw['numbers'])
        all_eurozahlen.extend(draw['eurozahlen'])

    number_freq = Counter(all_numbers)
    euro_freq = Counter(all_eurozahlen)

    total_draws = len(draws)

    # Relative Häufigkeit
    number_relative = {n: round(c / total_draws * 100, 2) for n, c in number_freq.items()}
    euro_relative = {n: round(c / total_draws * 100, 2) for n, c in euro_freq.items()}

    # Erwartungswert (theoretisch sollte jede Zahl gleich oft kommen)
    expected_number = total_draws * 5 / 50  # 5 aus 50
    expected_euro = total_draws * 2 / 12    # 2 aus 12

    # Abweichung vom Erwartungswert
    deviation_numbers = {n: round(c - expected_number, 2) for n, c in number_freq.items()}
    deviation_euro = {n: round(c - expected_euro, 2) for n, c in euro_freq.items()}

    return {
        'numbers': dict(number_freq.most_common()),
        'eurozahlen': dict(euro_freq.most_common()),
        'numbers_relative': number_relative,
        'eurozahlen_relative': euro_relative,
        'hot_numbers': [n for n, _ in number_freq.most_common(15)],
        'cold_numbers': [n for n, _ in number_freq.most_common()[-15:]],
        'hot_eurozahlen': [n for n, _ in euro_freq.most_common(5)],
        'cold_eurozahlen': [n for n, _ in euro_freq.most_common()[-5:]],
        'expected_number': round(expected_number, 2),
        'expected_euro': round(expected_euro, 2),
        'deviation_numbers': deviation_numbers,
        'deviation_euro': deviation_euro
    }

def analyze_gaps(draws):
    """
    2. Gap-Analyse - Wie lange wurde jede Zahl nicht gezogen?
    """
    number_gaps = {}
    euro_gaps = {}

    # Hauptzahlen (1-50)
    for num in range(1, 51):
        for i, draw in enumerate(draws):
            if num in draw['numbers']:
                number_gaps[num] = i
                break
        else:
            number_gaps[num] = len(draws)

    # Eurozahlen (1-12)
    for num in range(1, 13):
        for i, draw in enumerate(draws):
            if num in draw['eurozahlen']:
                euro_gaps[num] = i
                break
        else:
            euro_gaps[num] = len(draws)

    overdue_numbers = sorted(number_gaps.items(), key=lambda x: x[1], reverse=True)[:15]
    overdue_euro = sorted(euro_gaps.items(), key=lambda x: x[1], reverse=True)[:5]

    # Durchschnittliche Lücke
    avg_gap_numbers = sum(number_gaps.values()) / len(number_gaps) if number_gaps else 0
    avg_gap_euro = sum(euro_gaps.values()) / len(euro_gaps) if euro_gaps else 0

    return {
        'number_gaps': number_gaps,
        'euro_gaps': euro_gaps,
        'overdue_numbers': [n for n, _ in overdue_numbers],
        'overdue_eurozahlen': [n for n, _ in overdue_euro],
        'avg_gap_numbers': round(avg_gap_numbers, 2),
        'avg_gap_euro': round(avg_gap_euro, 2),
        'max_gap_number': max(number_gaps.items(), key=lambda x: x[1]) if number_gaps else None,
        'max_gap_euro': max(euro_gaps.items(), key=lambda x: x[1]) if euro_gaps else None
    }

def analyze_trends(draws, window=20):
    """
    3. Trend-Analyse - Steigende vs fallende Zahlen
    """
    recent = draws[:window]
    older = draws[window:window*2]

    recent_freq = Counter()
    older_freq = Counter()
    recent_euro = Counter()
    older_euro = Counter()

    for draw in recent:
        recent_freq.update(draw['numbers'])
        recent_euro.update(draw['eurozahlen'])
    for draw in older:
        older_freq.update(draw['numbers'])
        older_euro.update(draw['eurozahlen'])

    # Trend für Hauptzahlen
    trends = {}
    for num in range(1, 51):
        r = recent_freq.get(num, 0)
        o = older_freq.get(num, 0)
        if o > 0:
            trend = round((r - o) / o * 100, 1)
        else:
            trend = r * 100 if r > 0 else 0
        trends[num] = trend

    # Trend für Eurozahlen
    euro_trends = {}
    for num in range(1, 13):
        r = recent_euro.get(num, 0)
        o = older_euro.get(num, 0)
        if o > 0:
            trend = round((r - o) / o * 100, 1)
        else:
            trend = r * 100 if r > 0 else 0
        euro_trends[num] = trend

    rising = sorted(trends.items(), key=lambda x: x[1], reverse=True)[:15]
    falling = sorted(trends.items(), key=lambda x: x[1])[:15]
    rising_euro = sorted(euro_trends.items(), key=lambda x: x[1], reverse=True)[:5]
    falling_euro = sorted(euro_trends.items(), key=lambda x: x[1])[:5]

    return {
        'number_trends': trends,
        'euro_trends': euro_trends,
        'rising_numbers': [n for n, _ in rising],
        'falling_numbers': [n for n, _ in falling],
        'rising_eurozahlen': [n for n, _ in rising_euro],
        'falling_eurozahlen': [n for n, _ in falling_euro],
        'window_size': window
    }

def analyze_delta(draws):
    """
    4. Delta-System - Analysiert Differenzen zwischen aufeinanderfolgenden Zahlen
    """
    all_deltas = []
    delta_frequency = Counter()

    for draw in draws:
        nums = sorted(draw['numbers'])
        for i in range(len(nums) - 1):
            delta = nums[i + 1] - nums[i]
            all_deltas.append(delta)
            delta_frequency[delta] += 1

    # Häufigste Deltas
    common_deltas = delta_frequency.most_common(10)

    # Durchschnittliches Delta
    avg_delta = sum(all_deltas) / len(all_deltas) if all_deltas else 0

    # Delta-Verteilung
    small_deltas = sum(1 for d in all_deltas if d <= 5)  # 1-5
    medium_deltas = sum(1 for d in all_deltas if 6 <= d <= 12)  # 6-12
    large_deltas = sum(1 for d in all_deltas if d > 12)  # 13+

    total = len(all_deltas) or 1

    return {
        'delta_frequency': dict(delta_frequency),
        'common_deltas': common_deltas,
        'avg_delta': round(avg_delta, 2),
        'small_delta_pct': round(small_deltas / total * 100, 1),
        'medium_delta_pct': round(medium_deltas / total * 100, 1),
        'large_delta_pct': round(large_deltas / total * 100, 1),
        'min_delta': min(all_deltas) if all_deltas else 0,
        'max_delta': max(all_deltas) if all_deltas else 0
    }

def analyze_positions(draws):
    """
    5. Positionsanalyse - Typische Bereiche für jede Position (1-5)
    """
    positions = {1: [], 2: [], 3: [], 4: [], 5: []}

    for draw in draws:
        nums = sorted(draw['numbers'])
        for i, num in enumerate(nums, 1):
            positions[i].append(num)

    position_stats = {}
    for pos, values in positions.items():
        if values:
            position_stats[pos] = {
                'min': min(values),
                'max': max(values),
                'avg': round(sum(values) / len(values), 1),
                'median': sorted(values)[len(values) // 2],
                'most_common': Counter(values).most_common(5),
                'typical_range': [
                    sorted(values)[int(len(values) * 0.1)],  # 10. Perzentil
                    sorted(values)[int(len(values) * 0.9)]   # 90. Perzentil
                ]
            }

    return {
        'position_stats': position_stats,
        'recommended_ranges': {
            1: [1, 15],    # Position 1 typisch 1-15
            2: [8, 25],    # Position 2 typisch 8-25
            3: [18, 35],   # Position 3 typisch 18-35
            4: [28, 43],   # Position 4 typisch 28-43
            5: [38, 50]    # Position 5 typisch 38-50
        }
    }

def analyze_sum_distribution(draws):
    """
    6. Summen-Analyse - Quersumme aller 5 Zahlen
    """
    sums = []
    euro_sums = []

    for draw in draws:
        sums.append(sum(draw['numbers']))
        euro_sums.append(sum(draw['eurozahlen']))

    if not sums:
        return {}

    # Statistiken für Hauptzahlen
    avg_sum = sum(sums) / len(sums)
    min_sum = min(sums)
    max_sum = max(sums)

    # Optimaler Bereich (Mittlere 60%)
    sorted_sums = sorted(sums)
    optimal_min = sorted_sums[int(len(sorted_sums) * 0.2)]
    optimal_max = sorted_sums[int(len(sorted_sums) * 0.8)]

    # Summen-Verteilung in Gruppen
    sum_groups = Counter()
    for s in sums:
        if s < 80:
            sum_groups['very_low'] += 1
        elif s < 110:
            sum_groups['low'] += 1
        elif s < 140:
            sum_groups['medium'] += 1
        elif s < 170:
            sum_groups['high'] += 1
        else:
            sum_groups['very_high'] += 1

    return {
        'avg_sum': round(avg_sum, 1),
        'min_sum': min_sum,
        'max_sum': max_sum,
        'optimal_range': [optimal_min, optimal_max],
        'recommended_range': [95, 160],  # Basierend auf Statistik
        'sum_distribution': dict(sum_groups),
        'euro_avg_sum': round(sum(euro_sums) / len(euro_sums), 1) if euro_sums else 0,
        'euro_optimal_range': [7, 18]  # 2 aus 12
    }

def analyze_odd_even(draws):
    """
    7. Gerade/Ungerade-Verteilung
    """
    distributions = Counter()
    euro_distributions = Counter()

    for draw in draws:
        odd = sum(1 for n in draw['numbers'] if n % 2 == 1)
        even = 5 - odd
        distributions[f"{odd}:{even}"] += 1

        euro_odd = sum(1 for n in draw['eurozahlen'] if n % 2 == 1)
        euro_even = 2 - euro_odd
        euro_distributions[f"{euro_odd}:{euro_even}"] += 1

    total = len(draws) or 1

    # Berechne Prozentsätze
    dist_pct = {k: round(v / total * 100, 1) for k, v in distributions.items()}
    euro_pct = {k: round(v / total * 100, 1) for k, v in euro_distributions.items()}

    # Beste Verteilungen
    best = distributions.most_common(3)

    return {
        'distributions': dict(distributions),
        'distributions_pct': dist_pct,
        'euro_distributions': dict(euro_distributions),
        'euro_distributions_pct': euro_pct,
        'most_common': [d for d, _ in best],
        'recommended': ['3:2', '2:3'],  # Statistisch am häufigsten
        'avoid': ['5:0', '0:5']  # Sehr selten
    }

def analyze_high_low(draws):
    """
    8. Hoch/Tief-Verteilung (1-25 = Tief, 26-50 = Hoch)
    """
    distributions = Counter()

    for draw in draws:
        low = sum(1 for n in draw['numbers'] if n <= 25)
        high = 5 - low
        distributions[f"{low}:{high}"] += 1

    total = len(draws) or 1
    dist_pct = {k: round(v / total * 100, 1) for k, v in distributions.items()}

    return {
        'distributions': dict(distributions),
        'distributions_pct': dist_pct,
        'most_common': [d for d, _ in distributions.most_common(3)],
        'recommended': ['3:2', '2:3'],
        'low_range': [1, 25],
        'high_range': [26, 50]
    }

def analyze_decades(draws):
    """
    9. Dekaden-Balance (1-10, 11-20, 21-30, 31-40, 41-50)
    """
    decade_counts = {f"{i*10+1}-{(i+1)*10}": [] for i in range(5)}
    decade_presence = Counter()

    for draw in draws:
        nums = draw['numbers']
        for i in range(5):
            count = sum(1 for n in nums if i*10 < n <= (i+1)*10)
            decade_counts[f"{i*10+1}-{(i+1)*10}"].append(count)
            if count > 0:
                decade_presence[f"{i*10+1}-{(i+1)*10}"] += 1

    # Durchschnitt pro Dekade
    decade_avg = {}
    for decade, counts in decade_counts.items():
        decade_avg[decade] = round(sum(counts) / len(counts), 2) if counts else 0

    # Wie oft sind alle 5 Dekaden vertreten?
    all_decades_count = 0
    for draw in draws:
        decades_hit = set()
        for n in draw['numbers']:
            decades_hit.add((n - 1) // 10)
        if len(decades_hit) == 5:
            all_decades_count += 1

    return {
        'decade_avg': decade_avg,
        'decade_presence': dict(decade_presence),
        'all_decades_pct': round(all_decades_count / len(draws) * 100, 1) if draws else 0,
        'recommended': 'Mindestens 3-4 verschiedene Dekaden abdecken'
    }

def analyze_primes(draws):
    """
    10. Primzahlen-Analyse
    """
    prime_counts = []

    for draw in draws:
        count = sum(1 for n in draw['numbers'] if n in PRIMES)
        prime_counts.append(count)

    if not prime_counts:
        return {}

    distribution = Counter(prime_counts)

    return {
        'prime_numbers': PRIMES,
        'avg_primes': round(sum(prime_counts) / len(prime_counts), 2),
        'distribution': dict(distribution),
        'most_common_count': distribution.most_common(1)[0][0] if distribution else 0,
        'recommended': '1-2 Primzahlen pro Tipp'
    }

def analyze_consecutive(draws):
    """
    11. Konsekutiv-Analyse - Aufeinanderfolgende Zahlen
    """
    consecutive_counts = Counter()

    for draw in draws:
        nums = sorted(draw['numbers'])
        max_consecutive = 1
        current = 1

        for i in range(len(nums) - 1):
            if nums[i + 1] - nums[i] == 1:
                current += 1
                max_consecutive = max(max_consecutive, current)
            else:
                current = 1

        consecutive_counts[max_consecutive] += 1

    total = len(draws) or 1

    # Prozentuale Verteilung
    pct = {k: round(v / total * 100, 1) for k, v in consecutive_counts.items()}

    # Häufigkeit von Paaren (2er-Folgen)
    pairs_count = sum(1 for d in draws
                      for i in range(len(sorted(d['numbers']))-1)
                      if sorted(d['numbers'])[i+1] - sorted(d['numbers'])[i] == 1)

    return {
        'distribution': dict(consecutive_counts),
        'distribution_pct': pct,
        'most_common': consecutive_counts.most_common(1)[0] if consecutive_counts else None,
        'no_consecutive_pct': round(consecutive_counts.get(1, 0) / total * 100, 1),
        'pairs_per_draw': round(pairs_count / len(draws), 2) if draws else 0,
        'recommendation': 'Maximal 1 Paar aufeinanderfolgender Zahlen'
    }

def analyze_pairs(draws, limit=200):
    """
    12. Zahlenpaare & Dreiergruppen
    """
    pairs = Counter()
    triplets = Counter()
    euro_pairs = Counter()

    for draw in draws[:limit]:
        nums = draw['numbers']
        euros = draw['eurozahlen']

        # Paare
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                pairs[tuple(sorted([nums[i], nums[j]]))] += 1

                # Dreiergruppen
                for k in range(j + 1, len(nums)):
                    triplets[tuple(sorted([nums[i], nums[j], nums[k]]))] += 1

        # Eurozahlen-Paare
        if len(euros) == 2:
            euro_pairs[tuple(sorted(euros))] += 1

    return {
        'top_pairs': [list(p) for p, _ in pairs.most_common(30)],
        'top_triplets': [list(t) for t, _ in triplets.most_common(20)],
        'top_euro_pairs': [list(p) for p, _ in euro_pairs.most_common(15)],
        'pair_frequency': {str(list(k)): v for k, v in pairs.most_common(50)},
        'total_unique_pairs': len(pairs),
        'total_unique_triplets': len(triplets)
    }

def analyze_weekday_patterns(draws):
    """
    13. Wochentags-Muster - Unterschiede Dienstag vs Freitag
    """
    tuesday_freq = Counter()
    friday_freq = Counter()
    tuesday_euro = Counter()
    friday_euro = Counter()
    tuesday_count = 0
    friday_count = 0

    for draw in draws[:200]:
        try:
            day, month, year = map(int, draw['date'].split('.'))
            date = datetime(year, month, day)

            if date.weekday() == 1:  # Dienstag
                tuesday_freq.update(draw['numbers'])
                tuesday_euro.update(draw['eurozahlen'])
                tuesday_count += 1
            elif date.weekday() == 4:  # Freitag
                friday_freq.update(draw['numbers'])
                friday_euro.update(draw['eurozahlen'])
                friday_count += 1
        except:
            pass

    return {
        'tuesday_hot': [n for n, _ in tuesday_freq.most_common(10)],
        'friday_hot': [n for n, _ in friday_freq.most_common(10)],
        'tuesday_euro_hot': [n for n, _ in tuesday_euro.most_common(5)],
        'friday_euro_hot': [n for n, _ in friday_euro.most_common(5)],
        'tuesday_draws': tuesday_count,
        'friday_draws': friday_count,
        'tuesday_frequency': dict(tuesday_freq),
        'friday_frequency': dict(friday_freq)
    }

def analyze_eurozahlen_patterns(draws):
    """
    14. Spezielle Eurozahlen-Analyse
    """
    # Folge-Muster (welche Eurozahl folgt welcher)
    follows = Counter()
    for i in range(len(draws) - 1):
        current = tuple(draws[i]['eurozahlen'])
        previous = tuple(draws[i + 1]['eurozahlen'])
        follows[(previous, current)] += 1

    # Häufigkeit pro Kombination
    combo_freq = Counter()
    for draw in draws:
        combo = tuple(sorted(draw['eurozahlen']))
        combo_freq[combo] += 1

    # Gerade/Ungerade für Eurozahlen
    odd_even = Counter()
    for draw in draws:
        odd = sum(1 for n in draw['eurozahlen'] if n % 2 == 1)
        odd_even[f"{odd}:{2-odd}"] += 1

    return {
        'follow_patterns': {f"{k[0]}->{k[1]}": v for k, v in follows.most_common(20)},
        'top_combinations': [list(c) for c, _ in combo_freq.most_common(20)],
        'combination_frequency': {str(list(k)): v for k, v in combo_freq.most_common(30)},
        'odd_even_distribution': dict(odd_even),
        'total_combinations': len(combo_freq)
    }

# =====================================================
# HAUPTFUNKTION
# =====================================================

def run_analysis():
    """Führt die komplette Eurojackpot-Analyse durch"""

    print("=" * 60)
    print("📊 LottoGenius - EUROJACKPOT KI-Analyse")
    print("=" * 60)
    print(f"📅 {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
    print()

    # Lade Daten
    data = load_json('eurojackpot_data.json')
    draws = data.get('draws', [])

    if not draws:
        print("⚠️ Keine Eurojackpot-Daten vorhanden")
        return

    print(f"📊 Analysiere {len(draws)} Eurojackpot-Ziehungen...")
    print()

    # Führe alle Analysen durch
    print("  1️⃣ Häufigkeitsanalyse...")
    frequency = analyze_frequency(draws)

    print("  2️⃣ Gap-Analyse...")
    gaps = analyze_gaps(draws)

    print("  3️⃣ Trend-Analyse...")
    trends = analyze_trends(draws)

    print("  4️⃣ Delta-System...")
    delta = analyze_delta(draws)

    print("  5️⃣ Positionsanalyse...")
    positions = analyze_positions(draws)

    print("  6️⃣ Summen-Analyse...")
    sums = analyze_sum_distribution(draws)

    print("  7️⃣ Gerade/Ungerade...")
    odd_even = analyze_odd_even(draws)

    print("  8️⃣ Hoch/Tief...")
    high_low = analyze_high_low(draws)

    print("  9️⃣ Dekaden-Balance...")
    decades = analyze_decades(draws)

    print("  🔟 Primzahlen-Analyse...")
    primes = analyze_primes(draws)

    print("  1️⃣1️⃣ Konsekutiv-Analyse...")
    consecutive = analyze_consecutive(draws)

    print("  1️⃣2️⃣ Zahlenpaare & Dreiergruppen...")
    pairs = analyze_pairs(draws)

    print("  1️⃣3️⃣ Wochentags-Muster...")
    weekday = analyze_weekday_patterns(draws)

    print("  1️⃣4️⃣ Eurozahlen-Muster...")
    eurozahlen = analyze_eurozahlen_patterns(draws)

    # Erstelle Gesamt-Analyse
    analysis = {
        'timestamp': datetime.now().isoformat(),
        'total_draws': len(draws),
        'last_draw': draws[0] if draws else None,
        'frequency': frequency,
        'gaps': gaps,
        'trends': trends,
        'delta': delta,
        'positions': positions,
        'sum_distribution': sums,
        'odd_even': odd_even,
        'high_low': high_low,
        'decades': decades,
        'primes': primes,
        'consecutive': consecutive,
        'pairs': pairs,
        'weekday_patterns': weekday,
        'eurozahlen_patterns': eurozahlen,
        # KI-Empfehlungen basierend auf Analyse
        'ki_recommendations': {
            'hot_numbers': frequency['hot_numbers'][:10],
            'cold_numbers': frequency['cold_numbers'][:10],
            'overdue_numbers': gaps['overdue_numbers'][:10],
            'rising_numbers': trends['rising_numbers'][:10],
            'optimal_sum_range': sums.get('optimal_range', [95, 160]),
            'best_odd_even': odd_even.get('most_common', ['3:2', '2:3']),
            'best_high_low': high_low.get('most_common', ['3:2', '2:3']),
            'common_deltas': [d for d, _ in delta.get('common_deltas', [])[:5]],
            'hot_eurozahlen': frequency['hot_eurozahlen'],
            'overdue_eurozahlen': gaps['overdue_eurozahlen'],
            'top_euro_pairs': pairs['top_euro_pairs'][:5]
        }
    }

    # Speichern
    save_json('eurojackpot_analysis.json', analysis)

    print()
    print("=" * 60)
    print("✅ Eurojackpot-Analyse abgeschlossen!")
    print("=" * 60)
    print()
    print(f"🔥 Heiße Zahlen: {frequency['hot_numbers'][:5]}")
    print(f"❄️ Kalte Zahlen: {frequency['cold_numbers'][:5]}")
    print(f"⏰ Überfällig: {gaps['overdue_numbers'][:5]}")
    print(f"📈 Steigend: {trends['rising_numbers'][:5]}")
    print(f"🌟 Heiße Eurozahlen: {frequency['hot_eurozahlen']}")
    print(f"📊 Optimale Summe: {sums.get('optimal_range', [95, 160])}")
    print(f"⚖️ Beste Gerade/Ungerade: {odd_even.get('most_common', [])}")
    print()

    return analysis

if __name__ == "__main__":
    run_analysis()
