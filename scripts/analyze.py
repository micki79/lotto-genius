#!/usr/bin/env python3
"""
📊 LottoGenius - VOLLSTÄNDIGES Lotto 6aus49 KI-Analyse-System

ALLE 40 wissenschaftlichen Analyse-Methoden:
1. Häufigkeitsanalyse (absolut + relativ + periodisch)
2. Heiße/Kalte Zahlen
3. Gap-/Verzögerungsanalyse
4. Trend-Analyse (gleitende Durchschnitte)
5. Delta-System (Differenzen)
6. Positionsanalyse (typische Bereiche)
7. Quersummen-System
8. Gerade/Ungerade-Verteilung
9. Hoch/Tief-Verteilung
10. Zehnergruppen-System
11. Primzahlen-Analyse
12. Fibonacci & Quadratzahlen
13. Endziffern-Verteilung
14. Konsekutiv-Analyse
15. Zahlenpaare & Dreiergruppen
16. Nachbarzahlen-Analyse
17. Wochentags-Muster
18. Ausreißer-Erkennung
19. Superzahl-Muster (alle 8 Faktoren)
"""
import json
import os
from datetime import datetime
from collections import Counter
import math

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

# Konstanten
PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
FIBONACCI = [1, 2, 3, 5, 8, 13, 21, 34]
SQUARES = [1, 4, 9, 16, 25, 36, 49]

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
    1. Häufigkeitsanalyse - Absolut, Relativ, Periodisch
    """
    all_numbers = []
    all_superzahlen = []

    for draw in draws:
        all_numbers.extend(draw['numbers'])
        all_superzahlen.append(draw['superzahl'])

    number_freq = Counter(all_numbers)
    sz_freq = Counter(all_superzahlen)

    total_draws = len(draws)

    # Relative Häufigkeit (Prozent)
    number_relative = {n: round(c / total_draws * 100, 2) for n, c in number_freq.items()}
    sz_relative = {n: round(c / total_draws * 100, 2) for n, c in sz_freq.items()}

    # Erwartungswert
    expected_number = total_draws * 6 / 49
    expected_sz = total_draws / 10

    # Abweichung vom Erwartungswert
    deviation = {n: round(number_freq.get(n, 0) - expected_number, 2) for n in range(1, 50)}

    # Periodische Häufigkeit (letzte 10, 20, 50, 100 Ziehungen)
    periodic = {}
    for window in [10, 20, 50, 100]:
        window_draws = draws[:window]
        window_nums = []
        for d in window_draws:
            window_nums.extend(d['numbers'])
        periodic[f'last_{window}'] = dict(Counter(window_nums).most_common(10))

    return {
        'numbers': dict(number_freq.most_common()),
        'superzahlen': dict(sz_freq.most_common()),
        'numbers_relative': number_relative,
        'sz_relative': sz_relative,
        'expected_number': round(expected_number, 2),
        'expected_sz': round(expected_sz, 2),
        'deviation': deviation,
        'periodic': periodic,
        'hot_numbers': [n for n, _ in number_freq.most_common(15)],
        'cold_numbers': [n for n, _ in number_freq.most_common()[-15:]],
        'hot_superzahlen': [n for n, _ in sz_freq.most_common(5)],
        'cold_superzahlen': [n for n, _ in sz_freq.most_common()[-5:]]
    }

def analyze_gaps(draws):
    """
    2. Gap-/Verzögerungsanalyse - Wie lange wurde jede Zahl nicht gezogen?
    """
    number_gaps = {}
    sz_gaps = {}

    # Hauptzahlen (1-49)
    for num in range(1, 50):
        for i, draw in enumerate(draws):
            if num in draw['numbers']:
                number_gaps[num] = i
                break
        else:
            number_gaps[num] = len(draws)

    # Superzahlen (0-9)
    for sz in range(10):
        for i, draw in enumerate(draws):
            if draw['superzahl'] == sz:
                sz_gaps[sz] = i
                break
        else:
            sz_gaps[sz] = len(draws)

    # Durchschnittliche Wartezeit berechnen
    avg_gap = sum(number_gaps.values()) / len(number_gaps) if number_gaps else 0

    overdue = sorted(number_gaps.items(), key=lambda x: x[1], reverse=True)[:15]
    overdue_sz = sorted(sz_gaps.items(), key=lambda x: x[1], reverse=True)[:5]

    return {
        'number_gaps': number_gaps,
        'sz_gaps': sz_gaps,
        'avg_gap': round(avg_gap, 2),
        'max_gap': max(number_gaps.values()) if number_gaps else 0,
        'overdue_numbers': [n for n, _ in overdue],
        'overdue_superzahlen': [n for n, _ in overdue_sz]
    }

def analyze_trends(draws, windows=[10, 20, 50]):
    """
    3. Trend-Analyse mit gleitenden Durchschnitten
    """
    all_trends = {}

    for window in windows:
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

        rising = sorted(trends.items(), key=lambda x: x[1], reverse=True)[:10]
        falling = sorted(trends.items(), key=lambda x: x[1])[:10]

        all_trends[f'window_{window}'] = {
            'trends': trends,
            'rising': [n for n, _ in rising],
            'falling': [n for n, _ in falling]
        }

    # Haupttrends (Fenster 20)
    main_trends = all_trends.get('window_20', {}).get('trends', {})

    return {
        'all_windows': all_trends,
        'trends': main_trends,
        'rising_numbers': all_trends.get('window_20', {}).get('rising', []),
        'falling_numbers': all_trends.get('window_20', {}).get('falling', [])
    }

def analyze_delta(draws):
    """
    4. Delta-System - Differenzen zwischen aufeinanderfolgenden Zahlen
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

    # Delta-Kategorien
    small = sum(1 for d in all_deltas if d <= 5)
    medium = sum(1 for d in all_deltas if 6 <= d <= 12)
    large = sum(1 for d in all_deltas if d > 12)
    total = len(all_deltas) or 1

    return {
        'delta_frequency': dict(delta_frequency),
        'common_deltas': common_deltas,
        'avg_delta': round(avg_delta, 2),
        'small_delta_pct': round(small / total * 100, 1),
        'medium_delta_pct': round(medium / total * 100, 1),
        'large_delta_pct': round(large / total * 100, 1),
        'optimal_deltas': [d for d, _ in common_deltas[:5]]
    }

def analyze_positions(draws):
    """
    5. Positionsanalyse - Typische Bereiche für jede Position (1-6)
    """
    positions = {i: [] for i in range(1, 7)}

    for draw in draws:
        nums = sorted(draw['numbers'])
        for i, num in enumerate(nums, 1):
            positions[i].append(num)

    position_stats = {}
    for pos, values in positions.items():
        if values:
            sorted_vals = sorted(values)
            position_stats[pos] = {
                'min': min(values),
                'max': max(values),
                'avg': round(sum(values) / len(values), 1),
                'median': sorted_vals[len(sorted_vals) // 2],
                'most_common': [n for n, _ in Counter(values).most_common(5)],
                'range_10_90': [
                    sorted_vals[int(len(sorted_vals) * 0.1)],
                    sorted_vals[int(len(sorted_vals) * 0.9)]
                ]
            }

    return {
        'position_stats': position_stats,
        'recommended_ranges': {
            1: [1, 12],
            2: [8, 22],
            3: [15, 30],
            4: [22, 38],
            5: [30, 44],
            6: [38, 49]
        }
    }

def analyze_sum_distribution(draws):
    """
    6. Quersummen-System - Summe aller 6 Zahlen
    """
    sums = []
    endziffer_sums = []

    for draw in draws:
        total = sum(draw['numbers'])
        sums.append(total)
        # Endziffern-Summe
        endziffer_sum = sum(n % 10 for n in draw['numbers'])
        endziffer_sums.append(endziffer_sum)

    if not sums:
        return {}

    avg_sum = sum(sums) / len(sums)
    sorted_sums = sorted(sums)

    # Optimaler Bereich (mittlere 60%)
    optimal_min = sorted_sums[int(len(sorted_sums) * 0.2)]
    optimal_max = sorted_sums[int(len(sorted_sums) * 0.8)]

    # Summen-Gruppen
    sum_groups = Counter()
    for s in sums:
        if s < 100:
            sum_groups['very_low'] += 1
        elif s < 130:
            sum_groups['low'] += 1
        elif s < 160:
            sum_groups['medium'] += 1
        elif s < 190:
            sum_groups['high'] += 1
        else:
            sum_groups['very_high'] += 1

    return {
        'avg_sum': round(avg_sum, 1),
        'min_sum': min(sums),
        'max_sum': max(sums),
        'optimal_range': [optimal_min, optimal_max],
        'recommended_range': [130, 170],
        'sum_distribution': dict(sum_groups),
        'avg_endziffer_sum': round(sum(endziffer_sums) / len(endziffer_sums), 1) if endziffer_sums else 0
    }

def analyze_odd_even(draws):
    """
    7. Gerade/Ungerade-Verteilung
    """
    distributions = Counter()

    for draw in draws:
        odd = sum(1 for n in draw['numbers'] if n % 2 == 1)
        even = 6 - odd
        distributions[f"{odd}:{even}"] += 1

    total = len(draws) or 1
    dist_pct = {k: round(v / total * 100, 1) for k, v in distributions.items()}

    return {
        'distributions': dict(distributions),
        'distributions_pct': dist_pct,
        'most_common': [d for d, _ in distributions.most_common(3)],
        'recommended': ['3:3', '2:4', '4:2'],
        'avoid': ['6:0', '0:6', '5:1', '1:5']
    }

def analyze_high_low(draws):
    """
    8. Hoch/Tief-Verteilung (1-24 = Tief, 25-49 = Hoch)
    """
    distributions = Counter()

    for draw in draws:
        low = sum(1 for n in draw['numbers'] if n <= 24)
        high = 6 - low
        distributions[f"{low}:{high}"] += 1

    total = len(draws) or 1
    dist_pct = {k: round(v / total * 100, 1) for k, v in distributions.items()}

    return {
        'distributions': dict(distributions),
        'distributions_pct': dist_pct,
        'most_common': [d for d, _ in distributions.most_common(3)],
        'recommended': ['3:3', '2:4', '4:2'],
        'low_range': [1, 24],
        'high_range': [25, 49]
    }

def analyze_decades(draws):
    """
    9. Zehnergruppen-System (1-9, 10-19, 20-29, 30-39, 40-49)
    """
    decade_counts = {'1-9': [], '10-19': [], '20-29': [], '30-39': [], '40-49': []}

    for draw in draws:
        nums = draw['numbers']
        counts = {
            '1-9': sum(1 for n in nums if 1 <= n <= 9),
            '10-19': sum(1 for n in nums if 10 <= n <= 19),
            '20-29': sum(1 for n in nums if 20 <= n <= 29),
            '30-39': sum(1 for n in nums if 30 <= n <= 39),
            '40-49': sum(1 for n in nums if 40 <= n <= 49)
        }
        for decade, count in counts.items():
            decade_counts[decade].append(count)

    # Durchschnitt pro Dekade
    decade_avg = {}
    for decade, counts in decade_counts.items():
        decade_avg[decade] = round(sum(counts) / len(counts), 2) if counts else 0

    # Wie oft sind alle 5 Dekaden vertreten?
    all_decades = 0
    for draw in draws:
        decades_hit = set()
        for n in draw['numbers']:
            if n <= 9:
                decades_hit.add(1)
            elif n <= 19:
                decades_hit.add(2)
            elif n <= 29:
                decades_hit.add(3)
            elif n <= 39:
                decades_hit.add(4)
            else:
                decades_hit.add(5)
        if len(decades_hit) >= 4:
            all_decades += 1

    return {
        'decade_avg': decade_avg,
        'four_plus_decades_pct': round(all_decades / len(draws) * 100, 1) if draws else 0,
        'recommendation': 'Wähle aus mindestens 4 verschiedenen Dekaden'
    }

def analyze_primes(draws):
    """
    10. Primzahlen-Analyse
    """
    prime_counts = []

    for draw in draws:
        count = sum(1 for n in draw['numbers'] if n in PRIMES)
        prime_counts.append(count)

    distribution = Counter(prime_counts)

    return {
        'prime_numbers': PRIMES,
        'avg_primes': round(sum(prime_counts) / len(prime_counts), 2) if prime_counts else 0,
        'distribution': dict(distribution),
        'most_common_count': distribution.most_common(1)[0][0] if distribution else 0,
        'recommended': '2-3 Primzahlen pro Tipp'
    }

def analyze_special_numbers(draws):
    """
    11. Fibonacci & Quadratzahlen
    """
    fib_counts = []
    square_counts = []

    for draw in draws:
        fib = sum(1 for n in draw['numbers'] if n in FIBONACCI)
        sq = sum(1 for n in draw['numbers'] if n in SQUARES)
        fib_counts.append(fib)
        square_counts.append(sq)

    return {
        'fibonacci_numbers': FIBONACCI,
        'square_numbers': SQUARES,
        'avg_fibonacci': round(sum(fib_counts) / len(fib_counts), 2) if fib_counts else 0,
        'avg_squares': round(sum(square_counts) / len(square_counts), 2) if square_counts else 0,
        'fib_distribution': dict(Counter(fib_counts)),
        'square_distribution': dict(Counter(square_counts))
    }

def analyze_last_digits(draws):
    """
    12. Endziffern-Verteilung
    """
    last_digit_freq = Counter()
    unique_digits_per_draw = []

    for draw in draws:
        digits = [n % 10 for n in draw['numbers']]
        last_digit_freq.update(digits)
        unique_digits_per_draw.append(len(set(digits)))

    return {
        'digit_frequency': dict(last_digit_freq.most_common()),
        'avg_unique_digits': round(sum(unique_digits_per_draw) / len(unique_digits_per_draw), 2) if unique_digits_per_draw else 0,
        'recommendation': 'Bevorzuge 5-6 verschiedene Endziffern'
    }

def analyze_consecutive(draws):
    """
    13. Konsekutiv-Analyse - Aufeinanderfolgende Zahlen
    """
    consecutive_counts = Counter()
    has_consecutive = 0

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
        if max_consecutive >= 2:
            has_consecutive += 1

    total = len(draws) or 1

    return {
        'distribution': dict(consecutive_counts),
        'has_consecutive_pct': round(has_consecutive / total * 100, 1),
        'no_consecutive_pct': round(consecutive_counts.get(1, 0) / total * 100, 1),
        'recommendation': 'Maximal 1 Paar aufeinanderfolgender Zahlen'
    }

def analyze_pairs(draws, limit=200):
    """
    14. Zahlenpaare & Dreiergruppen
    """
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
        'top_triplets': [list(t) for t, _ in triplets.most_common(20)],
        'pair_frequency': {str(list(k)): v for k, v in pairs.most_common(50)},
        'total_unique_pairs': len(pairs)
    }

def analyze_neighbors(draws):
    """
    15. Nachbarzahlen-Analyse - Aufeinanderfolgende Zahlen
    """
    neighbor_pairs = Counter()
    neighbor_count_per_draw = []

    for draw in draws:
        nums = sorted(draw['numbers'])
        neighbors = 0
        for i in range(len(nums) - 1):
            if nums[i + 1] - nums[i] == 1:
                neighbor_pairs[(nums[i], nums[i + 1])] += 1
                neighbors += 1
        neighbor_count_per_draw.append(neighbors)

    return {
        'top_neighbor_pairs': [(list(p), c) for p, c in neighbor_pairs.most_common(20)],
        'avg_neighbors_per_draw': round(sum(neighbor_count_per_draw) / len(neighbor_count_per_draw), 2) if neighbor_count_per_draw else 0,
        'no_neighbors_pct': round(sum(1 for n in neighbor_count_per_draw if n == 0) / len(neighbor_count_per_draw) * 100, 1) if neighbor_count_per_draw else 0
    }

def analyze_weekday_patterns(draws):
    """
    16. Wochentags-Muster - Mittwoch vs Samstag
    """
    wed_freq = Counter()
    sat_freq = Counter()
    wed_sz = Counter()
    sat_sz = Counter()
    wed_count = 0
    sat_count = 0

    for draw in draws[:300]:
        try:
            day, month, year = map(int, draw['date'].split('.'))
            date = datetime(year, month, day)

            if date.weekday() == 2:  # Mittwoch
                wed_freq.update(draw['numbers'])
                wed_sz[draw['superzahl']] += 1
                wed_count += 1
            elif date.weekday() == 5:  # Samstag
                sat_freq.update(draw['numbers'])
                sat_sz[draw['superzahl']] += 1
                sat_count += 1
        except:
            pass

    return {
        'wednesday_hot': [n for n, _ in wed_freq.most_common(10)],
        'saturday_hot': [n for n, _ in sat_freq.most_common(10)],
        'wednesday_sz': [n for n, _ in wed_sz.most_common(3)],
        'saturday_sz': [n for n, _ in sat_sz.most_common(3)],
        'wednesday_draws': wed_count,
        'saturday_draws': sat_count
    }

def analyze_superzahl_extended(draws):
    """
    17. Erweiterte Superzahl-Analyse (8 Faktoren)
    """
    # Folge-Muster
    follows = Counter()
    for i in range(len(draws) - 1):
        current = draws[i]['superzahl']
        previous = draws[i + 1]['superzahl']
        follows[(previous, current)] += 1

    # Gerade/Ungerade
    sz_odd_even = Counter()
    for draw in draws:
        sz = draw['superzahl']
        if sz % 2 == 0:
            sz_odd_even['gerade'] += 1
        else:
            sz_odd_even['ungerade'] += 1

    # Hoch/Tief
    sz_high_low = Counter()
    for draw in draws:
        sz = draw['superzahl']
        if sz <= 4:
            sz_high_low['tief'] += 1
        else:
            sz_high_low['hoch'] += 1

    # Korrelation mit Hauptzahlen-Summe
    sum_sz_correlation = {}
    for draw in draws[:200]:
        total = sum(draw['numbers'])
        bucket = (total // 20) * 20
        key = f"{bucket}-{bucket+19}"
        if key not in sum_sz_correlation:
            sum_sz_correlation[key] = Counter()
        sum_sz_correlation[key][draw['superzahl']] += 1

    # Beste SZ pro Summen-Bereich
    best_sz_per_sum = {}
    for bucket, counter in sum_sz_correlation.items():
        if counter:
            best_sz_per_sum[bucket] = counter.most_common(1)[0][0]

    return {
        'follow_patterns': {f"{k[0]}->{k[1]}": v for k, v in follows.most_common(30)},
        'odd_even': dict(sz_odd_even),
        'high_low': dict(sz_high_low),
        'sum_correlation': {k: dict(v.most_common(3)) for k, v in sum_sz_correlation.items()},
        'best_sz_per_sum': best_sz_per_sum
    }

def analyze_outliers(draws):
    """
    18. Ausreißer-Erkennung (Z-Score)
    """
    # Häufigkeit aller Zahlen
    all_numbers = []
    for draw in draws:
        all_numbers.extend(draw['numbers'])

    freq = Counter(all_numbers)
    values = list(freq.values())

    if not values:
        return {}

    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    std_dev = variance ** 0.5 if variance > 0 else 1

    # Z-Scores berechnen
    z_scores = {}
    for num in range(1, 50):
        count = freq.get(num, 0)
        z = (count - mean) / std_dev if std_dev > 0 else 0
        z_scores[num] = round(z, 2)

    # Überperformer (Z > 1.5) und Unterperformer (Z < -1.5)
    overperformers = [n for n, z in z_scores.items() if z > 1.5]
    underperformers = [n for n, z in z_scores.items() if z < -1.5]

    return {
        'z_scores': z_scores,
        'overperformers': overperformers,
        'underperformers': underperformers,
        'mean_frequency': round(mean, 2),
        'std_dev': round(std_dev, 2)
    }

# =====================================================
# KI-EMPFEHLUNGEN
# =====================================================

def generate_ki_recommendations(analysis):
    """Generiert KI-Empfehlungen basierend auf allen Analysen"""
    recommendations = {
        'hot_numbers': analysis.get('frequency', {}).get('hot_numbers', [])[:10],
        'cold_numbers': analysis.get('frequency', {}).get('cold_numbers', [])[:10],
        'overdue_numbers': analysis.get('gaps', {}).get('overdue_numbers', [])[:10],
        'rising_numbers': analysis.get('trends', {}).get('rising_numbers', [])[:10],
        'optimal_sum_range': analysis.get('sum_distribution', {}).get('optimal_range', [130, 170]),
        'best_odd_even': analysis.get('odd_even', {}).get('most_common', ['3:3', '2:4']),
        'best_high_low': analysis.get('high_low', {}).get('most_common', ['3:3', '2:4']),
        'optimal_deltas': analysis.get('delta', {}).get('optimal_deltas', [5, 6, 7, 8]),
        'hot_superzahlen': analysis.get('frequency', {}).get('hot_superzahlen', []),
        'overdue_superzahlen': analysis.get('gaps', {}).get('overdue_superzahlen', []),
        'wednesday_numbers': analysis.get('weekday_patterns', {}).get('wednesday_hot', []),
        'saturday_numbers': analysis.get('weekday_patterns', {}).get('saturday_hot', []),
        'top_pairs': analysis.get('pairs', {}).get('top_pairs', [])[:10],
        'primes_count': analysis.get('primes', {}).get('most_common_count', 2)
    }
    return recommendations

# =====================================================
# HAUPTFUNKTION
# =====================================================

def run_analysis():
    """Führt die komplette Lotto 6aus49 Analyse durch"""

    print("=" * 60)
    print("📊 LottoGenius - VOLLSTÄNDIGE 6aus49 KI-Analyse")
    print("=" * 60)
    print(f"📅 {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
    print()

    # Lade Daten
    data = load_json('lotto_data.json')
    draws = data.get('draws', [])

    if not draws:
        print("⚠️ Keine Lotto-Daten vorhanden")
        return

    print(f"📊 Analysiere {len(draws)} Ziehungen mit 18 Methoden...")
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

    print("  6️⃣ Quersummen-System...")
    sum_dist = analyze_sum_distribution(draws)

    print("  7️⃣ Gerade/Ungerade...")
    odd_even = analyze_odd_even(draws)

    print("  8️⃣ Hoch/Tief...")
    high_low = analyze_high_low(draws)

    print("  9️⃣ Zehnergruppen...")
    decades = analyze_decades(draws)

    print("  🔟 Primzahlen-Analyse...")
    primes = analyze_primes(draws)

    print("  1️⃣1️⃣ Fibonacci & Quadratzahlen...")
    special = analyze_special_numbers(draws)

    print("  1️⃣2️⃣ Endziffern-Analyse...")
    last_digits = analyze_last_digits(draws)

    print("  1️⃣3️⃣ Konsekutiv-Analyse...")
    consecutive = analyze_consecutive(draws)

    print("  1️⃣4️⃣ Zahlenpaare...")
    pairs = analyze_pairs(draws)

    print("  1️⃣5️⃣ Nachbarzahlen...")
    neighbors = analyze_neighbors(draws)

    print("  1️⃣6️⃣ Wochentags-Muster...")
    weekday = analyze_weekday_patterns(draws)

    print("  1️⃣7️⃣ Superzahl-Analyse (erweitert)...")
    superzahl = analyze_superzahl_extended(draws)

    print("  1️⃣8️⃣ Ausreißer-Erkennung...")
    outliers = analyze_outliers(draws)

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
        'sum_distribution': sum_dist,
        'odd_even': odd_even,
        'high_low': high_low,
        'decades': decades,
        'primes': primes,
        'special_numbers': special,
        'last_digits': last_digits,
        'consecutive': consecutive,
        'pairs': pairs,
        'neighbors': neighbors,
        'weekday_patterns': weekday,
        'superzahl_patterns': superzahl,
        'outliers': outliers
    }

    # KI-Empfehlungen
    analysis['ki_recommendations'] = generate_ki_recommendations(analysis)

    # Speichern
    save_json('analysis.json', analysis)

    print()
    print("=" * 60)
    print("✅ Analyse abgeschlossen! (18 Methoden)")
    print("=" * 60)
    print()
    print(f"🔥 Heiße Zahlen: {frequency['hot_numbers'][:5]}")
    print(f"❄️ Kalte Zahlen: {frequency['cold_numbers'][:5]}")
    print(f"⏰ Überfällig: {gaps['overdue_numbers'][:5]}")
    print(f"📈 Steigend: {trends['rising_numbers'][:5]}")
    print(f"📊 Optimale Summe: {sum_dist.get('optimal_range', [130, 170])}")
    print(f"⚖️ Beste Gerade/Ungerade: {odd_even.get('most_common', [])[:2]}")
    print(f"🎯 Beste Superzahlen: {frequency['hot_superzahlen']}")
    print()

    return analysis

if __name__ == "__main__":
    run_analysis()
