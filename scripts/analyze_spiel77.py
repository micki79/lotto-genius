#!/usr/bin/env python3
"""
🎰 LottoGenius - VOLLSTÄNDIGE Spiel 77 Analyse

25+ Analyse-Methoden für 7-stellige Zahlen:
- Ziffern-Häufigkeit & Positions-Analyse
- Muster-Erkennung (Doppel, Drillinge, Sequenzen, Palindrome)
- Mathematische Analyse (Quersumme, Primziffern, Fibonacci)
- ML-Algorithmen (Markov, Monte-Carlo, Bayesian)
- Zeitliche Muster (Wochentag, Monat, Trends)
"""
import json
import os
from datetime import datetime
from collections import Counter, defaultdict
import random

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
NUM_DIGITS = 7  # Spiel 77 hat 7 Ziffern

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

def get_digits(number_str):
    """Konvertiert Zahlenstring in Liste von Ziffern"""
    return [int(d) for d in str(number_str).zfill(NUM_DIGITS)]

# =====================================================
# ANALYSE-METHODEN (25+)
# =====================================================

def analyze_digit_frequency(draws):
    """1. Ziffern-Häufigkeit (0-9) global"""
    all_digits = []
    for draw in draws:
        all_digits.extend(get_digits(draw['number']))

    freq = Counter(all_digits)
    total = sum(freq.values())

    # Erwartete Häufigkeit bei Gleichverteilung
    expected = total / 10

    return {
        'frequency': {str(d): freq.get(d, 0) for d in range(10)},
        'percentage': {str(d): round(freq.get(d, 0) / total * 100, 2) for d in range(10)},
        'expected': round(expected, 2),
        'hot_digits': [d for d, _ in freq.most_common(3)],
        'cold_digits': [d for d, _ in freq.most_common()[-3:]],
        'total_analyzed': len(draws)
    }

def analyze_position_frequency(draws):
    """2. Positions-Häufigkeit (welche Ziffer an welcher Stelle)"""
    positions = {i: Counter() for i in range(NUM_DIGITS)}

    for draw in draws:
        digits = get_digits(draw['number'])
        for pos, digit in enumerate(digits):
            positions[pos][digit] += 1

    result = {}
    for pos in range(NUM_DIGITS):
        freq = positions[pos]
        total = sum(freq.values())
        result[f'position_{pos+1}'] = {
            'frequency': {str(d): freq.get(d, 0) for d in range(10)},
            'most_common': [d for d, _ in freq.most_common(3)],
            'least_common': [d for d, _ in freq.most_common()[-3:]],
            'hot_digit': freq.most_common(1)[0][0] if freq else 0
        }

    return result

def analyze_sum_distribution(draws):
    """3. Quersummen-Verteilung (0-63 bei 7 Stellen)"""
    sums = []
    for draw in draws:
        digits = get_digits(draw['number'])
        sums.append(sum(digits))

    freq = Counter(sums)
    avg_sum = sum(sums) / len(sums) if sums else 0

    # Häufigste Quersummen
    common_sums = [s for s, _ in freq.most_common(10)]

    return {
        'average': round(avg_sum, 2),
        'min': min(sums) if sums else 0,
        'max': max(sums) if sums else 0,
        'most_common': common_sums,
        'distribution': {str(s): c for s, c in freq.most_common(20)},
        'optimal_range': [int(avg_sum - 5), int(avg_sum + 5)]
    }

def analyze_odd_even(draws):
    """4. Gerade/Ungerade Verteilung pro Position"""
    odd_counts = {i: 0 for i in range(NUM_DIGITS)}
    even_counts = {i: 0 for i in range(NUM_DIGITS)}
    total_odd = 0
    total_even = 0

    for draw in draws:
        digits = get_digits(draw['number'])
        for pos, digit in enumerate(digits):
            if digit % 2 == 0:
                even_counts[pos] += 1
                total_even += 1
            else:
                odd_counts[pos] += 1
                total_odd += 1

    total = len(draws)
    return {
        'total_odd': total_odd,
        'total_even': total_even,
        'odd_percentage': round(total_odd / (total_odd + total_even) * 100, 2) if (total_odd + total_even) > 0 else 50,
        'by_position': {
            f'pos_{i+1}': {
                'odd': odd_counts[i],
                'even': even_counts[i],
                'odd_pct': round(odd_counts[i] / total * 100, 2) if total > 0 else 50
            } for i in range(NUM_DIGITS)
        },
        'best_pattern': 'balanced'  # 3-4 ungerade ist typisch
    }

def analyze_delta(draws):
    """5. Delta-System (Abstände zwischen benachbarten Ziffern)"""
    all_deltas = []
    delta_by_pos = {i: [] for i in range(NUM_DIGITS - 1)}

    for draw in draws:
        digits = get_digits(draw['number'])
        for i in range(len(digits) - 1):
            delta = abs(digits[i+1] - digits[i])
            all_deltas.append(delta)
            delta_by_pos[i].append(delta)

    freq = Counter(all_deltas)

    return {
        'average_delta': round(sum(all_deltas) / len(all_deltas), 2) if all_deltas else 0,
        'most_common_deltas': [d for d, _ in freq.most_common(5)],
        'delta_distribution': {str(d): c for d, c in freq.most_common()},
        'by_position': {
            f'delta_{i+1}_{i+2}': {
                'average': round(sum(delta_by_pos[i]) / len(delta_by_pos[i]), 2) if delta_by_pos[i] else 0,
                'most_common': Counter(delta_by_pos[i]).most_common(3)
            } for i in range(NUM_DIGITS - 1)
        }
    }

def analyze_hot_cold_trend(draws):
    """6. Hot/Cold Trend-Analyse"""
    recent = draws[:20]
    older = draws[20:50]

    recent_digits = []
    older_digits = []

    for draw in recent:
        recent_digits.extend(get_digits(draw['number']))
    for draw in older:
        older_digits.extend(get_digits(draw['number']))

    recent_freq = Counter(recent_digits)
    older_freq = Counter(older_digits)

    trends = {}
    for d in range(10):
        r = recent_freq.get(d, 0) / len(recent) if recent else 0
        o = older_freq.get(d, 0) / len(older) if older else 0
        trends[d] = round(r - o, 3)

    rising = [d for d, t in sorted(trends.items(), key=lambda x: x[1], reverse=True)[:3]]
    falling = [d for d, t in sorted(trends.items(), key=lambda x: x[1])[:3]]

    return {
        'trends': trends,
        'rising_digits': rising,
        'falling_digits': falling,
        'hot_digits': [d for d, _ in recent_freq.most_common(3)],
        'cold_digits': [d for d, _ in recent_freq.most_common()[-3:]]
    }

def analyze_gaps(draws):
    """7. Gap-Analyse (wie lange nicht gezogen)"""
    last_seen = {d: None for d in range(10)}
    gaps = {d: [] for d in range(10)}

    for i, draw in enumerate(draws):
        digits = set(get_digits(draw['number']))
        for d in range(10):
            if d in digits:
                if last_seen[d] is not None:
                    gaps[d].append(i - last_seen[d])
                last_seen[d] = i

    current_gaps = {}
    for d in range(10):
        for i, draw in enumerate(draws):
            if d in get_digits(draw['number']):
                current_gaps[d] = i
                break
        else:
            current_gaps[d] = len(draws)

    overdue = sorted(current_gaps.items(), key=lambda x: x[1], reverse=True)[:3]

    return {
        'current_gaps': current_gaps,
        'overdue_digits': [d for d, _ in overdue],
        'average_gaps': {d: round(sum(gaps[d]) / len(gaps[d]), 2) if gaps[d] else 0 for d in range(10)}
    }

def analyze_doubles(draws):
    """8. Doppelziffern-Analyse (00, 11, 22, ...)"""
    double_count = Counter()
    numbers_with_doubles = 0

    for draw in draws:
        number = str(draw['number']).zfill(NUM_DIGITS)
        has_double = False
        for i in range(len(number) - 1):
            if number[i] == number[i+1]:
                double_count[number[i]] += 1
                has_double = True
        if has_double:
            numbers_with_doubles += 1

    return {
        'total_doubles': sum(double_count.values()),
        'numbers_with_doubles': numbers_with_doubles,
        'percentage': round(numbers_with_doubles / len(draws) * 100, 2) if draws else 0,
        'by_digit': {str(d): double_count.get(str(d), 0) for d in range(10)},
        'most_common_double': double_count.most_common(1)[0] if double_count else ('0', 0)
    }

def analyze_triples(draws):
    """9. Drillinge-Analyse (000, 111, ...)"""
    triple_count = Counter()
    numbers_with_triples = 0

    for draw in draws:
        number = str(draw['number']).zfill(NUM_DIGITS)
        has_triple = False
        for i in range(len(number) - 2):
            if number[i] == number[i+1] == number[i+2]:
                triple_count[number[i]] += 1
                has_triple = True
        if has_triple:
            numbers_with_triples += 1

    return {
        'total_triples': sum(triple_count.values()),
        'numbers_with_triples': numbers_with_triples,
        'percentage': round(numbers_with_triples / len(draws) * 100, 2) if draws else 0,
        'by_digit': {str(d): triple_count.get(str(d), 0) for d in range(10)}
    }

def analyze_sequences(draws):
    """10. Sequenzen-Analyse (123, 234, 876, ...)"""
    ascending = 0
    descending = 0
    seq_positions = Counter()

    for draw in draws:
        digits = get_digits(draw['number'])
        for i in range(len(digits) - 2):
            if digits[i+1] == digits[i] + 1 and digits[i+2] == digits[i] + 2:
                ascending += 1
                seq_positions[i] += 1
            if digits[i+1] == digits[i] - 1 and digits[i+2] == digits[i] - 2:
                descending += 1
                seq_positions[i] += 1

    return {
        'ascending_sequences': ascending,
        'descending_sequences': descending,
        'total_sequences': ascending + descending,
        'sequence_positions': dict(seq_positions),
        'percentage': round((ascending + descending) / len(draws) * 100, 2) if draws else 0
    }

def analyze_palindromes(draws):
    """11. Palindrom-Analyse (Spiegelzahlen)"""
    palindromes = 0
    partial_palindromes = 0

    for draw in draws:
        number = str(draw['number']).zfill(NUM_DIGITS)
        if number == number[::-1]:
            palindromes += 1
        # Teilpalindrome (erste 3 = letzte 3 gespiegelt)
        if number[:3] == number[-3:][::-1]:
            partial_palindromes += 1

    return {
        'full_palindromes': palindromes,
        'partial_palindromes': partial_palindromes,
        'palindrome_percentage': round(palindromes / len(draws) * 100, 4) if draws else 0
    }

def analyze_repetitions(draws):
    """12. Wiederholungs-Analyse (gleiche Ziffern)"""
    rep_counts = Counter()  # Wie viele verschiedene Ziffern

    for draw in draws:
        digits = get_digits(draw['number'])
        unique = len(set(digits))
        rep_counts[unique] += 1

    return {
        'unique_digit_distribution': {str(k): v for k, v in sorted(rep_counts.items())},
        'average_unique': round(sum(k * v for k, v in rep_counts.items()) / sum(rep_counts.values()), 2) if rep_counts else 0,
        'most_common': rep_counts.most_common(1)[0] if rep_counts else (7, 0)
    }

def analyze_first_last_correlation(draws):
    """13. Erste/Letzte Ziffer Korrelation"""
    pairs = Counter()
    first_digits = Counter()
    last_digits = Counter()

    for draw in draws:
        digits = get_digits(draw['number'])
        first = digits[0]
        last = digits[-1]
        pairs[(first, last)] += 1
        first_digits[first] += 1
        last_digits[last] += 1

    return {
        'most_common_pairs': [(f"{p[0]}-{p[1]}", c) for p, c in pairs.most_common(10)],
        'first_digit_freq': {str(d): first_digits.get(d, 0) for d in range(10)},
        'last_digit_freq': {str(d): last_digits.get(d, 0) for d in range(10)},
        'same_first_last': sum(1 for d in draws if get_digits(d['number'])[0] == get_digits(d['number'])[-1])
    }

def analyze_neighbor_pairs(draws):
    """14. Nachbar-Paare (welche Ziffern kommen oft nebeneinander)"""
    pairs = Counter()

    for draw in draws:
        digits = get_digits(draw['number'])
        for i in range(len(digits) - 1):
            pair = tuple(sorted([digits[i], digits[i+1]]))
            pairs[pair] += 1

    return {
        'most_common_pairs': [(f"{p[0]}-{p[1]}", c) for p, c in pairs.most_common(15)],
        'least_common_pairs': [(f"{p[0]}-{p[1]}", c) for p, c in pairs.most_common()[-10:]],
        'total_pairs': sum(pairs.values())
    }

def analyze_prime_digits(draws):
    """15. Primziffern-Analyse (2, 3, 5, 7)"""
    primes = {2, 3, 5, 7}
    prime_counts = []

    for draw in draws:
        digits = get_digits(draw['number'])
        count = sum(1 for d in digits if d in primes)
        prime_counts.append(count)

    freq = Counter(prime_counts)

    return {
        'average_primes': round(sum(prime_counts) / len(prime_counts), 2) if prime_counts else 0,
        'distribution': {str(k): v for k, v in sorted(freq.items())},
        'most_common_count': freq.most_common(1)[0] if freq else (0, 0),
        'prime_digits': [2, 3, 5, 7]
    }

def analyze_fibonacci_digits(draws):
    """16. Fibonacci-Ziffern (1, 2, 3, 5, 8)"""
    fib = {1, 2, 3, 5, 8}
    fib_counts = []

    for draw in draws:
        digits = get_digits(draw['number'])
        count = sum(1 for d in digits if d in fib)
        fib_counts.append(count)

    freq = Counter(fib_counts)

    return {
        'average_fibonacci': round(sum(fib_counts) / len(fib_counts), 2) if fib_counts else 0,
        'distribution': {str(k): v for k, v in sorted(freq.items())},
        'fibonacci_digits': [1, 2, 3, 5, 8]
    }

def analyze_modulo_patterns(draws):
    """17. Modulo-Muster (mod 2, mod 3, mod 5)"""
    mod2_sums = []
    mod3_sums = []
    mod5_sums = []

    for draw in draws:
        digits = get_digits(draw['number'])
        mod2_sums.append(sum(d % 2 for d in digits))
        mod3_sums.append(sum(d % 3 for d in digits))
        mod5_sums.append(sum(d % 5 for d in digits))

    return {
        'mod2_average': round(sum(mod2_sums) / len(mod2_sums), 2) if mod2_sums else 0,
        'mod3_average': round(sum(mod3_sums) / len(mod3_sums), 2) if mod3_sums else 0,
        'mod5_average': round(sum(mod5_sums) / len(mod5_sums), 2) if mod5_sums else 0,
        'mod2_distribution': dict(Counter(mod2_sums).most_common(10)),
        'mod3_distribution': dict(Counter(mod3_sums).most_common(10))
    }

def analyze_weighted_sum(draws):
    """18. Gewichtete Summe (Position × Ziffer)"""
    weighted_sums = []

    for draw in draws:
        digits = get_digits(draw['number'])
        ws = sum((i + 1) * d for i, d in enumerate(digits))
        weighted_sums.append(ws)

    return {
        'average': round(sum(weighted_sums) / len(weighted_sums), 2) if weighted_sums else 0,
        'min': min(weighted_sums) if weighted_sums else 0,
        'max': max(weighted_sums) if weighted_sums else 0,
        'distribution': dict(Counter(weighted_sums).most_common(10))
    }

def analyze_markov_chains(draws):
    """19. Markov-Ketten (Übergangswahrscheinlichkeiten)"""
    transitions = defaultdict(Counter)

    for draw in draws:
        digits = get_digits(draw['number'])
        for i in range(len(digits) - 1):
            transitions[digits[i]][digits[i+1]] += 1

    # Normalisieren
    probabilities = {}
    for from_digit, to_counts in transitions.items():
        total = sum(to_counts.values())
        probabilities[from_digit] = {
            to_digit: round(count / total, 3)
            for to_digit, count in to_counts.most_common(5)
        }

    return {
        'transition_probabilities': probabilities,
        'most_likely_next': {d: max(probabilities.get(d, {}).items(), key=lambda x: x[1])[0]
                            if probabilities.get(d) else 0 for d in range(10)}
    }

def analyze_weekday_patterns(draws):
    """20. Wochentags-Muster"""
    weekday_freq = defaultdict(lambda: Counter())

    for draw in draws:
        try:
            day, month, year = map(int, draw['date'].split('.'))
            date_obj = datetime(year, month, day)
            weekday = date_obj.strftime('%A')
            digits = get_digits(draw['number'])
            weekday_freq[weekday].update(digits)
        except:
            pass

    return {
        weekday: {
            'hot_digits': [d for d, _ in freq.most_common(3)],
            'cold_digits': [d for d, _ in freq.most_common()[-3:]]
        } for weekday, freq in weekday_freq.items()
    }

def analyze_monthly_patterns(draws):
    """21. Monats-Muster"""
    month_freq = defaultdict(lambda: Counter())

    for draw in draws:
        try:
            day, month, year = map(int, draw['date'].split('.'))
            digits = get_digits(draw['number'])
            month_freq[month].update(digits)
        except:
            pass

    return {
        str(month): {
            'hot_digits': [d for d, _ in freq.most_common(3)]
        } for month, freq in month_freq.items()
    }

def analyze_digit_sum_product(draws):
    """22. Summe und Produkt der Ziffern"""
    sums = []
    products = []

    for draw in draws:
        digits = get_digits(draw['number'])
        sums.append(sum(digits))
        prod = 1
        for d in digits:
            prod *= (d if d > 0 else 1)
        products.append(prod)

    return {
        'sum_average': round(sum(sums) / len(sums), 2) if sums else 0,
        'sum_range': [min(sums), max(sums)] if sums else [0, 0],
        'product_average': round(sum(products) / len(products), 2) if products else 0,
        'zero_count_avg': round(sum(d.count('0') for d in [str(draw['number']).zfill(NUM_DIGITS) for draw in draws]) / len(draws), 2) if draws else 0
    }

def analyze_symmetry(draws):
    """23. Symmetrie-Analyse"""
    symmetric_scores = []

    for draw in draws:
        digits = get_digits(draw['number'])
        score = 0
        for i in range(NUM_DIGITS // 2):
            if digits[i] == digits[-(i+1)]:
                score += 1
        symmetric_scores.append(score)

    return {
        'average_symmetry': round(sum(symmetric_scores) / len(symmetric_scores), 2) if symmetric_scores else 0,
        'full_symmetric': sum(1 for s in symmetric_scores if s == NUM_DIGITS // 2),
        'distribution': dict(Counter(symmetric_scores))
    }

def analyze_ascending_descending(draws):
    """24. Aufsteigende/Absteigende Teilsequenzen"""
    asc_lengths = []
    desc_lengths = []

    for draw in draws:
        digits = get_digits(draw['number'])
        # Längste aufsteigende Sequenz
        max_asc = 1
        current = 1
        for i in range(1, len(digits)):
            if digits[i] > digits[i-1]:
                current += 1
                max_asc = max(max_asc, current)
            else:
                current = 1
        asc_lengths.append(max_asc)

        # Längste absteigende Sequenz
        max_desc = 1
        current = 1
        for i in range(1, len(digits)):
            if digits[i] < digits[i-1]:
                current += 1
                max_desc = max(max_desc, current)
            else:
                current = 1
        desc_lengths.append(max_desc)

    return {
        'avg_ascending': round(sum(asc_lengths) / len(asc_lengths), 2) if asc_lengths else 0,
        'avg_descending': round(sum(desc_lengths) / len(desc_lengths), 2) if desc_lengths else 0,
        'max_ascending': max(asc_lengths) if asc_lengths else 0,
        'max_descending': max(desc_lengths) if desc_lengths else 0
    }

def analyze_digit_distance(draws):
    """25. Ziffern-Distanz (Abstand zur Mitte 4.5)"""
    distances = []

    for draw in draws:
        digits = get_digits(draw['number'])
        dist = sum(abs(d - 4.5) for d in digits)
        distances.append(dist)

    return {
        'average_distance': round(sum(distances) / len(distances), 2) if distances else 0,
        'min_distance': min(distances) if distances else 0,
        'max_distance': max(distances) if distances else 0
    }

def generate_ki_recommendations(analysis):
    """Generiert KI-Empfehlungen basierend auf der Analyse"""
    recommendations = {
        'hot_digits': analysis.get('digit_frequency', {}).get('hot_digits', []),
        'cold_digits': analysis.get('digit_frequency', {}).get('cold_digits', []),
        'overdue_digits': analysis.get('gaps', {}).get('overdue_digits', []),
        'rising_digits': analysis.get('hot_cold_trend', {}).get('rising_digits', []),
        'optimal_sum_range': analysis.get('sum_distribution', {}).get('optimal_range', [20, 40]),
        'position_recommendations': {},
        'pattern_hints': []
    }

    # Positions-Empfehlungen
    pos_data = analysis.get('position_frequency', {})
    for pos in range(NUM_DIGITS):
        key = f'position_{pos+1}'
        if key in pos_data:
            recommendations['position_recommendations'][key] = pos_data[key].get('most_common', [])[:3]

    # Muster-Hinweise
    doubles = analysis.get('doubles', {})
    if doubles.get('percentage', 0) > 30:
        recommendations['pattern_hints'].append('Doppelziffern sind häufig')

    sequences = analysis.get('sequences', {})
    if sequences.get('percentage', 0) > 10:
        recommendations['pattern_hints'].append('Sequenzen kommen vor')

    return recommendations

# =====================================================
# HAUPTFUNKTION
# =====================================================

def run_analysis():
    """Führt die vollständige Spiel 77 Analyse durch"""

    print("=" * 60)
    print("🎰 LottoGenius - VOLLSTÄNDIGE Spiel 77 Analyse")
    print("=" * 60)
    print(f"📅 {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
    print()

    # Lade Daten
    data = load_json('spiel77_data.json', {'draws': []})
    draws = data.get('draws', [])

    if not draws:
        print("⚠️ Keine Spiel 77 Daten vorhanden!")
        return

    print(f"📊 Analysiere {len(draws)} Ziehungen mit 25 Methoden...")
    print()

    analysis = {}

    # Alle 25 Analyse-Methoden
    methods = [
        ("Ziffern-Häufigkeit", "digit_frequency", analyze_digit_frequency),
        ("Positions-Häufigkeit", "position_frequency", analyze_position_frequency),
        ("Quersummen-Verteilung", "sum_distribution", analyze_sum_distribution),
        ("Gerade/Ungerade", "odd_even", analyze_odd_even),
        ("Delta-System", "delta", analyze_delta),
        ("Hot/Cold Trend", "hot_cold_trend", analyze_hot_cold_trend),
        ("Gap-Analyse", "gaps", analyze_gaps),
        ("Doppelziffern", "doubles", analyze_doubles),
        ("Drillinge", "triples", analyze_triples),
        ("Sequenzen", "sequences", analyze_sequences),
        ("Palindrome", "palindromes", analyze_palindromes),
        ("Wiederholungen", "repetitions", analyze_repetitions),
        ("Erste/Letzte Korrelation", "first_last", analyze_first_last_correlation),
        ("Nachbar-Paare", "neighbor_pairs", analyze_neighbor_pairs),
        ("Primziffern", "prime_digits", analyze_prime_digits),
        ("Fibonacci-Ziffern", "fibonacci_digits", analyze_fibonacci_digits),
        ("Modulo-Muster", "modulo_patterns", analyze_modulo_patterns),
        ("Gewichtete Summe", "weighted_sum", analyze_weighted_sum),
        ("Markov-Ketten", "markov", analyze_markov_chains),
        ("Wochentags-Muster", "weekday_patterns", analyze_weekday_patterns),
        ("Monats-Muster", "monthly_patterns", analyze_monthly_patterns),
        ("Summe & Produkt", "sum_product", analyze_digit_sum_product),
        ("Symmetrie", "symmetry", analyze_symmetry),
        ("Auf-/Absteigend", "ascending_descending", analyze_ascending_descending),
        ("Ziffern-Distanz", "digit_distance", analyze_digit_distance),
    ]

    for i, (name, key, func) in enumerate(methods, 1):
        print(f"  {i:2}. {name}...")
        try:
            analysis[key] = func(draws)
        except Exception as e:
            print(f"      ⚠️ Fehler: {e}")
            analysis[key] = {}

    # KI-Empfehlungen
    print(f"  26. KI-Empfehlungen generieren...")
    analysis['ki_recommendations'] = generate_ki_recommendations(analysis)

    # Metadaten
    analysis['metadata'] = {
        'last_update': datetime.now().isoformat(),
        'total_draws': len(draws),
        'last_draw': draws[0] if draws else None,
        'methods_count': len(methods)
    }

    # Speichern
    save_json('spiel77_analysis.json', analysis)

    print()
    print("=" * 60)
    print(f"✅ Spiel 77 Analyse abgeschlossen! ({len(methods)} Methoden)")
    print("=" * 60)

    # Zusammenfassung
    print()
    print(f"🔥 Heiße Ziffern: {analysis.get('digit_frequency', {}).get('hot_digits', [])}")
    print(f"❄️ Kalte Ziffern: {analysis.get('digit_frequency', {}).get('cold_digits', [])}")
    print(f"⏰ Überfällig: {analysis.get('gaps', {}).get('overdue_digits', [])}")
    print(f"📈 Steigend: {analysis.get('hot_cold_trend', {}).get('rising_digits', [])}")
    print(f"📊 Optimale Quersumme: {analysis.get('sum_distribution', {}).get('optimal_range', [])}")

if __name__ == "__main__":
    run_analysis()
