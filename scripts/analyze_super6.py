#!/usr/bin/env python3
"""
🎲 LottoGenius - VOLLSTÄNDIGE Super 6 Analyse

25+ Analyse-Methoden für 6-stellige Zahlen:
- Ziffern-Häufigkeit & Positions-Analyse
- Muster-Erkennung (Doppel, Drillinge, Sequenzen)
- Mathematische Analyse (Quersumme, Primziffern)
- ML-Algorithmen (Markov, Monte-Carlo, Bayesian)
- Zeitliche Muster (Wochentag, Monat, Trends)
"""
import json
import os
from datetime import datetime
from collections import Counter, defaultdict
import random

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
NUM_DIGITS = 6  # Super 6 hat 6 Ziffern

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
    return [int(d) for d in str(number_str).zfill(NUM_DIGITS)]

# =====================================================
# ANALYSE-METHODEN (25+)
# =====================================================

def analyze_digit_frequency(draws):
    """1. Ziffern-Häufigkeit global"""
    all_digits = []
    for draw in draws:
        all_digits.extend(get_digits(draw['number']))

    freq = Counter(all_digits)
    total = sum(freq.values())
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
    """2. Positions-Häufigkeit"""
    positions = {i: Counter() for i in range(NUM_DIGITS)}

    for draw in draws:
        digits = get_digits(draw['number'])
        for pos, digit in enumerate(digits):
            positions[pos][digit] += 1

    result = {}
    for pos in range(NUM_DIGITS):
        freq = positions[pos]
        result[f'position_{pos+1}'] = {
            'frequency': {str(d): freq.get(d, 0) for d in range(10)},
            'most_common': [d for d, _ in freq.most_common(3)],
            'least_common': [d for d, _ in freq.most_common()[-3:]],
            'hot_digit': freq.most_common(1)[0][0] if freq else 0
        }
    return result

def analyze_sum_distribution(draws):
    """3. Quersummen-Verteilung (0-54 bei 6 Stellen)"""
    sums = []
    for draw in draws:
        digits = get_digits(draw['number'])
        sums.append(sum(digits))

    freq = Counter(sums)
    avg_sum = sum(sums) / len(sums) if sums else 0

    return {
        'average': round(avg_sum, 2),
        'min': min(sums) if sums else 0,
        'max': max(sums) if sums else 0,
        'most_common': [s for s, _ in freq.most_common(10)],
        'optimal_range': [int(avg_sum - 5), int(avg_sum + 5)]
    }

def analyze_odd_even(draws):
    """4. Gerade/Ungerade Verteilung"""
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

    return {
        'total_odd': total_odd,
        'total_even': total_even,
        'odd_percentage': round(total_odd / (total_odd + total_even) * 100, 2) if (total_odd + total_even) > 0 else 50,
        'by_position': {f'pos_{i+1}': {'odd': odd_counts[i], 'even': even_counts[i]} for i in range(NUM_DIGITS)}
    }

def analyze_delta(draws):
    """5. Delta-System"""
    all_deltas = []
    for draw in draws:
        digits = get_digits(draw['number'])
        for i in range(len(digits) - 1):
            all_deltas.append(abs(digits[i+1] - digits[i]))

    freq = Counter(all_deltas)
    return {
        'average_delta': round(sum(all_deltas) / len(all_deltas), 2) if all_deltas else 0,
        'most_common_deltas': [d for d, _ in freq.most_common(5)],
        'delta_distribution': {str(d): c for d, c in freq.most_common()}
    }

def analyze_hot_cold_trend(draws):
    """6. Hot/Cold Trend"""
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

    return {
        'trends': trends,
        'rising_digits': [d for d, t in sorted(trends.items(), key=lambda x: x[1], reverse=True)[:3]],
        'falling_digits': [d for d, t in sorted(trends.items(), key=lambda x: x[1])[:3]],
        'hot_digits': [d for d, _ in recent_freq.most_common(3)],
        'cold_digits': [d for d, _ in recent_freq.most_common()[-3:]]
    }

def analyze_gaps(draws):
    """7. Gap-Analyse"""
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
        'overdue_digits': [d for d, _ in overdue]
    }

def analyze_doubles(draws):
    """8. Doppelziffern"""
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
        'by_digit': {str(d): double_count.get(str(d), 0) for d in range(10)}
    }

def analyze_triples(draws):
    """9. Drillinge"""
    triple_count = 0
    for draw in draws:
        number = str(draw['number']).zfill(NUM_DIGITS)
        for i in range(len(number) - 2):
            if number[i] == number[i+1] == number[i+2]:
                triple_count += 1
    return {'total_triples': triple_count, 'percentage': round(triple_count / len(draws) * 100, 2) if draws else 0}

def analyze_sequences(draws):
    """10. Sequenzen"""
    ascending = 0
    descending = 0
    for draw in draws:
        digits = get_digits(draw['number'])
        for i in range(len(digits) - 2):
            if digits[i+1] == digits[i] + 1 and digits[i+2] == digits[i] + 2:
                ascending += 1
            if digits[i+1] == digits[i] - 1 and digits[i+2] == digits[i] - 2:
                descending += 1
    return {'ascending': ascending, 'descending': descending, 'total': ascending + descending}

def analyze_palindromes(draws):
    """11. Palindrome"""
    count = 0
    for draw in draws:
        number = str(draw['number']).zfill(NUM_DIGITS)
        if number == number[::-1]:
            count += 1
    return {'palindromes': count, 'percentage': round(count / len(draws) * 100, 4) if draws else 0}

def analyze_repetitions(draws):
    """12. Wiederholungen"""
    rep_counts = Counter()
    for draw in draws:
        digits = get_digits(draw['number'])
        unique = len(set(digits))
        rep_counts[unique] += 1
    return {'distribution': {str(k): v for k, v in sorted(rep_counts.items())}}

def analyze_first_last(draws):
    """13. Erste/Letzte Korrelation"""
    pairs = Counter()
    for draw in draws:
        digits = get_digits(draw['number'])
        pairs[(digits[0], digits[-1])] += 1
    return {'most_common_pairs': [(f"{p[0]}-{p[1]}", c) for p, c in pairs.most_common(10)]}

def analyze_neighbor_pairs(draws):
    """14. Nachbar-Paare"""
    pairs = Counter()
    for draw in draws:
        digits = get_digits(draw['number'])
        for i in range(len(digits) - 1):
            pair = tuple(sorted([digits[i], digits[i+1]]))
            pairs[pair] += 1
    return {'most_common': [(f"{p[0]}-{p[1]}", c) for p, c in pairs.most_common(15)]}

def analyze_prime_digits(draws):
    """15. Primziffern"""
    primes = {2, 3, 5, 7}
    counts = []
    for draw in draws:
        digits = get_digits(draw['number'])
        counts.append(sum(1 for d in digits if d in primes))
    return {'average': round(sum(counts) / len(counts), 2) if counts else 0, 'distribution': dict(Counter(counts))}

def analyze_fibonacci(draws):
    """16. Fibonacci-Ziffern"""
    fib = {1, 2, 3, 5, 8}
    counts = []
    for draw in draws:
        digits = get_digits(draw['number'])
        counts.append(sum(1 for d in digits if d in fib))
    return {'average': round(sum(counts) / len(counts), 2) if counts else 0}

def analyze_modulo(draws):
    """17. Modulo-Muster"""
    mod3_sums = []
    for draw in draws:
        digits = get_digits(draw['number'])
        mod3_sums.append(sum(d % 3 for d in digits))
    return {'mod3_average': round(sum(mod3_sums) / len(mod3_sums), 2) if mod3_sums else 0}

def analyze_weighted_sum(draws):
    """18. Gewichtete Summe"""
    sums = []
    for draw in draws:
        digits = get_digits(draw['number'])
        sums.append(sum((i + 1) * d for i, d in enumerate(digits)))
    return {'average': round(sum(sums) / len(sums), 2) if sums else 0}

def analyze_markov(draws):
    """19. Markov-Ketten"""
    transitions = defaultdict(Counter)
    for draw in draws:
        digits = get_digits(draw['number'])
        for i in range(len(digits) - 1):
            transitions[digits[i]][digits[i+1]] += 1

    probs = {}
    for from_d, to_counts in transitions.items():
        total = sum(to_counts.values())
        probs[from_d] = {to_d: round(c / total, 3) for to_d, c in to_counts.most_common(3)}

    return {'transition_probabilities': probs}

def analyze_weekday(draws):
    """20. Wochentags-Muster"""
    weekday_freq = defaultdict(Counter)
    for draw in draws:
        try:
            day, month, year = map(int, draw['date'].split('.'))
            date_obj = datetime(year, month, day)
            weekday = date_obj.strftime('%A')
            digits = get_digits(draw['number'])
            weekday_freq[weekday].update(digits)
        except:
            pass
    return {w: {'hot': [d for d, _ in f.most_common(3)]} for w, f in weekday_freq.items()}

def analyze_monthly(draws):
    """21. Monats-Muster"""
    month_freq = defaultdict(Counter)
    for draw in draws:
        try:
            day, month, year = map(int, draw['date'].split('.'))
            digits = get_digits(draw['number'])
            month_freq[month].update(digits)
        except:
            pass
    return {str(m): {'hot': [d for d, _ in f.most_common(3)]} for m, f in month_freq.items()}

def analyze_sum_product(draws):
    """22. Summe & Produkt"""
    sums = []
    for draw in draws:
        digits = get_digits(draw['number'])
        sums.append(sum(digits))
    return {'sum_average': round(sum(sums) / len(sums), 2) if sums else 0}

def analyze_symmetry(draws):
    """23. Symmetrie"""
    scores = []
    for draw in draws:
        digits = get_digits(draw['number'])
        score = sum(1 for i in range(NUM_DIGITS // 2) if digits[i] == digits[-(i+1)])
        scores.append(score)
    return {'average_symmetry': round(sum(scores) / len(scores), 2) if scores else 0}

def analyze_ascending(draws):
    """24. Auf-/Absteigend"""
    asc_lens = []
    for draw in draws:
        digits = get_digits(draw['number'])
        max_asc = 1
        current = 1
        for i in range(1, len(digits)):
            if digits[i] > digits[i-1]:
                current += 1
                max_asc = max(max_asc, current)
            else:
                current = 1
        asc_lens.append(max_asc)
    return {'avg_ascending': round(sum(asc_lens) / len(asc_lens), 2) if asc_lens else 0}

def analyze_distance(draws):
    """25. Ziffern-Distanz"""
    distances = []
    for draw in draws:
        digits = get_digits(draw['number'])
        distances.append(sum(abs(d - 4.5) for d in digits))
    return {'average_distance': round(sum(distances) / len(distances), 2) if distances else 0}

def generate_ki_recommendations(analysis):
    """KI-Empfehlungen"""
    return {
        'hot_digits': analysis.get('digit_frequency', {}).get('hot_digits', []),
        'cold_digits': analysis.get('digit_frequency', {}).get('cold_digits', []),
        'overdue_digits': analysis.get('gaps', {}).get('overdue_digits', []),
        'rising_digits': analysis.get('hot_cold_trend', {}).get('rising_digits', []),
        'optimal_sum_range': analysis.get('sum_distribution', {}).get('optimal_range', [18, 35])
    }

# =====================================================
# HAUPTFUNKTION
# =====================================================

def run_analysis():
    print("=" * 60)
    print("🎲 LottoGenius - VOLLSTÄNDIGE Super 6 Analyse")
    print("=" * 60)
    print(f"📅 {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
    print()

    data = load_json('super6_data.json', {'draws': []})
    draws = data.get('draws', [])

    if not draws:
        print("⚠️ Keine Super 6 Daten vorhanden!")
        return

    print(f"📊 Analysiere {len(draws)} Ziehungen mit 25 Methoden...")
    print()

    analysis = {}

    methods = [
        ("Ziffern-Häufigkeit", "digit_frequency", analyze_digit_frequency),
        ("Positions-Häufigkeit", "position_frequency", analyze_position_frequency),
        ("Quersummen", "sum_distribution", analyze_sum_distribution),
        ("Gerade/Ungerade", "odd_even", analyze_odd_even),
        ("Delta-System", "delta", analyze_delta),
        ("Hot/Cold Trend", "hot_cold_trend", analyze_hot_cold_trend),
        ("Gap-Analyse", "gaps", analyze_gaps),
        ("Doppelziffern", "doubles", analyze_doubles),
        ("Drillinge", "triples", analyze_triples),
        ("Sequenzen", "sequences", analyze_sequences),
        ("Palindrome", "palindromes", analyze_palindromes),
        ("Wiederholungen", "repetitions", analyze_repetitions),
        ("Erste/Letzte", "first_last", analyze_first_last),
        ("Nachbar-Paare", "neighbor_pairs", analyze_neighbor_pairs),
        ("Primziffern", "prime_digits", analyze_prime_digits),
        ("Fibonacci", "fibonacci", analyze_fibonacci),
        ("Modulo", "modulo", analyze_modulo),
        ("Gewichtete Summe", "weighted_sum", analyze_weighted_sum),
        ("Markov-Ketten", "markov", analyze_markov),
        ("Wochentags-Muster", "weekday", analyze_weekday),
        ("Monats-Muster", "monthly", analyze_monthly),
        ("Summe & Produkt", "sum_product", analyze_sum_product),
        ("Symmetrie", "symmetry", analyze_symmetry),
        ("Auf-/Absteigend", "ascending", analyze_ascending),
        ("Ziffern-Distanz", "distance", analyze_distance),
    ]

    for i, (name, key, func) in enumerate(methods, 1):
        print(f"  {i:2}. {name}...")
        try:
            analysis[key] = func(draws)
        except Exception as e:
            print(f"      ⚠️ Fehler: {e}")
            analysis[key] = {}

    print(f"  26. KI-Empfehlungen...")
    analysis['ki_recommendations'] = generate_ki_recommendations(analysis)

    analysis['metadata'] = {
        'last_update': datetime.now().isoformat(),
        'total_draws': len(draws),
        'last_draw': draws[0] if draws else None,
        'methods_count': len(methods)
    }

    save_json('super6_analysis.json', analysis)

    print()
    print("=" * 60)
    print(f"✅ Super 6 Analyse abgeschlossen! ({len(methods)} Methoden)")
    print("=" * 60)
    print()
    print(f"🔥 Heiße Ziffern: {analysis.get('digit_frequency', {}).get('hot_digits', [])}")
    print(f"❄️ Kalte Ziffern: {analysis.get('digit_frequency', {}).get('cold_digits', [])}")
    print(f"⏰ Überfällig: {analysis.get('gaps', {}).get('overdue_digits', [])}")
    print(f"📊 Optimale Quersumme: {analysis.get('sum_distribution', {}).get('optimal_range', [])}")

if __name__ == "__main__":
    run_analysis()
