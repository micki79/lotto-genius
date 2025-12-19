#!/usr/bin/env python3
"""
🌀 LottoGenius - VOLLSTÄNDIGE Glücksspirale Analyse

25+ Analyse-Methoden für 7-stellige Zahlen:
- Ziffern-Häufigkeit & Positions-Analyse
- Muster-Erkennung (Doppel, Sequenzen, Palindrome)
- Mathematische Analyse (Quersumme, Primziffern)
- ML-Algorithmen (Markov, Monte-Carlo, Bayesian)
- Zeitliche Muster (Monat, Jahreszeit, Trends)

Besonderheit: Glücksspirale läuft nur Samstags!
"""
import json
import os
from datetime import datetime
from collections import Counter, defaultdict
import random

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
NUM_DIGITS = 7  # Glücksspirale hat 7 Ziffern

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

    return {
        'frequency': {str(d): freq.get(d, 0) for d in range(10)},
        'percentage': {str(d): round(freq.get(d, 0) / total * 100, 2) for d in range(10)},
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
            'hot_digit': freq.most_common(1)[0][0] if freq else 0
        }
    return result

def analyze_sum_distribution(draws):
    """3. Quersummen-Verteilung"""
    sums = [sum(get_digits(draw['number'])) for draw in draws]
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
    """4. Gerade/Ungerade"""
    total_odd = 0
    total_even = 0
    for draw in draws:
        for d in get_digits(draw['number']):
            if d % 2 == 0:
                total_even += 1
            else:
                total_odd += 1
    return {
        'odd_percentage': round(total_odd / (total_odd + total_even) * 100, 2) if (total_odd + total_even) > 0 else 50
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
        'most_common_deltas': [d for d, _ in freq.most_common(5)]
    }

def analyze_hot_cold_trend(draws):
    """6. Hot/Cold Trend"""
    recent = draws[:15]
    older = draws[15:40]

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
        r = recent_freq.get(d, 0) / max(len(recent), 1)
        o = older_freq.get(d, 0) / max(len(older), 1)
        trends[d] = round(r - o, 3)

    return {
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
    return {'current_gaps': current_gaps, 'overdue_digits': [d for d, _ in overdue]}

def analyze_doubles(draws):
    """8. Doppelziffern"""
    count = 0
    for draw in draws:
        number = str(draw['number']).zfill(NUM_DIGITS)
        for i in range(len(number) - 1):
            if number[i] == number[i+1]:
                count += 1
                break
    return {'percentage': round(count / len(draws) * 100, 2) if draws else 0}

def analyze_triples(draws):
    """9. Drillinge"""
    count = 0
    for draw in draws:
        number = str(draw['number']).zfill(NUM_DIGITS)
        for i in range(len(number) - 2):
            if number[i] == number[i+1] == number[i+2]:
                count += 1
                break
    return {'percentage': round(count / len(draws) * 100, 2) if draws else 0}

def analyze_sequences(draws):
    """10. Sequenzen"""
    total = 0
    for draw in draws:
        digits = get_digits(draw['number'])
        for i in range(len(digits) - 2):
            if digits[i+1] == digits[i] + 1 and digits[i+2] == digits[i] + 2:
                total += 1
            elif digits[i+1] == digits[i] - 1 and digits[i+2] == digits[i] - 2:
                total += 1
    return {'total_sequences': total}

def analyze_palindromes(draws):
    """11. Palindrome"""
    count = sum(1 for draw in draws if str(draw['number']).zfill(NUM_DIGITS) == str(draw['number']).zfill(NUM_DIGITS)[::-1])
    return {'palindromes': count}

def analyze_repetitions(draws):
    """12. Wiederholungen"""
    dist = Counter(len(set(get_digits(draw['number']))) for draw in draws)
    return {'distribution': {str(k): v for k, v in sorted(dist.items())}}

def analyze_first_last(draws):
    """13. Erste/Letzte Korrelation"""
    pairs = Counter((get_digits(draw['number'])[0], get_digits(draw['number'])[-1]) for draw in draws)
    return {'most_common': [(f"{p[0]}-{p[1]}", c) for p, c in pairs.most_common(10)]}

def analyze_neighbor_pairs(draws):
    """14. Nachbar-Paare"""
    pairs = Counter()
    for draw in draws:
        digits = get_digits(draw['number'])
        for i in range(len(digits) - 1):
            pairs[tuple(sorted([digits[i], digits[i+1]]))] += 1
    return {'most_common': [(f"{p[0]}-{p[1]}", c) for p, c in pairs.most_common(15)]}

def analyze_prime_digits(draws):
    """15. Primziffern"""
    primes = {2, 3, 5, 7}
    counts = [sum(1 for d in get_digits(draw['number']) if d in primes) for draw in draws]
    return {'average': round(sum(counts) / len(counts), 2) if counts else 0}

def analyze_fibonacci(draws):
    """16. Fibonacci"""
    fib = {1, 2, 3, 5, 8}
    counts = [sum(1 for d in get_digits(draw['number']) if d in fib) for draw in draws]
    return {'average': round(sum(counts) / len(counts), 2) if counts else 0}

def analyze_modulo(draws):
    """17. Modulo"""
    mod3 = [sum(d % 3 for d in get_digits(draw['number'])) for draw in draws]
    return {'mod3_average': round(sum(mod3) / len(mod3), 2) if mod3 else 0}

def analyze_weighted_sum(draws):
    """18. Gewichtete Summe"""
    sums = [sum((i + 1) * d for i, d in enumerate(get_digits(draw['number']))) for draw in draws]
    return {'average': round(sum(sums) / len(sums), 2) if sums else 0}

def analyze_markov(draws):
    """19. Markov-Ketten"""
    trans = defaultdict(Counter)
    for draw in draws:
        digits = get_digits(draw['number'])
        for i in range(len(digits) - 1):
            trans[digits[i]][digits[i+1]] += 1
    probs = {}
    for from_d, to_counts in trans.items():
        total = sum(to_counts.values())
        probs[from_d] = {to_d: round(c / total, 3) for to_d, c in to_counts.most_common(3)}
    return {'transition_probabilities': probs}

def analyze_monthly(draws):
    """20. Monats-Muster"""
    month_freq = defaultdict(Counter)
    for draw in draws:
        try:
            day, month, year = map(int, draw['date'].split('.'))
            digits = get_digits(draw['number'])
            month_freq[month].update(digits)
        except:
            pass
    return {str(m): {'hot': [d for d, _ in f.most_common(3)]} for m, f in month_freq.items()}

def analyze_quarterly(draws):
    """21. Quartals-Muster"""
    quarter_freq = defaultdict(Counter)
    for draw in draws:
        try:
            day, month, year = map(int, draw['date'].split('.'))
            quarter = (month - 1) // 3 + 1
            digits = get_digits(draw['number'])
            quarter_freq[quarter].update(digits)
        except:
            pass
    return {f'Q{q}': {'hot': [d for d, _ in f.most_common(3)]} for q, f in quarter_freq.items()}

def analyze_sum_product(draws):
    """22. Summe & Produkt"""
    sums = [sum(get_digits(draw['number'])) for draw in draws]
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
        max_asc = current = 1
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
    distances = [sum(abs(d - 4.5) for d in get_digits(draw['number'])) for draw in draws]
    return {'average_distance': round(sum(distances) / len(distances), 2) if distances else 0}

def generate_ki_recommendations(analysis):
    """KI-Empfehlungen"""
    return {
        'hot_digits': analysis.get('digit_frequency', {}).get('hot_digits', []),
        'cold_digits': analysis.get('digit_frequency', {}).get('cold_digits', []),
        'overdue_digits': analysis.get('gaps', {}).get('overdue_digits', []),
        'rising_digits': analysis.get('hot_cold_trend', {}).get('rising_digits', []),
        'optimal_sum_range': analysis.get('sum_distribution', {}).get('optimal_range', [25, 40])
    }

# =====================================================
# HAUPTFUNKTION
# =====================================================

def run_analysis():
    print("=" * 60)
    print("🌀 LottoGenius - VOLLSTÄNDIGE Glücksspirale Analyse")
    print("=" * 60)
    print(f"📅 {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
    print()

    data = load_json('gluecksspirale_data.json', {'draws': []})
    draws = data.get('draws', [])

    if not draws:
        print("⚠️ Keine Glücksspirale Daten vorhanden!")
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
        ("Monats-Muster", "monthly", analyze_monthly),
        ("Quartals-Muster", "quarterly", analyze_quarterly),
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
            print(f"      ⚠️ {e}")
            analysis[key] = {}

    print(f"  26. KI-Empfehlungen...")
    analysis['ki_recommendations'] = generate_ki_recommendations(analysis)

    analysis['metadata'] = {
        'last_update': datetime.now().isoformat(),
        'total_draws': len(draws),
        'last_draw': draws[0] if draws else None,
        'game': 'Glücksspirale',
        'draw_day': 'Samstag'
    }

    save_json('gluecksspirale_analysis.json', analysis)

    print()
    print("=" * 60)
    print(f"✅ Glücksspirale Analyse abgeschlossen! ({len(methods)} Methoden)")
    print("=" * 60)
    print()
    print(f"🔥 Heiße Ziffern: {analysis.get('digit_frequency', {}).get('hot_digits', [])}")
    print(f"❄️ Kalte Ziffern: {analysis.get('digit_frequency', {}).get('cold_digits', [])}")
    print(f"⏰ Überfällig: {analysis.get('gaps', {}).get('overdue_digits', [])}")
    print(f"📊 Optimale Quersumme: {analysis.get('sum_distribution', {}).get('optimal_range', [])}")

if __name__ == "__main__":
    run_analysis()
