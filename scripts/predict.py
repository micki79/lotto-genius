#!/usr/bin/env python3
"""
🍀 LottoGenius - VOLLSTÄNDIGES Multi-KI Vorhersage-System

Integriert 7 kostenlose KI-APIs:
1. Google Gemini (1M Tokens/Tag)
2. Groq (ultraschnell)
3. HuggingFace (tausende Modelle)
4. OpenRouter (50+ Modelle)
5. Together AI ($25 Startguthaben)
6. DeepSeek (komplett kostenlos)
7. Lokale ML-Algorithmen (Neuronales Netz, LSTM, etc.)

Plus: 6-Faktoren Superzahl-Analyse & Kontinuierliches Lernen
"""
import json
import os
from datetime import datetime, timedelta
import random
from collections import Counter
import requests
import time
import hashlib

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

# =====================================================
# 7 KOSTENLOSE KI-PROVIDER KONFIGURATION
# =====================================================

KI_PROVIDERS = {
    'gemini': {
        'name': 'Google Gemini',
        'emoji': '🔮',
        'url': 'https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent',
        'free_tier': '1M Tokens/Tag',
        'env_key': 'GEMINI_API_KEY'
    },
    'groq': {
        'name': 'Groq (Llama)',
        'emoji': '⚡',
        'url': 'https://api.groq.com/openai/v1/chat/completions',
        'free_tier': '1000 Requests/Tag',
        'env_key': 'GROQ_API_KEY'
    },
    'huggingface': {
        'name': 'HuggingFace',
        'emoji': '🤗',
        'url': 'https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1',
        'free_tier': 'Kostenloser Tier',
        'env_key': 'HUGGINGFACE_API_KEY'
    },
    'openrouter': {
        'name': 'OpenRouter',
        'emoji': '🌐',
        'url': 'https://openrouter.ai/api/v1/chat/completions',
        'free_tier': '50+ Modelle',
        'env_key': 'OPENROUTER_API_KEY'
    },
    'together': {
        'name': 'Together AI',
        'emoji': '🚀',
        'url': 'https://api.together.xyz/v1/chat/completions',
        'free_tier': '$25 Startguthaben',
        'env_key': 'TOGETHER_API_KEY'
    },
    'deepseek': {
        'name': 'DeepSeek',
        'emoji': '🧠',
        'url': 'https://api.deepseek.com/v1/chat/completions',
        'free_tier': 'Komplett kostenlos',
        'env_key': 'DEEPSEEK_API_KEY'
    },
    'local_ml': {
        'name': 'Lokale ML-Modelle',
        'emoji': '🖥️',
        'url': None,
        'free_tier': 'Immer verfügbar',
        'env_key': None
    }
}

# =====================================================
# HILFSFUNKTIONEN
# =====================================================

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

def get_api_key(provider):
    """Holt API-Key aus Environment Variable oder secrets.json"""
    env_key = KI_PROVIDERS.get(provider, {}).get('env_key')
    if env_key:
        key = os.environ.get(env_key)
        if key:
            return key
    
    # Fallback: secrets.json
    secrets = load_json('secrets.json', {})
    return secrets.get(provider)

# =====================================================
# SUPERZAHL-ANALYSE (6-FAKTOREN-ALGORITHMUS)
# =====================================================

class SuperzahlAnalyzer:
    """
    Analysiert Superzahl-Muster mit 6 verschiedenen Faktoren:
    1. Häufigkeit (20%) - Wie oft wurde jede SZ gezogen?
    2. Trend (25%) - Ist sie aktuell "heiß" oder "kalt"?
    3. Wochentag (15%) - Unterschiede Mittwoch vs Samstag
    4. Lücke (20%) - Wie lange nicht gezogen (überfällig)?
    5. Folge-Muster (15%) - Welche SZ kommt nach welcher?
    6. Anti-Serie (5%) - Vermeidet Wiederholungen
    """
    
    def __init__(self, draws):
        self.draws = draws
        self.patterns = {}
        if draws:
            self.analyze_all_patterns()
    
    def analyze_all_patterns(self):
        """Analysiert alle 6 Faktoren"""
        # 1. Häufigkeit
        sz_freq = Counter(d['superzahl'] for d in self.draws)
        self.patterns['frequency'] = dict(sz_freq)
        
        # 2. Lücken
        sz_gaps = {}
        for sz in range(10):
            for i, d in enumerate(self.draws):
                if d['superzahl'] == sz:
                    sz_gaps[sz] = i
                    break
            else:
                sz_gaps[sz] = len(self.draws)
        self.patterns['gaps'] = sz_gaps
        
        # 3. Trend (letzte 20 vs vorherige 20)
        recent = self.draws[:20]
        older = self.draws[20:40]
        recent_freq = Counter(d['superzahl'] for d in recent)
        older_freq = Counter(d['superzahl'] for d in older)
        
        trends = {}
        for sz in range(10):
            r = recent_freq.get(sz, 0)
            o = older_freq.get(sz, 0)
            trends[sz] = r - o
        self.patterns['trends'] = trends
        
        # 4. Wochentag-Muster
        wed_freq = Counter()
        sat_freq = Counter()
        for d in self.draws[:100]:
            try:
                day, month, year = map(int, d['date'].split('.'))
                date_obj = datetime(year, month, day)
                if date_obj.weekday() == 2:  # Mittwoch
                    wed_freq[d['superzahl']] += 1
                elif date_obj.weekday() == 5:  # Samstag
                    sat_freq[d['superzahl']] += 1
            except:
                pass
        self.patterns['wednesday'] = dict(wed_freq)
        self.patterns['saturday'] = dict(sat_freq)
        
        # 5. Folge-Muster
        follows = Counter()
        for i in range(len(self.draws) - 1):
            current = self.draws[i]['superzahl']
            previous = self.draws[i + 1]['superzahl']
            follows[(previous, current)] += 1
        self.patterns['follows'] = {f"{k[0]}->{k[1]}": v for k, v in follows.most_common(30)}
        
        # 6. Letzte Superzahl (für Anti-Serie)
        self.patterns['last_sz'] = self.draws[0]['superzahl'] if self.draws else None
    
    def predict_best_superzahl(self):
        """Berechnet die beste Superzahl basierend auf allen 6 Faktoren"""
        if not self.patterns:
            return random.randint(0, 9), [(i, 10) for i in range(10)]
        
        scores = {}
        
        freq = self.patterns.get('frequency', {})
        gaps = self.patterns.get('gaps', {})
        trends = self.patterns.get('trends', {})
        last_sz = self.patterns.get('last_sz')
        
        # Normalisierung
        max_freq = max(freq.values()) if freq else 1
        max_gap = max(gaps.values()) if gaps else 1
        max_trend = max(abs(v) for v in trends.values()) if trends and any(trends.values()) else 1
        
        # Bestimme Wochentag für die nächste Ziehung
        today = datetime.now()
        days_to_wed = (2 - today.weekday() + 7) % 7
        days_to_sat = (5 - today.weekday() + 7) % 7
        is_wednesday = days_to_wed < days_to_sat or (days_to_wed == 0 and today.hour < 19)
        
        day_freq = self.patterns.get('wednesday' if is_wednesday else 'saturday', {})
        max_day = max(day_freq.values()) if day_freq else 1
        
        for sz in range(10):
            score = 0
            
            # 1. Häufigkeit (20%) - Häufige bevorzugen
            score += (freq.get(sz, freq.get(str(sz), 0)) / max_freq) * 20
            
            # 2. Trend (25%) - Steigende bevorzugen
            trend_val = trends.get(sz, 0)
            normalized_trend = (trend_val + max_trend) / (2 * max_trend) if max_trend > 0 else 0.5
            score += normalized_trend * 25
            
            # 3. Wochentag (15%)
            score += (day_freq.get(sz, day_freq.get(str(sz), 0)) / max_day) * 15
            
            # 4. Lücke (20%) - Überfällige bevorzugen
            score += (gaps.get(sz, gaps.get(str(sz), 0)) / max_gap) * 20
            
            # 5. Folge-Muster (15%)
            if last_sz is not None:
                follow_key = f"{last_sz}->{sz}"
                follow_count = 0
                for k, v in self.patterns.get('follows', {}).items():
                    if k == follow_key:
                        follow_count = v
                        break
                max_follow = max(self.patterns.get('follows', {}).values()) if self.patterns.get('follows') else 1
                score += (follow_count / max_follow) * 15
            
            # 6. Anti-Serie (5%) - Vermeidet Wiederholung
            if sz != last_sz:
                score += 5
            
            # Kleiner Zufallsfaktor für Diversität
            score += random.random() * 3
            
            scores[sz] = round(score, 2)
        
        # Sortiere und gib Ranking zurück
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_sz = ranked[0][0]
        
        return best_sz, ranked

# =====================================================
# KI-API AUFRUFE
# =====================================================

def call_gemini_api(prompt, api_key):
    """Ruft Google Gemini API auf"""
    if not api_key:
        return None
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={api_key}"
    
    try:
        response = requests.post(url, json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 1000}
        }, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            return data.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text')
    except Exception as e:
        print(f"  ❌ Gemini Fehler: {e}")
    
    return None

def call_groq_api(prompt, api_key):
    """Ruft Groq API auf (ultraschnell)"""
    if not api_key:
        return None
    
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "llama-3.1-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 1000
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"  ❌ Groq Fehler: {e}")
    
    return None

def call_huggingface_api(prompt, api_key):
    """Ruft HuggingFace Inference API auf"""
    if not api_key:
        return None
    
    try:
        response = requests.post(
            "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"inputs": prompt, "parameters": {"max_new_tokens": 500}},
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get('generated_text', '')
    except Exception as e:
        print(f"  ❌ HuggingFace Fehler: {e}")
    
    return None

def call_openrouter_api(prompt, api_key):
    """Ruft OpenRouter API auf"""
    if not api_key:
        return None
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://lotto-genius.github.io",
                "X-Title": "LottoGenius"
            },
            json={
                "model": "meta-llama/llama-3.1-8b-instruct:free",
                "messages": [{"role": "user", "content": prompt}]
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"  ❌ OpenRouter Fehler: {e}")
    
    return None

def call_together_api(prompt, api_key):
    """Ruft Together AI API auf"""
    if not api_key:
        return None
    
    try:
        response = requests.post(
            "https://api.together.xyz/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "meta-llama/Llama-3-70b-chat-hf",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1000
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"  ❌ Together Fehler: {e}")
    
    return None

def call_deepseek_api(prompt, api_key):
    """Ruft DeepSeek API auf"""
    if not api_key:
        return None
    
    try:
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1000
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"  ❌ DeepSeek Fehler: {e}")
    
    return None

# =====================================================
# LOKALE ML-MODELLE (immer verfügbar)
# =====================================================

class LocalMLModels:
    """
    Implementiert verschiedene ML-Algorithmen lokal:
    - Neuronales Netz (simuliert)
    - LSTM-ähnliche Sequenzanalyse
    - Random Forest (simuliert)
    - Bayesian Inference
    - Monte-Carlo Simulation
    """
    
    def __init__(self, draws, analysis):
        self.draws = draws
        self.analysis = analysis
        self.freq = analysis.get('frequency', {})
        self.gaps = analysis.get('gaps', {})
    
    def get_hot_numbers(self, n=15):
        """Heißeste Zahlen"""
        hot = self.freq.get('hot_numbers', [])
        if not hot:
            all_nums = []
            for d in self.draws[:50]:
                all_nums.extend(d['numbers'])
            hot = [n for n, _ in Counter(all_nums).most_common(n)]
        return hot[:n]
    
    def get_cold_numbers(self, n=10):
        """Kälteste Zahlen"""
        cold = self.freq.get('cold_numbers', [])
        if not cold:
            all_nums = []
            for d in self.draws[:200]:
                all_nums.extend(d['numbers'])
            freq = Counter(all_nums)
            cold = [n for n, _ in freq.most_common()[-n:]]
        return cold[:n]
    
    def get_overdue_numbers(self, n=10):
        """Überfällige Zahlen"""
        overdue = self.gaps.get('overdue_numbers', [])
        if not overdue:
            gaps = {}
            for num in range(1, 50):
                for i, d in enumerate(self.draws):
                    if num in d['numbers']:
                        gaps[num] = i
                        break
                else:
                    gaps[num] = len(self.draws)
            overdue = [n for n, _ in sorted(gaps.items(), key=lambda x: x[1], reverse=True)[:n]]
        return overdue[:n]
    
    def neural_network(self):
        """Simuliertes Neuronales Netz"""
        hot = self.get_hot_numbers(20)
        cold = self.get_cold_numbers(10)
        overdue = self.get_overdue_numbers(10)
        
        weights = {}
        for n in range(1, 50):
            w = 0
            if n in hot[:10]: w += 3.0
            elif n in hot[10:]: w += 1.5
            if n in cold: w += 0.5
            if n in overdue[:5]: w += 2.5
            elif n in overdue[5:]: w += 1.0
            # Sigmoid-ähnliche Aktivierung
            w = w / (1 + abs(w - 2))
            weights[n] = w + random.random() * 0.5
        
        sorted_nums = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        return [n for n, _ in sorted_nums[:6]]
    
    def lstm_sequence(self):
        """LSTM-ähnliche Sequenzanalyse"""
        # Analysiere Sequenzen der letzten Ziehungen
        sequences = []
        for i in range(min(10, len(self.draws))):
            sequences.extend(self.draws[i]['numbers'])
        
        # Finde häufige Folgemuster
        seq_freq = Counter()
        for i in range(len(sequences) - 1):
            seq_freq[(sequences[i], sequences[i+1])] += 1
        
        predicted = set()
        for (a, b), _ in seq_freq.most_common(20):
            predicted.add(b)
            if len(predicted) >= 6:
                break
        
        result = list(predicted)[:6]
        while len(result) < 6:
            n = random.choice(self.get_hot_numbers(20))
            if n not in result:
                result.append(n)
        
        return sorted(result)
    
    def random_forest(self):
        """Simulierter Random Forest"""
        # Mehrere "Entscheidungsbäume"
        trees = []
        
        # Baum 1: Nur heiße Zahlen
        trees.append(random.sample(self.get_hot_numbers(15), 6))
        
        # Baum 2: Mix heiß + kalt
        hot = self.get_hot_numbers(10)
        cold = self.get_cold_numbers(6)
        trees.append(random.sample(hot, 4) + random.sample(cold, 2))
        
        # Baum 3: Überfällige + heiß
        overdue = self.get_overdue_numbers(8)
        trees.append(random.sample(overdue, 3) + random.sample(hot, 3))
        
        # Voting
        votes = Counter()
        for tree in trees:
            votes.update(tree)
        
        return sorted([n for n, _ in votes.most_common(6)])
    
    def bayesian_inference(self):
        """Bayesian Wahrscheinlichkeitsberechnung"""
        # Prior: Gleichverteilung
        prior = {n: 1/49 for n in range(1, 50)}
        
        # Likelihood basierend auf Häufigkeit
        all_nums = []
        for d in self.draws[:100]:
            all_nums.extend(d['numbers'])
        freq = Counter(all_nums)
        total = sum(freq.values())
        
        likelihood = {n: freq.get(n, 1) / total for n in range(1, 50)}
        
        # Posterior = Prior * Likelihood
        posterior = {n: prior[n] * likelihood[n] for n in range(1, 50)}
        
        # Normalisieren
        total_post = sum(posterior.values())
        posterior = {n: p / total_post for n, p in posterior.items()}
        
        # Top 6
        sorted_post = sorted(posterior.items(), key=lambda x: x[1], reverse=True)
        return sorted([n for n, _ in sorted_post[:6]])
    
    def monte_carlo(self, simulations=1000):
        """Monte-Carlo Simulation"""
        hot = self.get_hot_numbers(25)
        results = Counter()
        
        for _ in range(simulations):
            sample = random.choices(hot if hot else list(range(1, 50)), k=6)
            sample = list(set(sample))
            while len(sample) < 6:
                sample.append(random.randint(1, 49))
            for n in sample[:6]:
                results[n] += 1
        
        return sorted([n for n, _ in results.most_common(6)])
    
    def ensemble_all(self):
        """Kombiniert alle ML-Modelle"""
        all_preds = [
            self.neural_network(),
            self.lstm_sequence(),
            self.random_forest(),
            self.bayesian_inference(),
            self.monte_carlo()
        ]
        
        votes = Counter()
        for pred in all_preds:
            votes.update(pred)
        
        return sorted([n for n, _ in votes.most_common(6)])

# =====================================================
# HAUPTVORHERSAGE-FUNKTION
# =====================================================

def generate_predictions():
    """Hauptfunktion: Generiert Vorhersagen mit allen KI-Systemen"""
    
    print("=" * 60)
    print("🍀 LottoGenius - VOLLSTÄNDIGES Multi-KI System")
    print("=" * 60)
    print(f"📅 {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}")
    print()
    
    # Lade Daten
    lotto_data = load_json('lotto_data.json', {'draws': []})
    analysis = load_json('analysis.json', {})
    predictions = load_json('predictions.json', {'predictions': [], 'history': []})
    methods = load_json('methods.json', {'methods': {}})
    provider_scores = load_json('provider_scores.json', {})
    
    draws = lotto_data.get('draws', [])
    if not draws:
        print("⚠️ Keine Lotto-Daten vorhanden!")
        return
    
    print(f"📊 Analysiere {len(draws)} historische Ziehungen...")
    
    # Archiviere alte Vorhersagen
    if predictions.get('predictions'):
        predictions['history'].extend(predictions['predictions'])
        predictions['history'] = predictions['history'][-500:]
    
    # Initialisiere Superzahl-Analyzer
    sz_analyzer = SuperzahlAnalyzer(draws)
    best_sz, sz_ranking = sz_analyzer.predict_best_superzahl()
    
    print(f"\n🎯 Superzahl-Analyse (6-Faktoren):")
    print(f"   Beste Superzahl: {best_sz}")
    print(f"   Top 3: {sz_ranking[:3]}")
    
    # Initialisiere ML-Modelle
    ml_models = LocalMLModels(draws, analysis)
    
    # Erstelle KI-Prompt
    hot_numbers = ml_models.get_hot_numbers(10)
    cold_numbers = ml_models.get_cold_numbers(10)
    overdue = ml_models.get_overdue_numbers(10)
    last_draw = draws[0]
    
    ki_prompt = f"""Du bist ein Lotto-Analyse-Experte. Analysiere diese Daten für Lotto 6 aus 49:

LETZTE ZIEHUNG: {last_draw['date']} - Zahlen: {last_draw['numbers']}, Superzahl: {last_draw['superzahl']}

STATISTIK (basierend auf {len(draws)} Ziehungen):
- Heiße Zahlen (häufig): {hot_numbers}
- Kalte Zahlen (selten): {cold_numbers}
- Überfällige Zahlen: {overdue}
- Beste Superzahl laut Analyse: {best_sz}

Generiere 2 verschiedene Tipps. Antworte NUR mit diesem JSON-Format:
{{
  "predictions": [
    {{"numbers": [1,2,3,4,5,6], "superzahl": 0, "strategy": "Beschreibung", "confidence": 75}},
    {{"numbers": [7,8,9,10,11,12], "superzahl": 1, "strategy": "Beschreibung", "confidence": 70}}
  ]
}}"""
    
    new_predictions = []
    ki_results = {}
    
    # ===== 1. LOKALE ML-MODELLE (immer verfügbar) =====
    print("\n🖥️ Lokale ML-Modelle:")
    
    local_methods = [
        ('neural_network', '🧠 Neuronales Netz', ml_models.neural_network),
        ('lstm', '📈 LSTM Sequenz', ml_models.lstm_sequence),
        ('random_forest', '🌲 Random Forest', ml_models.random_forest),
        ('bayesian', '📊 Bayesian', ml_models.bayesian_inference),
        ('monte_carlo', '🎲 Monte-Carlo', ml_models.monte_carlo),
        ('ensemble_ml', '🏆 ML Ensemble', ml_models.ensemble_all)
    ]
    
    for method_id, method_name, method_fn in local_methods:
        try:
            nums = method_fn()
            sz_idx = local_methods.index((method_id, method_name, method_fn)) % len(sz_ranking)
            sz = sz_ranking[sz_idx][0]
            
            new_predictions.append({
                'numbers': sorted(nums)[:6],
                'superzahl': sz,
                'method': method_id,
                'method_name': method_name,
                'provider': 'local_ml',
                'confidence': 60 + random.random() * 30,
                'timestamp': datetime.now().isoformat(),
                'verified': False
            })
            print(f"   ✅ {method_name}: {sorted(nums)[:6]} | SZ: {sz}")
            ki_results['local_ml'] = True
        except Exception as e:
            print(f"   ❌ {method_name}: {e}")
    
    # ===== 2. EXTERNE KI-APIS =====
    print("\n🌐 Externe KI-APIs:")
    
    ki_apis = [
        ('gemini', '🔮 Google Gemini', call_gemini_api),
        ('groq', '⚡ Groq', call_groq_api),
        ('huggingface', '🤗 HuggingFace', call_huggingface_api),
        ('openrouter', '🌐 OpenRouter', call_openrouter_api),
        ('together', '🚀 Together AI', call_together_api),
        ('deepseek', '🧠 DeepSeek', call_deepseek_api)
    ]
    
    for provider_id, provider_name, api_fn in ki_apis:
        api_key = get_api_key(provider_id)
        
        if not api_key:
            print(f"   ⏭️ {provider_name}: Kein API-Key")
            continue
        
        try:
            result = api_fn(ki_prompt, api_key)
            
            if result:
                # Parse JSON aus Antwort
                json_match = None
                try:
                    import re
                    json_match = re.search(r'\{[\s\S]*\}', result)
                    if json_match:
                        parsed = json.loads(json_match.group())
                        if 'predictions' in parsed:
                            for pred in parsed['predictions']:
                                nums = pred.get('numbers', [])[:6]
                                if len(nums) == 6 and all(1 <= n <= 49 for n in nums):
                                    new_predictions.append({
                                        'numbers': sorted(nums),
                                        'superzahl': pred.get('superzahl', best_sz) % 10,
                                        'method': f'{provider_id}_ki',
                                        'method_name': f'{provider_name} KI',
                                        'provider': provider_id,
                                        'strategy': pred.get('strategy', ''),
                                        'confidence': pred.get('confidence', 70),
                                        'timestamp': datetime.now().isoformat(),
                                        'verified': False
                                    })
                            print(f"   ✅ {provider_name}: {len(parsed['predictions'])} Vorhersagen")
                            ki_results[provider_id] = True
                            continue
                except:
                    pass
                
                print(f"   ⚠️ {provider_name}: Konnte Antwort nicht parsen")
            else:
                print(f"   ❌ {provider_name}: Keine Antwort")
                
        except Exception as e:
            print(f"   ❌ {provider_name}: {e}")
    
    # ===== 3. ENSEMBLE-VOTING =====
    print("\n🏆 Ensemble-Voting:")
    
    if len(new_predictions) >= 3:
        all_numbers = Counter()
        all_superzahlen = Counter()
        
        for pred in new_predictions:
            all_numbers.update(pred['numbers'])
            all_superzahlen[pred['superzahl']] += 1
        
        # Champion-Tipp: Zahlen mit meisten Stimmen
        top_voted = [n for n, _ in all_numbers.most_common(6)]
        top_sz = all_superzahlen.most_common(1)[0][0]
        
        new_predictions.insert(0, {
            'numbers': sorted(top_voted),
            'superzahl': top_sz,
            'method': 'ensemble_champion',
            'method_name': '🏆 CHAMPION (Alle KIs)',
            'provider': 'ensemble',
            'strategy': f'Voting aus {len(new_predictions)} KI-Vorhersagen',
            'confidence': 90,
            'timestamp': datetime.now().isoformat(),
            'verified': False,
            'is_champion': True
        })
        
        print(f"   ✅ Champion-Tipp: {sorted(top_voted)} | SZ: {top_sz}")
        print(f"   📊 Basiert auf {len(new_predictions)-1} KI-Vorhersagen")
    
    # ===== SPEICHERN =====
    
    # Berechne nächste Ziehung
    now = datetime.now()
    days_to_wed = (2 - now.weekday() + 7) % 7
    days_to_sat = (5 - now.weekday() + 7) % 7
    if days_to_wed == 0 and now.hour >= 19:
        days_to_wed = 7
    if days_to_sat == 0 and now.hour >= 20:
        days_to_sat = 7
    next_days = min(days_to_wed if days_to_wed > 0 else 7, days_to_sat if days_to_sat > 0 else 7)
    next_draw = now + timedelta(days=next_days)
    next_draw_str = next_draw.strftime('%d.%m.%Y')
    
    predictions['predictions'] = new_predictions
    predictions['last_update'] = datetime.now().isoformat()
    predictions['next_draw'] = next_draw_str
    predictions['ki_stats'] = {
        'providers_used': list(ki_results.keys()),
        'total_predictions': len(new_predictions),
        'best_superzahl': best_sz,
        'superzahl_ranking': [(sz, round(score, 1)) for sz, score in sz_ranking[:5]],
        'superzahl_patterns': sz_analyzer.patterns
    }
    
    save_json('predictions.json', predictions)
    
    print("\n" + "=" * 60)
    print(f"✅ {len(new_predictions)} Vorhersagen generiert!")
    print(f"📅 Nächste Ziehung: {next_draw_str}")
    print(f"🎯 Beste Superzahl: {best_sz}")
    print(f"🤖 KI-Provider verwendet: {len(ki_results)}")
    print("=" * 60)

if __name__ == "__main__":
    generate_predictions()
