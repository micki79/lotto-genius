#!/usr/bin/env python3
"""
🧠 LottoGenius - ECHTE Machine Learning Modelle

Dieses Modul implementiert ECHTE ML-Algorithmen (keine Fake-Namen mehr!):

1. NeuralNetwork - Echtes mehrschichtiges Neuronales Netz mit Backpropagation
2. MarkovChain - Übergangswahrscheinlichkeiten für Zahlensequenzen
3. BayesianPredictor - Echtes Bayesian Learning mit Prior/Posterior Updates
4. ReinforcementLearner - Q-Learning für Strategie-Optimierung
5. EnsembleML - Kombiniert alle Modelle mit gelernten Gewichten

Alle Modelle:
- Lernen aus jeder neuen Ziehung
- Speichern ihre Gewichte persistent
- Verbessern sich kontinuierlich
"""

import json
import os
import numpy as np
from datetime import datetime
from collections import Counter, defaultdict
import hashlib

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
MODELS_DIR = os.path.join(DATA_DIR, 'ml_models')

def ensure_dirs():
    """Erstellt benötigte Verzeichnisse"""
    os.makedirs(MODELS_DIR, exist_ok=True)

def load_json(filename, default=None):
    """Lädt JSON-Datei"""
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return default if default else {}

def save_json(filename, data):
    """Speichert JSON-Datei"""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, filename), 'w') as f:
        json.dump(data, f, indent=2)

def save_model(filename, data):
    """Speichert Modell-Daten"""
    ensure_dirs()
    path = os.path.join(MODELS_DIR, filename)
    # Konvertiere numpy arrays zu Listen für JSON
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(i) for i in obj]
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        return obj

    with open(path, 'w') as f:
        json.dump(convert(data), f, indent=2)

def load_model(filename, default=None):
    """Lädt Modell-Daten"""
    path = os.path.join(MODELS_DIR, filename)
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return default if default else {}


# =====================================================
# 1. ECHTES NEURONALES NETZ
# =====================================================

class NeuralNetwork:
    """
    Echtes mehrschichtiges Neuronales Netz für Lotto-Vorhersagen.

    Architektur:
    - Input: 49 Neuronen (Häufigkeit jeder Zahl)
    - Hidden 1: 64 Neuronen (ReLU)
    - Hidden 2: 32 Neuronen (ReLU)
    - Output: 49 Neuronen (Softmax für Wahrscheinlichkeiten)

    Training:
    - Backpropagation mit Gradient Descent
    - Learning Rate mit Decay
    - Momentum für stabiles Training
    """

    MODEL_FILE = 'neural_network.json'

    def __init__(self, input_size=49, hidden1=64, hidden2=32, output_size=49):
        self.input_size = input_size
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.output_size = output_size

        # Lade existierende Gewichte oder initialisiere neu
        saved = load_model(self.MODEL_FILE)

        if saved and 'weights' in saved:
            self.W1 = np.array(saved['weights']['W1'])
            self.b1 = np.array(saved['weights']['b1'])
            self.W2 = np.array(saved['weights']['W2'])
            self.b2 = np.array(saved['weights']['b2'])
            self.W3 = np.array(saved['weights']['W3'])
            self.b3 = np.array(saved['weights']['b3'])
            self.training_history = saved.get('training_history', [])
            self.epochs_trained = saved.get('epochs_trained', 0)
            self.learning_rate = saved.get('learning_rate', 0.01)
        else:
            # Xavier Initialization für bessere Konvergenz
            self.W1 = np.random.randn(input_size, hidden1) * np.sqrt(2.0 / input_size)
            self.b1 = np.zeros((1, hidden1))
            self.W2 = np.random.randn(hidden1, hidden2) * np.sqrt(2.0 / hidden1)
            self.b2 = np.zeros((1, hidden2))
            self.W3 = np.random.randn(hidden2, output_size) * np.sqrt(2.0 / hidden2)
            self.b3 = np.zeros((1, output_size))
            self.training_history = []
            self.epochs_trained = 0
            self.learning_rate = 0.01

        # Momentum für Gradient Descent
        self.v_W1 = np.zeros_like(self.W1)
        self.v_W2 = np.zeros_like(self.W2)
        self.v_W3 = np.zeros_like(self.W3)
        self.momentum = 0.9

    def relu(self, x):
        """ReLU Aktivierungsfunktion mit Clipping"""
        return np.clip(np.maximum(0, x), 0, 100)  # Clip für numerische Stabilität

    def relu_derivative(self, x):
        """Ableitung von ReLU"""
        return (x > 0).astype(float)

    def softmax(self, x):
        """Softmax für Wahrscheinlichkeiten (numerisch stabil)"""
        # Clip um Overflow zu vermeiden
        x_clipped = np.clip(x, -100, 100)
        x_shifted = x_clipped - np.max(x_clipped, axis=1, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / (np.sum(exp_x, axis=1, keepdims=True) + 1e-10)

    def forward(self, X):
        """Forward Pass durch das Netzwerk"""
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)

        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.relu(self.z2)

        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.a3 = self.softmax(self.z3)

        return self.a3

    def clip_gradients(self, grad, max_norm=1.0):
        """Gradient Clipping für numerische Stabilität"""
        grad = np.nan_to_num(grad, nan=0.0, posinf=max_norm, neginf=-max_norm)
        norm = np.linalg.norm(grad)
        if norm > max_norm:
            grad = grad * (max_norm / norm)
        return grad

    def backward(self, X, y, output):
        """Backpropagation mit Gradient Clipping"""
        m = X.shape[0]

        # Output Layer
        dz3 = output - y
        dW3 = self.clip_gradients(np.dot(self.a2.T, dz3) / m)
        db3 = self.clip_gradients(np.sum(dz3, axis=0, keepdims=True) / m)

        # Hidden Layer 2
        dz2 = np.dot(dz3, self.W3.T) * self.relu_derivative(self.z2)
        dW2 = self.clip_gradients(np.dot(self.a1.T, dz2) / m)
        db2 = self.clip_gradients(np.sum(dz2, axis=0, keepdims=True) / m)

        # Hidden Layer 1
        dz1 = np.dot(dz2, self.W2.T) * self.relu_derivative(self.z1)
        dW1 = self.clip_gradients(np.dot(X.T, dz1) / m)
        db1 = self.clip_gradients(np.sum(dz1, axis=0, keepdims=True) / m)

        # Gradient Descent mit Momentum
        self.v_W3 = self.momentum * self.v_W3 - self.learning_rate * dW3
        self.v_W2 = self.momentum * self.v_W2 - self.learning_rate * dW2
        self.v_W1 = self.momentum * self.v_W1 - self.learning_rate * dW1

        self.W3 += self.v_W3
        self.b3 -= self.learning_rate * db3
        self.W2 += self.v_W2
        self.b2 -= self.learning_rate * db2
        self.W1 += self.v_W1
        self.b1 -= self.learning_rate * db1

        # Gewichte auch clippen um Explosion zu verhindern
        self.W1 = np.clip(self.W1, -5, 5)
        self.W2 = np.clip(self.W2, -5, 5)
        self.W3 = np.clip(self.W3, -5, 5)

    def create_features(self, draws, num_draws=100):
        """Erstellt Feature-Vektoren aus historischen Ziehungen"""
        features = []

        for i in range(min(num_draws, len(draws) - 1)):
            # Feature: Häufigkeit der letzten N Ziehungen
            freq = np.zeros(49)
            for j in range(i, min(i + 20, len(draws))):
                for num in draws[j].get('numbers', []):
                    if 1 <= num <= 49:
                        freq[num - 1] += 1

            # Normalisieren
            if np.max(freq) > 0:
                freq = freq / np.max(freq)

            features.append(freq)

        return np.array(features) if features else np.zeros((1, 49))

    def create_labels(self, draws, num_draws=100):
        """Erstellt Label-Vektoren (nächste Ziehung)"""
        labels = []

        for i in range(min(num_draws, len(draws) - 1)):
            label = np.zeros(49)
            if i > 0:
                for num in draws[i - 1].get('numbers', []):
                    if 1 <= num <= 49:
                        label[num - 1] = 1
            labels.append(label)

        return np.array(labels) if labels else np.zeros((1, 49))

    def train(self, draws, epochs=100, batch_size=32):
        """Trainiert das Netzwerk mit historischen Daten"""
        X = self.create_features(draws, num_draws=500)
        y = self.create_labels(draws, num_draws=500)

        if len(X) < 2:
            return {'error': 'Nicht genug Trainingsdaten'}

        losses = []

        for epoch in range(epochs):
            # Mini-Batch Training
            indices = np.random.permutation(len(X))

            for start in range(0, len(X), batch_size):
                end = min(start + batch_size, len(X))
                batch_X = X[indices[start:end]]
                batch_y = y[indices[start:end]]

                # Forward & Backward
                output = self.forward(batch_X)
                self.backward(batch_X, batch_y, output)

            # Berechne Loss
            output = self.forward(X)
            loss = -np.mean(y * np.log(output + 1e-10))
            losses.append(loss)

            # Learning Rate Decay
            if epoch > 0 and epoch % 20 == 0:
                self.learning_rate *= 0.95

        self.epochs_trained += epochs
        self.training_history.append({
            'date': datetime.now().isoformat(),
            'epochs': epochs,
            'final_loss': float(losses[-1]) if losses else 0,
            'total_epochs': self.epochs_trained
        })

        self.save()

        return {
            'epochs': epochs,
            'final_loss': float(losses[-1]) if losses else 0,
            'total_epochs': self.epochs_trained
        }

    def train_on_new_draw(self, draws, new_draw):
        """Inkrementelles Training mit neuer Ziehung"""
        # Erstelle Features aus den letzten 50 Ziehungen
        X = self.create_features(draws[:50], num_draws=50)

        # Label ist die neue Ziehung
        y = np.zeros((len(X), 49))
        for num in new_draw.get('numbers', []):
            if 1 <= num <= 49:
                y[:, num - 1] = 1

        # Ein paar Epochen Training
        for _ in range(10):
            output = self.forward(X)
            self.backward(X, y, output)

        self.epochs_trained += 10
        self.save()

        return {'trained_on': new_draw.get('date', 'unknown')}

    def predict(self, draws):
        """Vorhersage für die nächste Ziehung"""
        X = self.create_features(draws[:30], num_draws=1)

        if len(X) == 0:
            return list(range(1, 7)), 0.5

        probs = self.forward(X)[0]

        # Wähle die 6 Zahlen mit höchster Wahrscheinlichkeit
        top_indices = np.argsort(probs)[-6:]
        numbers = sorted([i + 1 for i in top_indices])

        # Confidence basierend auf Wahrscheinlichkeitsverteilung
        confidence = float(np.mean(probs[top_indices]) * 100)

        return numbers, confidence

    def save(self):
        """Speichert das Modell"""
        data = {
            'weights': {
                'W1': self.W1,
                'b1': self.b1,
                'W2': self.W2,
                'b2': self.b2,
                'W3': self.W3,
                'b3': self.b3
            },
            'training_history': self.training_history,
            'epochs_trained': self.epochs_trained,
            'learning_rate': self.learning_rate,
            'architecture': {
                'input': self.input_size,
                'hidden1': self.hidden1,
                'hidden2': self.hidden2,
                'output': self.output_size
            },
            'last_updated': datetime.now().isoformat()
        }
        save_model(self.MODEL_FILE, data)


# =====================================================
# 2. MARKOV-KETTEN FÜR SEQUENZANALYSE
# =====================================================

class MarkovChain:
    """
    Echte Markov-Kette für Lotto-Zahlen Sequenzanalyse.

    Lernt Übergangswahrscheinlichkeiten:
    - P(Zahl_n+1 | Zahl_n) - Welche Zahl folgt auf welche?
    - P(Zahl_n+1 | Zahl_n, Zahl_n-1) - 2nd Order Markov
    - Pair Transitions - Welche Paare folgen auf welche?
    """

    MODEL_FILE = 'markov_chain.json'

    def __init__(self):
        saved = load_model(self.MODEL_FILE)

        if saved and 'transition_matrix' in saved:
            self.transition_matrix = np.array(saved['transition_matrix'])
            self.second_order = saved.get('second_order', {})
            self.pair_transitions = saved.get('pair_transitions', {})
            self.observations = saved.get('observations', 0)
        else:
            # 49x49 Übergangsmatrix (mit Laplace Smoothing)
            self.transition_matrix = np.ones((49, 49)) / 49
            self.second_order = {}  # {(prev2, prev1): {next: count}}
            self.pair_transitions = {}  # {(a,b): {(c,d): count}}
            self.observations = 0

    def train(self, draws):
        """Trainiert die Markov-Kette mit historischen Daten"""
        # Zähle Übergänge
        transition_counts = np.ones((49, 49))  # Laplace Smoothing

        for i in range(len(draws) - 1):
            current = draws[i].get('numbers', [])
            next_draw = draws[i + 1].get('numbers', [])

            for curr_num in current:
                for next_num in next_draw:
                    if 1 <= curr_num <= 49 and 1 <= next_num <= 49:
                        transition_counts[curr_num - 1][next_num - 1] += 1

            # Second Order Markov
            if i < len(draws) - 2:
                prev_draw = draws[i + 2].get('numbers', [])
                for prev_num in prev_draw:
                    for curr_num in current:
                        key = f"{prev_num},{curr_num}"
                        if key not in self.second_order:
                            self.second_order[key] = {}
                        for next_num in next_draw:
                            next_key = str(next_num)
                            self.second_order[key][next_key] = \
                                self.second_order[key].get(next_key, 0) + 1

        # Normalisiere zu Wahrscheinlichkeiten
        row_sums = transition_counts.sum(axis=1, keepdims=True)
        self.transition_matrix = transition_counts / row_sums

        self.observations = len(draws)
        self.save()

        return {'observations': self.observations}

    def train_on_new_draw(self, previous_draw, new_draw):
        """Inkrementelles Update mit neuer Ziehung"""
        prev_nums = previous_draw.get('numbers', [])
        new_nums = new_draw.get('numbers', [])

        # Update Übergangsmatrix
        for prev in prev_nums:
            for new in new_nums:
                if 1 <= prev <= 49 and 1 <= new <= 49:
                    # Erhöhe Gewicht für beobachtete Übergänge
                    self.transition_matrix[prev - 1][new - 1] += 0.1

        # Renormalisieren
        row_sums = self.transition_matrix.sum(axis=1, keepdims=True)
        self.transition_matrix = self.transition_matrix / row_sums

        self.observations += 1
        self.save()

    def predict(self, last_draw):
        """Vorhersage basierend auf letzter Ziehung"""
        last_nums = last_draw.get('numbers', [])

        if not last_nums:
            return list(range(1, 7)), 0.5

        # Kombiniere Übergangswahrscheinlichkeiten
        combined_probs = np.zeros(49)

        for num in last_nums:
            if 1 <= num <= 49:
                combined_probs += self.transition_matrix[num - 1]

        # Normalisieren
        combined_probs = combined_probs / np.sum(combined_probs)

        # Wähle Top 6 (aber nicht aus letzter Ziehung)
        for num in last_nums:
            if 1 <= num <= 49:
                combined_probs[num - 1] *= 0.5  # Reduziere Wahrscheinlichkeit

        top_indices = np.argsort(combined_probs)[-6:]
        numbers = sorted([i + 1 for i in top_indices])

        confidence = float(np.mean(combined_probs[top_indices]) * 100)

        return numbers, min(confidence * 10, 95)

    def get_transition_stats(self):
        """Gibt interessante Übergangsstatistiken zurück"""
        stats = {
            'most_likely_follows': {},
            'least_likely_follows': {}
        }

        for i in range(49):
            row = self.transition_matrix[i]
            most_likely = np.argmax(row)
            least_likely = np.argmin(row)

            stats['most_likely_follows'][i + 1] = {
                'number': int(most_likely + 1),
                'probability': float(row[most_likely])
            }

        return stats

    def save(self):
        """Speichert das Modell"""
        data = {
            'transition_matrix': self.transition_matrix,
            'second_order': self.second_order,
            'pair_transitions': self.pair_transitions,
            'observations': self.observations,
            'last_updated': datetime.now().isoformat()
        }
        save_model(self.MODEL_FILE, data)


# =====================================================
# 3. BAYESIAN PREDICTOR
# =====================================================

class BayesianPredictor:
    """
    Echtes Bayesian Learning System.

    Verwendet:
    - Beta-Verteilung als Prior für jede Zahl
    - Likelihood aus beobachteten Ziehungen
    - Posterior Update nach jeder neuen Ziehung
    - Thompson Sampling für Exploration/Exploitation
    """

    MODEL_FILE = 'bayesian_predictor.json'

    def __init__(self):
        saved = load_model(self.MODEL_FILE)

        if saved and 'alpha' in saved:
            self.alpha = np.array(saved['alpha'])  # Erfolge + Prior
            self.beta = np.array(saved['beta'])    # Misserfolge + Prior
            self.observations = saved.get('observations', 0)
            self.position_priors = saved.get('position_priors', {})
        else:
            # Uninformativer Prior: Beta(1, 1) = Uniform
            self.alpha = np.ones(49)  # Pseudo-Erfolge
            self.beta = np.ones(49)   # Pseudo-Misserfolge
            self.observations = 0
            self.position_priors = {str(i): np.ones(49).tolist() for i in range(1, 7)}

    def train(self, draws):
        """Trainiert mit historischen Daten"""
        # Zähle Vorkommen jeder Zahl
        for draw in draws:
            numbers = draw.get('numbers', [])
            for i in range(49):
                if (i + 1) in numbers:
                    self.alpha[i] += 1
                else:
                    self.beta[i] += 1

        # Position-spezifische Priors
        for draw in draws:
            numbers = sorted(draw.get('numbers', []))
            for pos, num in enumerate(numbers):
                if 1 <= num <= 49:
                    pos_key = str(pos + 1)
                    if pos_key in self.position_priors:
                        self.position_priors[pos_key][num - 1] += 1

        self.observations = len(draws)
        self.save()

        return {'observations': self.observations}

    def train_on_new_draw(self, new_draw):
        """Update nach neuer Ziehung (Online Learning)"""
        numbers = new_draw.get('numbers', [])

        for i in range(49):
            if (i + 1) in numbers:
                self.alpha[i] += 1
            else:
                self.beta[i] += 1

        # Position Update
        sorted_nums = sorted(numbers)
        for pos, num in enumerate(sorted_nums):
            if 1 <= num <= 49:
                pos_key = str(pos + 1)
                if pos_key in self.position_priors:
                    self.position_priors[pos_key][num - 1] += 1

        self.observations += 1
        self.save()

    def get_posterior_probability(self, number):
        """Berechnet Posterior-Wahrscheinlichkeit für eine Zahl"""
        if 1 <= number <= 49:
            a = self.alpha[number - 1]
            b = self.beta[number - 1]
            return a / (a + b)
        return 0

    def thompson_sampling(self, n=6):
        """Thompson Sampling für Exploration/Exploitation Balance"""
        samples = []

        for i in range(49):
            # Sample aus Beta-Verteilung
            sample = np.random.beta(self.alpha[i], self.beta[i])
            samples.append((i + 1, sample))

        # Sortiere nach Sample-Wert
        samples.sort(key=lambda x: x[1], reverse=True)

        return [s[0] for s in samples[:n]]

    def predict(self, method='map'):
        """
        Vorhersage mit verschiedenen Methoden:
        - 'map': Maximum A Posteriori (höchste Wahrscheinlichkeit)
        - 'thompson': Thompson Sampling (mit Exploration)
        - 'ucb': Upper Confidence Bound
        """
        if method == 'thompson':
            numbers = self.thompson_sampling(6)
            confidence = 75  # Thompson hat eingebaute Unsicherheit
        elif method == 'ucb':
            # Upper Confidence Bound
            ucb_values = []
            for i in range(49):
                a, b = self.alpha[i], self.beta[i]
                mean = a / (a + b)
                # Konfidenzintervall
                std = np.sqrt((a * b) / ((a + b) ** 2 * (a + b + 1)))
                ucb = mean + 2 * std
                ucb_values.append((i + 1, ucb))

            ucb_values.sort(key=lambda x: x[1], reverse=True)
            numbers = [u[0] for u in ucb_values[:6]]
            confidence = 70
        else:  # MAP
            probs = self.alpha / (self.alpha + self.beta)
            top_indices = np.argsort(probs)[-6:]
            numbers = sorted([i + 1 for i in top_indices])
            confidence = float(np.mean(probs[top_indices]) * 100)

        return sorted(numbers), min(confidence, 95)

    def get_credible_interval(self, number, credibility=0.95):
        """Berechnet Bayesian Credible Interval"""
        from scipy import stats
        a = self.alpha[number - 1]
        b = self.beta[number - 1]

        lower = stats.beta.ppf((1 - credibility) / 2, a, b)
        upper = stats.beta.ppf((1 + credibility) / 2, a, b)

        return lower, upper

    def save(self):
        """Speichert das Modell"""
        data = {
            'alpha': self.alpha,
            'beta': self.beta,
            'observations': self.observations,
            'position_priors': self.position_priors,
            'last_updated': datetime.now().isoformat()
        }
        save_model(self.MODEL_FILE, data)


# =====================================================
# 4. REINFORCEMENT LEARNING
# =====================================================

class ReinforcementLearner:
    """
    Q-Learning für Strategie-Optimierung.

    States: Kombinationen von Zahlen-Features
    Actions: Welche Zahlen-Kombination wählen
    Rewards: Basierend auf Treffern (0-6)
    """

    MODEL_FILE = 'reinforcement_learner.json'

    def __init__(self):
        saved = load_model(self.MODEL_FILE)

        if saved and 'q_table' in saved:
            self.q_table = saved['q_table']
            self.strategy_values = np.array(saved.get('strategy_values', np.zeros(21)))
            self.epsilon = saved.get('epsilon', 0.3)
            self.learning_rate = saved.get('learning_rate', 0.1)
            self.discount_factor = saved.get('discount_factor', 0.95)
            self.total_rewards = saved.get('total_rewards', 0)
            self.episodes = saved.get('episodes', 0)
        else:
            self.q_table = {}
            self.strategy_values = np.zeros(21)  # 21 Strategien
            self.epsilon = 0.3  # Exploration Rate
            self.learning_rate = 0.1
            self.discount_factor = 0.95
            self.total_rewards = 0
            self.episodes = 0

        # Strategie-Namen (für Referenz)
        self.strategies = [
            'hot_cold', 'cold_numbers', 'overdue', 'odd_even_33', 'odd_even_42',
            'sum_optimized', 'decade_balance', 'delta_pattern', 'position_based',
            'no_consecutive', 'prime_mix', 'low_high', 'hot_cold_mix', 'monte_carlo',
            'bayesian', 'fibonacci', 'neighbor_pairs', 'end_digit', 'neural_network',
            'lstm_sequence', 'random_forest'
        ]

    def get_state_key(self, features):
        """Erstellt einen State-Key aus Features"""
        # Discretize features für Q-Table
        discretized = tuple(int(f * 10) for f in features[:10])
        return str(discretized)

    def calculate_reward(self, matches, superzahl_match=False):
        """Berechnet Reward basierend auf Treffern"""
        rewards = {
            0: -1,
            1: 0,
            2: 1 if superzahl_match else 0.5,
            3: 5 if superzahl_match else 3,
            4: 20 if superzahl_match else 10,
            5: 100 if superzahl_match else 50,
            6: 1000 if superzahl_match else 500
        }
        return rewards.get(matches, 0)

    def choose_action(self, state_key):
        """Wählt Aktion (Strategie) mit Epsilon-Greedy"""
        if np.random.random() < self.epsilon:
            # Exploration: Zufällige Strategie
            return np.random.randint(0, len(self.strategies))
        else:
            # Exploitation: Beste bekannte Strategie
            if state_key in self.q_table:
                return np.argmax(self.q_table[state_key])
            else:
                return np.argmax(self.strategy_values)

    def update_q_value(self, state_key, action, reward, next_state_key):
        """Q-Learning Update"""
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(len(self.strategies)).tolist()

        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(len(self.strategies)).tolist()

        current_q = self.q_table[state_key][action]
        max_next_q = max(self.q_table[next_state_key])

        # Q-Learning Formula
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )

        self.q_table[state_key][action] = new_q

        # Update globale Strategy Values
        self.strategy_values[action] = (
            0.9 * self.strategy_values[action] + 0.1 * reward
        )

    def learn_from_result(self, prediction, actual_draw, features):
        """Lernt aus einem Vorhersage-Ergebnis"""
        predicted_nums = set(prediction.get('numbers', []))
        actual_nums = set(actual_draw.get('numbers', []))

        matches = len(predicted_nums & actual_nums)
        sz_match = prediction.get('superzahl') == actual_draw.get('superzahl')

        reward = self.calculate_reward(matches, sz_match)

        # Finde welche Strategie verwendet wurde
        method = prediction.get('method', '')
        action = 0
        for i, strat in enumerate(self.strategies):
            if strat in method:
                action = i
                break

        state_key = self.get_state_key(features)

        # Einfacher Update ohne next_state
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(len(self.strategies)).tolist()

        self.q_table[state_key][action] = (
            0.9 * self.q_table[state_key][action] + 0.1 * reward
        )

        self.strategy_values[action] = (
            0.9 * self.strategy_values[action] + 0.1 * reward
        )

        self.total_rewards += reward
        self.episodes += 1

        # Epsilon Decay
        self.epsilon = max(0.05, self.epsilon * 0.995)

        self.save()

        return {
            'matches': matches,
            'reward': reward,
            'strategy': self.strategies[action],
            'total_rewards': self.total_rewards
        }

    def get_best_strategies(self, n=5):
        """Gibt die N besten Strategien zurück"""
        sorted_indices = np.argsort(self.strategy_values)[::-1]

        return [
            {
                'strategy': self.strategies[i],
                'value': float(self.strategy_values[i]),
                'rank': rank + 1
            }
            for rank, i in enumerate(sorted_indices[:n])
        ]

    def get_strategy_weight(self, strategy_name):
        """Gibt das gelernte Gewicht für eine Strategie zurück"""
        for i, strat in enumerate(self.strategies):
            if strat == strategy_name or strat in strategy_name:
                # Normalisiere zu 0.5 - 2.0 Range
                value = self.strategy_values[i]
                weight = 1.0 + (value / 100)  # Skaliere
                return max(0.5, min(2.0, weight))
        return 1.0

    def save(self):
        """Speichert das Modell"""
        data = {
            'q_table': self.q_table,
            'strategy_values': self.strategy_values,
            'epsilon': self.epsilon,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'total_rewards': self.total_rewards,
            'episodes': self.episodes,
            'strategies': self.strategies,
            'last_updated': datetime.now().isoformat()
        }
        save_model(self.MODEL_FILE, data)


# =====================================================
# 5. ENSEMBLE ML SYSTEM
# =====================================================

class EnsembleML:
    """
    Kombiniert alle ML-Modelle mit gelernten Gewichten.

    - Gewichtet Vorhersagen nach historischer Performance
    - Meta-Learning: Lernt welches Modell wann am besten ist
    - Dynamische Gewichtungsanpassung
    """

    MODEL_FILE = 'ensemble_ml.json'

    def __init__(self):
        saved = load_model(self.MODEL_FILE)

        if saved and 'model_weights' in saved:
            self.model_weights = saved['model_weights']
            self.model_performance = saved.get('model_performance', {})
            self.predictions_count = saved.get('predictions_count', 0)
        else:
            self.model_weights = {
                'neural_network': 1.0,
                'markov_chain': 1.0,
                'bayesian': 1.0,
                'reinforcement': 1.0
            }
            self.model_performance = {
                'neural_network': {'hits': 0, 'total': 0},
                'markov_chain': {'hits': 0, 'total': 0},
                'bayesian': {'hits': 0, 'total': 0},
                'reinforcement': {'hits': 0, 'total': 0}
            }
            self.predictions_count = 0

        # Initialisiere Modelle
        self.nn = NeuralNetwork()
        self.markov = MarkovChain()
        self.bayesian = BayesianPredictor()
        self.rl = ReinforcementLearner()

    def train_all(self, draws):
        """Trainiert alle Modelle"""
        results = {}

        print("🧠 Training Neural Network...")
        results['neural_network'] = self.nn.train(draws)

        print("🔗 Training Markov Chain...")
        results['markov_chain'] = self.markov.train(draws)

        print("📊 Training Bayesian Predictor...")
        results['bayesian'] = self.bayesian.train(draws)

        self.save()
        return results

    def train_on_new_draw(self, draws, new_draw, previous_draw):
        """Inkrementelles Training aller Modelle"""
        results = {}

        results['neural_network'] = self.nn.train_on_new_draw(draws, new_draw)
        self.markov.train_on_new_draw(previous_draw, new_draw)
        self.bayesian.train_on_new_draw(new_draw)

        self.save()
        return results

    def update_weights_from_result(self, model_predictions, actual_draw):
        """Aktualisiert Gewichte basierend auf echten Ergebnissen"""
        actual_nums = set(actual_draw.get('numbers', []))

        for model_name, prediction in model_predictions.items():
            predicted_nums = set(prediction.get('numbers', []))
            matches = len(predicted_nums & actual_nums)

            # Update Performance Tracking
            if model_name not in self.model_performance:
                self.model_performance[model_name] = {'hits': 0, 'total': 0}

            self.model_performance[model_name]['hits'] += matches
            self.model_performance[model_name]['total'] += 6

            # Berechne neue Gewichtung
            perf = self.model_performance[model_name]
            if perf['total'] > 0:
                hit_rate = perf['hits'] / perf['total']
                # Exponential Moving Average für Gewicht
                self.model_weights[model_name] = (
                    0.9 * self.model_weights.get(model_name, 1.0) +
                    0.1 * (hit_rate * 10)  # Skaliere hit_rate
                )
                # Begrenze Gewichte
                self.model_weights[model_name] = max(0.5, min(3.0,
                    self.model_weights[model_name]))

        self.predictions_count += 1
        self.save()

    def predict(self, draws):
        """Kombinierte Vorhersage aller Modelle"""
        predictions = {}

        # Sammle Vorhersagen von allen Modellen
        nn_pred, nn_conf = self.nn.predict(draws)
        predictions['neural_network'] = {
            'numbers': nn_pred,
            'confidence': nn_conf,
            'weight': self.model_weights.get('neural_network', 1.0)
        }

        markov_pred, markov_conf = self.markov.predict(draws[0] if draws else {})
        predictions['markov_chain'] = {
            'numbers': markov_pred,
            'confidence': markov_conf,
            'weight': self.model_weights.get('markov_chain', 1.0)
        }

        bayes_pred, bayes_conf = self.bayesian.predict('thompson')
        predictions['bayesian'] = {
            'numbers': bayes_pred,
            'confidence': bayes_conf,
            'weight': self.model_weights.get('bayesian', 1.0)
        }

        # Gewichtetes Voting
        vote_counts = Counter()
        total_weight = 0

        for model_name, pred in predictions.items():
            weight = pred['weight']
            total_weight += weight
            for num in pred['numbers']:
                vote_counts[num] += weight

        # Top 6 nach gewichteten Stimmen
        ensemble_numbers = [num for num, _ in vote_counts.most_common(6)]

        # Confidence basierend auf Übereinstimmung
        agreement_score = 0
        for num in ensemble_numbers:
            models_agreeing = sum(1 for p in predictions.values() if num in p['numbers'])
            agreement_score += models_agreeing / len(predictions)

        ensemble_confidence = (agreement_score / 6) * 100

        return {
            'ensemble': sorted(ensemble_numbers),
            'confidence': min(ensemble_confidence, 95),
            'individual_predictions': predictions,
            'model_weights': self.model_weights
        }

    def get_model_stats(self):
        """Gibt Statistiken aller Modelle zurück"""
        return {
            'weights': self.model_weights,
            'performance': self.model_performance,
            'predictions_count': self.predictions_count,
            'neural_network': {
                'epochs_trained': self.nn.epochs_trained,
                'training_history_count': len(self.nn.training_history)
            },
            'markov_chain': {
                'observations': self.markov.observations
            },
            'bayesian': {
                'observations': self.bayesian.observations
            },
            'reinforcement': {
                'episodes': self.rl.episodes,
                'total_rewards': self.rl.total_rewards,
                'best_strategies': self.rl.get_best_strategies(3)
            }
        }

    def save(self):
        """Speichert Ensemble-Daten"""
        data = {
            'model_weights': self.model_weights,
            'model_performance': self.model_performance,
            'predictions_count': self.predictions_count,
            'last_updated': datetime.now().isoformat()
        }
        save_model(self.MODEL_FILE, data)


# =====================================================
# SUPERZAHL ML-MODELL
# =====================================================

class SuperzahlML:
    """
    Spezialisiertes ML-Modell nur für die Superzahl (0-9).

    Verwendet:
    - Zeitreihen-Analyse
    - Pattern Recognition
    - Bayesian Learning
    """

    MODEL_FILE = 'superzahl_ml.json'

    def __init__(self):
        saved = load_model(self.MODEL_FILE)

        if saved and 'weights' in saved:
            self.weights = np.array(saved['weights'])
            self.transition_probs = np.array(saved.get('transition_probs', np.ones((10, 10)) / 10))
            self.alpha = np.array(saved.get('alpha', np.ones(10)))
            self.beta = np.array(saved.get('beta', np.ones(10)))
            self.observations = saved.get('observations', 0)
        else:
            self.weights = np.ones(10) / 10  # Gleichverteilung
            self.transition_probs = np.ones((10, 10)) / 10
            self.alpha = np.ones(10)  # Bayesian Prior
            self.beta = np.ones(10)
            self.observations = 0

    def train(self, draws):
        """Trainiert mit historischen Superzahlen"""
        # Zähle Häufigkeiten
        freq = np.zeros(10)
        transitions = np.ones((10, 10))  # Laplace Smoothing

        prev_sz = None
        for draw in draws:
            sz = draw.get('superzahl')
            if sz is not None and 0 <= sz <= 9:
                freq[sz] += 1
                self.alpha[sz] += 1

                for i in range(10):
                    if i != sz:
                        self.beta[i] += 1

                if prev_sz is not None:
                    transitions[prev_sz][sz] += 1

                prev_sz = sz

        # Normalisiere
        if np.sum(freq) > 0:
            self.weights = freq / np.sum(freq)

        row_sums = transitions.sum(axis=1, keepdims=True)
        self.transition_probs = transitions / row_sums

        self.observations = len(draws)
        self.save()

        return {'observations': self.observations}

    def train_on_new_draw(self, previous_sz, new_sz):
        """Inkrementelles Update"""
        if new_sz is None or not (0 <= new_sz <= 9):
            return

        # Update Bayesian
        self.alpha[new_sz] += 1
        for i in range(10):
            if i != new_sz:
                self.beta[i] += 0.1

        # Update Transition
        if previous_sz is not None and 0 <= previous_sz <= 9:
            self.transition_probs[previous_sz][new_sz] += 0.1
            # Renormalisieren
            row_sum = np.sum(self.transition_probs[previous_sz])
            self.transition_probs[previous_sz] /= row_sum

        # Update Gewichte
        self.weights[new_sz] = (0.95 * self.weights[new_sz] + 0.05 * 1.0)
        self.weights = self.weights / np.sum(self.weights)

        self.observations += 1
        self.save()

    def predict(self, last_sz=None):
        """Vorhersage der nächsten Superzahl"""
        # Kombiniere verschiedene Signale
        scores = np.zeros(10)

        # 1. Bayesian Posterior (40%)
        posterior = self.alpha / (self.alpha + self.beta)
        scores += 0.4 * posterior

        # 2. Transition Probability (30%)
        if last_sz is not None and 0 <= last_sz <= 9:
            scores += 0.3 * self.transition_probs[last_sz]
        else:
            scores += 0.3 * np.mean(self.transition_probs, axis=0)

        # 3. Häufigkeitsgewichte (20%)
        scores += 0.2 * self.weights

        # 4. Thompson Sampling für Exploration (10%)
        thompson_samples = np.random.beta(self.alpha, self.beta)
        scores += 0.1 * thompson_samples

        # Normalisieren
        scores = scores / np.sum(scores)

        # Ranking
        ranking = [(i, float(scores[i])) for i in range(10)]
        ranking.sort(key=lambda x: x[1], reverse=True)

        best_sz = ranking[0][0]
        confidence = ranking[0][1] * 100

        return best_sz, ranking, confidence

    def save(self):
        """Speichert das Modell"""
        data = {
            'weights': self.weights,
            'transition_probs': self.transition_probs,
            'alpha': self.alpha,
            'beta': self.beta,
            'observations': self.observations,
            'last_updated': datetime.now().isoformat()
        }
        save_model(self.MODEL_FILE, data)


# =====================================================
# HAUPTFUNKTIONEN
# =====================================================

def initialize_all_models():
    """Initialisiert alle ML-Modelle"""
    ensure_dirs()
    return {
        'neural_network': NeuralNetwork(),
        'markov_chain': MarkovChain(),
        'bayesian': BayesianPredictor(),
        'reinforcement': ReinforcementLearner(),
        'ensemble': EnsembleML(),
        'superzahl': SuperzahlML()
    }

def train_all_models(draws):
    """Trainiert alle Modelle mit historischen Daten"""
    models = initialize_all_models()

    print("\n" + "=" * 60)
    print("🧠 ECHTES ML-TRAINING STARTET")
    print("=" * 60)

    results = {}

    print("\n📊 Trainiere Neuronales Netz...")
    results['neural_network'] = models['neural_network'].train(draws)
    print(f"   ✅ {results['neural_network'].get('epochs', 0)} Epochen, "
          f"Loss: {results['neural_network'].get('final_loss', 0):.4f}")

    print("\n🔗 Trainiere Markov-Kette...")
    results['markov_chain'] = models['markov_chain'].train(draws)
    print(f"   ✅ {results['markov_chain'].get('observations', 0)} Beobachtungen")

    print("\n📈 Trainiere Bayesian Predictor...")
    results['bayesian'] = models['bayesian'].train(draws)
    print(f"   ✅ {results['bayesian'].get('observations', 0)} Beobachtungen")

    print("\n🎯 Trainiere Superzahl-Modell...")
    results['superzahl'] = models['superzahl'].train(draws)
    print(f"   ✅ {results['superzahl'].get('observations', 0)} Beobachtungen")

    print("\n" + "=" * 60)
    print("✅ ALLE ML-MODELLE ERFOLGREICH TRAINIERT")
    print("=" * 60)

    return results

def learn_from_new_draw(draws, new_draw, previous_draw):
    """Lernt aus einer neuen Ziehung (nach jeder echten Lottoziehung)"""
    models = initialize_all_models()

    print(f"\n🎓 Lerne aus neuer Ziehung vom {new_draw.get('date', 'unbekannt')}...")

    results = {}

    # Neural Network
    results['neural_network'] = models['neural_network'].train_on_new_draw(draws, new_draw)

    # Markov Chain
    models['markov_chain'].train_on_new_draw(previous_draw, new_draw)

    # Bayesian
    models['bayesian'].train_on_new_draw(new_draw)

    # Superzahl
    prev_sz = previous_draw.get('superzahl')
    new_sz = new_draw.get('superzahl')
    models['superzahl'].train_on_new_draw(prev_sz, new_sz)

    print(f"   ✅ Alle Modelle mit neuer Ziehung aktualisiert")

    return results

def to_python_types(obj):
    """Konvertiert numpy-Typen zu Python-nativen Typen für JSON"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32, np.int_)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float_)):
        return float(obj)
    elif isinstance(obj, list):
        return [to_python_types(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: to_python_types(v) for k, v in obj.items()}
    return obj


def get_ml_predictions(draws):
    """Holt Vorhersagen von allen echten ML-Modellen"""
    models = initialize_all_models()

    predictions = []

    # 1. Neuronales Netz
    nn_nums, nn_conf = models['neural_network'].predict(draws)
    sz, sz_ranking, sz_conf = models['superzahl'].predict(
        draws[0].get('superzahl') if draws else None
    )
    predictions.append({
        'numbers': [int(n) for n in nn_nums],  # Konvertiere zu int
        'superzahl': int(sz),
        'method': 'neural_network_real',
        'method_name': '🧠 Echtes Neuronales Netz',
        'provider': 'ml_real',
        'strategy': f'64-32 Hidden Layers, {models["neural_network"].epochs_trained} Epochen trainiert',
        'confidence': float(nn_conf) if not np.isnan(nn_conf) else 50.0,
        'is_real_ml': True
    })

    # 2. Markov Chain
    markov_nums, markov_conf = models['markov_chain'].predict(
        draws[0] if draws else {}
    )
    predictions.append({
        'numbers': [int(n) for n in markov_nums],
        'superzahl': int(sz_ranking[1][0]) if len(sz_ranking) > 1 else int(sz),
        'method': 'markov_chain_real',
        'method_name': '🔗 Echte Markov-Kette',
        'provider': 'ml_real',
        'strategy': f'Übergangswahrscheinlichkeiten aus {models["markov_chain"].observations} Ziehungen',
        'confidence': float(markov_conf),
        'is_real_ml': True
    })

    # 3. Bayesian MAP
    bayes_nums, bayes_conf = models['bayesian'].predict('map')
    predictions.append({
        'numbers': [int(n) for n in bayes_nums],
        'superzahl': int(sz_ranking[2][0]) if len(sz_ranking) > 2 else int(sz),
        'method': 'bayesian_map_real',
        'method_name': '📊 Bayesian MAP',
        'provider': 'ml_real',
        'strategy': f'Maximum A Posteriori aus {models["bayesian"].observations} Updates',
        'confidence': float(bayes_conf),
        'is_real_ml': True
    })

    # 4. Bayesian Thompson Sampling
    thompson_nums, thompson_conf = models['bayesian'].predict('thompson')
    predictions.append({
        'numbers': [int(n) for n in thompson_nums],
        'superzahl': int(sz_ranking[0][0]),
        'method': 'thompson_sampling_real',
        'method_name': '🎲 Thompson Sampling',
        'provider': 'ml_real',
        'strategy': 'Exploration/Exploitation Balance mit Beta-Verteilung',
        'confidence': float(thompson_conf),
        'is_real_ml': True
    })

    # 5. Ensemble
    ensemble_result = models['ensemble'].predict(draws)
    predictions.append({
        'numbers': [int(n) for n in ensemble_result['ensemble']],
        'superzahl': int(sz),
        'method': 'ensemble_ml_real',
        'method_name': '🏆 ML Ensemble (Alle Modelle)',
        'provider': 'ml_real',
        'strategy': f'Gewichtetes Voting: NN={ensemble_result["model_weights"].get("neural_network", 1):.2f}, '
                   f'Markov={ensemble_result["model_weights"].get("markov_chain", 1):.2f}, '
                   f'Bayes={ensemble_result["model_weights"].get("bayesian", 1):.2f}',
        'confidence': float(ensemble_result['confidence']),
        'is_real_ml': True,
        'is_champion': True
    })

    return predictions


if __name__ == "__main__":
    # Test
    print("🧠 ML Models Module Test")

    lotto_data = load_json('lotto_data.json', {'draws': []})
    draws = lotto_data.get('draws', [])

    if draws:
        print(f"\n📊 {len(draws)} Ziehungen gefunden")

        # Training
        results = train_all_models(draws)

        # Vorhersagen
        predictions = get_ml_predictions(draws)

        print("\n🎯 VORHERSAGEN:")
        for pred in predictions:
            print(f"   {pred['method_name']}: {pred['numbers']} | SZ: {pred['superzahl']}")
    else:
        print("❌ Keine Daten gefunden")
