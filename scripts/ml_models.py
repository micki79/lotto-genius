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
    """Konvertiert numpy-Typen zu Python-nativen Typen für JSON (NumPy 2.0 kompatibel)"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(np, 'integer') and isinstance(obj, np.integer):
        return int(obj)
    elif hasattr(np, 'floating') and isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, (np.int64, np.int32, np.intc)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
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


# =====================================================
# EUROJACKPOT ML-MODELLE
# =====================================================

class EurojackpotNeuralNetwork:
    """
    Neuronales Netz für Eurojackpot (5 aus 50 + 2 aus 12).

    Architektur:
    - Input: 50 Neuronen (Häufigkeit Hauptzahlen) + 12 (Eurozahlen)
    - Hidden 1: 64 Neuronen (ReLU)
    - Hidden 2: 32 Neuronen (ReLU)
    - Output: 50 Neuronen (Hauptzahlen) + 12 (Eurozahlen)
    """

    MODEL_FILE = 'eurojackpot_neural_network.json'

    def __init__(self):
        saved = load_model(self.MODEL_FILE)

        if saved and 'weights' in saved:
            self.W1_main = np.array(saved['weights']['W1_main'])
            self.b1_main = np.array(saved['weights']['b1_main'])
            self.W2_main = np.array(saved['weights']['W2_main'])
            self.b2_main = np.array(saved['weights']['b2_main'])
            self.W3_main = np.array(saved['weights']['W3_main'])
            self.b3_main = np.array(saved['weights']['b3_main'])

            self.W1_euro = np.array(saved['weights']['W1_euro'])
            self.b1_euro = np.array(saved['weights']['b1_euro'])
            self.W2_euro = np.array(saved['weights']['W2_euro'])
            self.b2_euro = np.array(saved['weights']['b2_euro'])

            self.epochs_trained = saved.get('epochs_trained', 0)
            self.learning_rate = saved.get('learning_rate', 0.01)
        else:
            # Hauptzahlen Netzwerk (50 -> 64 -> 32 -> 50)
            self.W1_main = np.random.randn(50, 64) * np.sqrt(2.0 / 50)
            self.b1_main = np.zeros((1, 64))
            self.W2_main = np.random.randn(64, 32) * np.sqrt(2.0 / 64)
            self.b2_main = np.zeros((1, 32))
            self.W3_main = np.random.randn(32, 50) * np.sqrt(2.0 / 32)
            self.b3_main = np.zeros((1, 50))

            # Eurozahlen Netzwerk (12 -> 24 -> 12)
            self.W1_euro = np.random.randn(12, 24) * np.sqrt(2.0 / 12)
            self.b1_euro = np.zeros((1, 24))
            self.W2_euro = np.random.randn(24, 12) * np.sqrt(2.0 / 24)
            self.b2_euro = np.zeros((1, 12))

            self.epochs_trained = 0
            self.learning_rate = 0.01

    def relu(self, x):
        return np.clip(np.maximum(0, x), 0, 100)

    def softmax(self, x):
        x_clipped = np.clip(x, -100, 100)
        x_shifted = x_clipped - np.max(x_clipped, axis=1, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / (np.sum(exp_x, axis=1, keepdims=True) + 1e-10)

    def forward_main(self, X):
        z1 = np.dot(X, self.W1_main) + self.b1_main
        a1 = self.relu(z1)
        z2 = np.dot(a1, self.W2_main) + self.b2_main
        a2 = self.relu(z2)
        z3 = np.dot(a2, self.W3_main) + self.b3_main
        return self.softmax(z3)

    def forward_euro(self, X):
        z1 = np.dot(X, self.W1_euro) + self.b1_euro
        a1 = self.relu(z1)
        z2 = np.dot(a1, self.W2_euro) + self.b2_euro
        return self.softmax(z2)

    def create_features(self, draws, num_draws=100):
        """Erstellt Feature-Vektoren für Hauptzahlen und Eurozahlen"""
        main_features = []
        euro_features = []

        for i in range(min(num_draws, len(draws) - 1)):
            # Hauptzahlen-Features
            main_freq = np.zeros(50)
            euro_freq = np.zeros(12)

            for j in range(i, min(i + 20, len(draws))):
                for num in draws[j].get('numbers', []):
                    if 1 <= num <= 50:
                        main_freq[num - 1] += 1
                for ez in draws[j].get('eurozahlen', []):
                    if 1 <= ez <= 12:
                        euro_freq[ez - 1] += 1

            if np.max(main_freq) > 0:
                main_freq = main_freq / np.max(main_freq)
            if np.max(euro_freq) > 0:
                euro_freq = euro_freq / np.max(euro_freq)

            main_features.append(main_freq)
            euro_features.append(euro_freq)

        return (np.array(main_features) if main_features else np.zeros((1, 50)),
                np.array(euro_features) if euro_features else np.zeros((1, 12)))

    def train(self, draws, epochs=100):
        """Trainiert das Netzwerk"""
        X_main, X_euro = self.create_features(draws, num_draws=300)

        if len(X_main) < 2:
            return {'error': 'Nicht genug Daten'}

        # Erstelle Labels
        y_main = np.zeros((len(X_main), 50))
        y_euro = np.zeros((len(X_euro), 12))

        for i in range(len(X_main) - 1):
            for num in draws[i].get('numbers', []):
                if 1 <= num <= 50:
                    y_main[i + 1, num - 1] = 1
            for ez in draws[i].get('eurozahlen', []):
                if 1 <= ez <= 12:
                    y_euro[i + 1, ez - 1] = 1

        for epoch in range(epochs):
            # Forward + Backward für Hauptzahlen
            output_main = self.forward_main(X_main)
            dz3 = output_main - y_main
            dW3 = np.clip(np.dot(self.relu(np.dot(self.relu(np.dot(X_main, self.W1_main) + self.b1_main), self.W2_main) + self.b2_main).T, dz3) / len(X_main), -1, 1)

            self.W3_main -= self.learning_rate * dW3
            self.W3_main = np.clip(self.W3_main, -5, 5)

            # Forward + Backward für Eurozahlen
            output_euro = self.forward_euro(X_euro)
            dz2 = output_euro - y_euro
            dW2 = np.clip(np.dot(self.relu(np.dot(X_euro, self.W1_euro) + self.b1_euro).T, dz2) / len(X_euro), -1, 1)

            self.W2_euro -= self.learning_rate * dW2
            self.W2_euro = np.clip(self.W2_euro, -5, 5)

            if epoch % 20 == 0:
                self.learning_rate *= 0.95

        self.epochs_trained += epochs
        self.save()

        return {'epochs': epochs, 'total_epochs': self.epochs_trained}

    def predict(self, draws):
        """Vorhersage für Eurojackpot"""
        X_main, X_euro = self.create_features(draws[:30], num_draws=1)

        main_probs = self.forward_main(X_main)[0]
        euro_probs = self.forward_euro(X_euro)[0]

        # Top 5 Hauptzahlen
        top_main = np.argsort(main_probs)[-5:]
        main_numbers = sorted([i + 1 for i in top_main])

        # Top 2 Eurozahlen
        top_euro = np.argsort(euro_probs)[-2:]
        euro_numbers = sorted([i + 1 for i in top_euro])

        confidence = float(np.mean(main_probs[top_main]) * 100)

        return main_numbers, euro_numbers, confidence

    def save(self):
        data = {
            'weights': {
                'W1_main': self.W1_main, 'b1_main': self.b1_main,
                'W2_main': self.W2_main, 'b2_main': self.b2_main,
                'W3_main': self.W3_main, 'b3_main': self.b3_main,
                'W1_euro': self.W1_euro, 'b1_euro': self.b1_euro,
                'W2_euro': self.W2_euro, 'b2_euro': self.b2_euro,
            },
            'epochs_trained': self.epochs_trained,
            'learning_rate': self.learning_rate,
            'last_updated': datetime.now().isoformat()
        }
        save_model(self.MODEL_FILE, data)


class EurojackpotBayesian:
    """Bayesian Predictor für Eurojackpot"""

    MODEL_FILE = 'eurojackpot_bayesian.json'

    def __init__(self):
        saved = load_model(self.MODEL_FILE)

        if saved and 'alpha_main' in saved:
            self.alpha_main = np.array(saved['alpha_main'])
            self.beta_main = np.array(saved['beta_main'])
            self.alpha_euro = np.array(saved['alpha_euro'])
            self.beta_euro = np.array(saved['beta_euro'])
            self.observations = saved.get('observations', 0)
        else:
            self.alpha_main = np.ones(50)
            self.beta_main = np.ones(50)
            self.alpha_euro = np.ones(12)
            self.beta_euro = np.ones(12)
            self.observations = 0

    def train(self, draws):
        """Trainiert mit historischen Daten"""
        for draw in draws:
            main_nums = draw.get('numbers', [])
            euro_nums = draw.get('eurozahlen', [])

            for i in range(50):
                if (i + 1) in main_nums:
                    self.alpha_main[i] += 1
                else:
                    self.beta_main[i] += 1

            for i in range(12):
                if (i + 1) in euro_nums:
                    self.alpha_euro[i] += 1
                else:
                    self.beta_euro[i] += 1

        self.observations = len(draws)
        self.save()
        return {'observations': self.observations}

    def predict(self, method='thompson'):
        """Vorhersage mit Thompson Sampling"""
        if method == 'thompson':
            main_samples = [np.random.beta(self.alpha_main[i], self.beta_main[i]) for i in range(50)]
            euro_samples = [np.random.beta(self.alpha_euro[i], self.beta_euro[i]) for i in range(12)]

            main_indices = np.argsort(main_samples)[-5:]
            euro_indices = np.argsort(euro_samples)[-2:]

            main_numbers = sorted([i + 1 for i in main_indices])
            euro_numbers = sorted([i + 1 for i in euro_indices])
            confidence = 75
        else:  # MAP
            main_probs = self.alpha_main / (self.alpha_main + self.beta_main)
            euro_probs = self.alpha_euro / (self.alpha_euro + self.beta_euro)

            main_indices = np.argsort(main_probs)[-5:]
            euro_indices = np.argsort(euro_probs)[-2:]

            main_numbers = sorted([i + 1 for i in main_indices])
            euro_numbers = sorted([i + 1 for i in euro_indices])
            confidence = float(np.mean(main_probs[main_indices]) * 100)

        return main_numbers, euro_numbers, confidence

    def save(self):
        data = {
            'alpha_main': self.alpha_main,
            'beta_main': self.beta_main,
            'alpha_euro': self.alpha_euro,
            'beta_euro': self.beta_euro,
            'observations': self.observations,
            'last_updated': datetime.now().isoformat()
        }
        save_model(self.MODEL_FILE, data)


class EurojackpotMarkov:
    """Markov-Kette für Eurojackpot"""

    MODEL_FILE = 'eurojackpot_markov.json'

    def __init__(self):
        saved = load_model(self.MODEL_FILE)

        if saved and 'transition_main' in saved:
            self.transition_main = np.array(saved['transition_main'])
            self.transition_euro = np.array(saved['transition_euro'])
            self.observations = saved.get('observations', 0)
        else:
            self.transition_main = np.ones((50, 50)) / 50
            self.transition_euro = np.ones((12, 12)) / 12
            self.observations = 0

    def train(self, draws):
        """Trainiert Übergangswahrscheinlichkeiten"""
        main_counts = np.ones((50, 50))
        euro_counts = np.ones((12, 12))

        for i in range(len(draws) - 1):
            current_main = draws[i].get('numbers', [])
            next_main = draws[i + 1].get('numbers', [])
            current_euro = draws[i].get('eurozahlen', [])
            next_euro = draws[i + 1].get('eurozahlen', [])

            for cm in current_main:
                for nm in next_main:
                    if 1 <= cm <= 50 and 1 <= nm <= 50:
                        main_counts[cm - 1][nm - 1] += 1

            for ce in current_euro:
                for ne in next_euro:
                    if 1 <= ce <= 12 and 1 <= ne <= 12:
                        euro_counts[ce - 1][ne - 1] += 1

        self.transition_main = main_counts / main_counts.sum(axis=1, keepdims=True)
        self.transition_euro = euro_counts / euro_counts.sum(axis=1, keepdims=True)

        self.observations = len(draws)
        self.save()
        return {'observations': self.observations}

    def predict(self, last_draw):
        """Vorhersage basierend auf letzter Ziehung"""
        last_main = last_draw.get('numbers', [])
        last_euro = last_draw.get('eurozahlen', [])

        # Kombiniere Übergangswahrscheinlichkeiten
        main_probs = np.zeros(50)
        for num in last_main:
            if 1 <= num <= 50:
                main_probs += self.transition_main[num - 1]

        # Reduziere Wahrscheinlichkeit der letzten Zahlen
        for num in last_main:
            if 1 <= num <= 50:
                main_probs[num - 1] *= 0.5

        euro_probs = np.zeros(12)
        for ez in last_euro:
            if 1 <= ez <= 12:
                euro_probs += self.transition_euro[ez - 1]

        for ez in last_euro:
            if 1 <= ez <= 12:
                euro_probs[ez - 1] *= 0.5

        main_indices = np.argsort(main_probs)[-5:]
        euro_indices = np.argsort(euro_probs)[-2:]

        main_numbers = sorted([i + 1 for i in main_indices])
        euro_numbers = sorted([i + 1 for i in euro_indices])
        confidence = float(np.mean(main_probs[main_indices]) / np.sum(main_probs) * 100)

        return main_numbers, euro_numbers, min(confidence * 10, 90)

    def save(self):
        data = {
            'transition_main': self.transition_main,
            'transition_euro': self.transition_euro,
            'observations': self.observations,
            'last_updated': datetime.now().isoformat()
        }
        save_model(self.MODEL_FILE, data)


class EurojackpotEnsembleML:
    """
    Ensemble aller Eurojackpot ML-Modelle.

    Integriert 6 echte ML-Algorithmen:
    1. Neural Network (Backpropagation)
    2. Markov Chain (Übergangswahrscheinlichkeiten)
    3. Bayesian Predictor (Thompson Sampling)
    4. Reinforcement Learner (Q-Learning)
    5. Eurozahl ML (Spezialisiert auf 2 aus 12)
    """

    MODEL_FILE = 'eurojackpot_ensemble_ml.json'

    def __init__(self):
        saved = load_model(self.MODEL_FILE)

        if saved and 'model_weights' in saved:
            self.model_weights = saved['model_weights']
            self.model_performance = saved.get('model_performance', {})
        else:
            self.model_weights = {
                'neural_network': 1.0,
                'markov': 1.0,
                'bayesian': 1.0,
                'reinforcement': 1.0,
                'eurozahl_ml': 1.0
            }
            self.model_performance = {}

        # Alle 5 ML-Modelle initialisieren
        self.nn = EurojackpotNeuralNetwork()
        self.markov = EurojackpotMarkov()
        self.bayesian = EurojackpotBayesian()
        self.reinforcement = None  # Lazy init wegen zirkulärer Abhängigkeit
        self.eurozahl_ml = None    # Lazy init

    def _init_reinforcement(self):
        """Lazy Initialization für ReinforcementLearner"""
        if self.reinforcement is None:
            self.reinforcement = EurojackpotReinforcementLearner()
        return self.reinforcement

    def _init_eurozahl_ml(self):
        """Lazy Initialization für EurozahlML"""
        if self.eurozahl_ml is None:
            self.eurozahl_ml = EurozahlML()
        return self.eurozahl_ml

    def train_all(self, draws):
        """Trainiert alle 5 ML-Modelle"""
        results = {}

        print("   🧠 Training Eurojackpot Neural Network...")
        results['neural_network'] = self.nn.train(draws)

        print("   🔗 Training Eurojackpot Markov Chain...")
        results['markov'] = self.markov.train(draws)

        print("   📊 Training Eurojackpot Bayesian...")
        results['bayesian'] = self.bayesian.train(draws)

        print("   🎯 Training Eurozahl ML...")
        eurozahl_ml = self._init_eurozahl_ml()
        results['eurozahl_ml'] = eurozahl_ml.train(draws)

        print("   🎮 Reinforcement Learner initialisiert (lernt aus Ergebnissen)...")
        rl = self._init_reinforcement()
        results['reinforcement'] = {'episodes': rl.episodes, 'total_rewards': rl.total_rewards}

        self.save()
        return results

    def predict(self, draws):
        """Kombinierte Vorhersage aus allen 5 ML-Modellen"""
        predictions = {}

        # 1. Neural Network
        nn_main, nn_euro, nn_conf = self.nn.predict(draws)
        predictions['neural_network'] = {
            'numbers': nn_main, 'eurozahlen': nn_euro,
            'confidence': nn_conf, 'weight': self.model_weights.get('neural_network', 1.0)
        }

        # 2. Markov
        markov_main, markov_euro, markov_conf = self.markov.predict(draws[0] if draws else {})
        predictions['markov'] = {
            'numbers': markov_main, 'eurozahlen': markov_euro,
            'confidence': markov_conf, 'weight': self.model_weights.get('markov', 1.0)
        }

        # 3. Bayesian
        bayes_main, bayes_euro, bayes_conf = self.bayesian.predict('thompson')
        predictions['bayesian'] = {
            'numbers': bayes_main, 'eurozahlen': bayes_euro,
            'confidence': bayes_conf, 'weight': self.model_weights.get('bayesian', 1.0)
        }

        # 4. Eurozahl ML (für bessere Eurozahlen-Vorhersage)
        eurozahl_ml = self._init_eurozahl_ml()
        last_euro = draws[0].get('eurozahlen') if draws else None
        ez_best, ez_ranking, ez_conf, ez_pairs = eurozahl_ml.predict(last_euro)
        predictions['eurozahl_ml'] = {
            'eurozahlen': ez_best,
            'confidence': ez_conf,
            'ranking': ez_ranking[:5],
            'top_pairs': ez_pairs,
            'weight': self.model_weights.get('eurozahl_ml', 1.0)
        }

        # 5. Reinforcement Learner (Strategie-Empfehlungen)
        rl = self._init_reinforcement()
        rl_result = rl.predict(draws)
        predictions['reinforcement'] = {
            'recommended_strategies': rl_result['recommended_strategies'],
            'weight': self.model_weights.get('reinforcement', 1.0)
        }

        # Gewichtetes Voting für Hauptzahlen
        main_votes = Counter()
        for model_name in ['neural_network', 'markov', 'bayesian']:
            pred = predictions[model_name]
            weight = pred['weight']
            for num in pred['numbers']:
                main_votes[num] += weight

        # Gewichtetes Voting für Eurozahlen (inkl. Eurozahl ML!)
        euro_votes = Counter()
        for model_name in ['neural_network', 'markov', 'bayesian']:
            pred = predictions[model_name]
            weight = pred['weight']
            for ez in pred['eurozahlen']:
                euro_votes[ez] += weight

        # Eurozahl ML bekommt extra Gewicht (spezialisiert!)
        ez_weight = self.model_weights.get('eurozahl_ml', 1.0) * 1.5
        for ez in predictions['eurozahl_ml']['eurozahlen']:
            euro_votes[ez] += ez_weight

        ensemble_main = sorted([num for num, _ in main_votes.most_common(5)])
        ensemble_euro = sorted([ez for ez, _ in euro_votes.most_common(2)])

        return {
            'ensemble': {'numbers': ensemble_main, 'eurozahlen': ensemble_euro},
            'individual': predictions,
            'model_weights': self.model_weights,
            'rl_strategies': rl_result['recommended_strategies']
        }

    def learn_from_result(self, prediction, actual_draw):
        """Lernt aus einem Ergebnis (für alle Modelle)"""
        # Reinforcement Learning Update
        rl = self._init_reinforcement()
        rl_result = rl.learn_from_result(prediction, actual_draw)

        # Eurozahl ML Update
        eurozahl_ml = self._init_eurozahl_ml()
        prev_euro = prediction.get('eurozahlen', [])
        new_euro = actual_draw.get('eurozahlen', [])
        eurozahl_ml.train_on_new_draw(prev_euro, new_euro)

        # Model Performance Update
        main_matches = len(set(prediction.get('numbers', [])) & set(actual_draw.get('numbers', [])))
        euro_matches = len(set(prediction.get('eurozahlen', [])) & set(actual_draw.get('eurozahlen', [])))

        method = prediction.get('method', 'unknown')
        if method not in self.model_performance:
            self.model_performance[method] = {'predictions': 0, 'main_matches': 0, 'euro_matches': 0}

        self.model_performance[method]['predictions'] += 1
        self.model_performance[method]['main_matches'] += main_matches
        self.model_performance[method]['euro_matches'] += euro_matches

        self.save()

        return {
            'rl_result': rl_result,
            'main_matches': main_matches,
            'euro_matches': euro_matches
        }

    def save(self):
        data = {
            'model_weights': self.model_weights,
            'model_performance': self.model_performance,
            'last_updated': datetime.now().isoformat()
        }
        save_model(self.MODEL_FILE, data)


# =====================================================
# EUROJACKPOT REINFORCEMENT LEARNER (Q-Learning)
# =====================================================

class EurojackpotReinforcementLearner:
    """
    Q-Learning für Eurojackpot Strategie-Optimierung.

    States: Kombinationen von Zahlen-Features
    Actions: Welche Strategie wählen
    Rewards: Basierend auf Treffern (0-5 Hauptzahlen + 0-2 Eurozahlen)
    """

    MODEL_FILE = 'eurojackpot_reinforcement_learner.json'

    def __init__(self):
        saved = load_model(self.MODEL_FILE)

        if saved and 'q_table' in saved:
            self.q_table = saved['q_table']
            self.strategy_values = np.array(saved.get('strategy_values', np.zeros(18)))
            self.epsilon = saved.get('epsilon', 0.3)
            self.learning_rate = saved.get('learning_rate', 0.1)
            self.discount_factor = saved.get('discount_factor', 0.95)
            self.total_rewards = saved.get('total_rewards', 0)
            self.episodes = saved.get('episodes', 0)
        else:
            self.q_table = {}
            self.strategy_values = np.zeros(18)  # 18 Strategien für Eurojackpot
            self.epsilon = 0.3  # Exploration Rate
            self.learning_rate = 0.1
            self.discount_factor = 0.95
            self.total_rewards = 0
            self.episodes = 0

        # Eurojackpot Strategie-Namen
        self.strategies = [
            'ej_hot_cold', 'ej_cold', 'ej_overdue', 'ej_odd_even_32', 'ej_odd_even_23',
            'ej_sum_optimized', 'ej_decade_balance', 'ej_delta', 'ej_position',
            'ej_no_consecutive', 'ej_prime_mix', 'ej_low_high', 'ej_hot_cold_mix',
            'ej_monte_carlo', 'ej_bayesian', 'ml_neural_network', 'ml_markov', 'ml_ensemble'
        ]

    def get_state_key(self, features):
        """Erstellt einen State-Key aus Features"""
        discretized = tuple(int(f * 10) for f in features[:10])
        return str(discretized)

    def calculate_reward(self, main_matches, euro_matches):
        """Berechnet Reward basierend auf Treffern (Eurojackpot Gewinnklassen)"""
        # Eurojackpot Gewinnklassen-basiertes Reward System
        if main_matches == 5 and euro_matches == 2:
            return 1000  # Jackpot
        elif main_matches == 5 and euro_matches == 1:
            return 500
        elif main_matches == 5 and euro_matches == 0:
            return 200
        elif main_matches == 4 and euro_matches == 2:
            return 100
        elif main_matches == 4 and euro_matches == 1:
            return 50
        elif main_matches == 4 and euro_matches == 0:
            return 20
        elif main_matches == 3 and euro_matches == 2:
            return 15
        elif main_matches == 2 and euro_matches == 2:
            return 10
        elif main_matches == 3 and euro_matches == 1:
            return 8
        elif main_matches == 3 and euro_matches == 0:
            return 5
        elif main_matches == 1 and euro_matches == 2:
            return 4
        elif main_matches == 2 and euro_matches == 1:
            return 3
        elif main_matches == 2 and euro_matches == 0:
            return 1
        elif main_matches == 1 and euro_matches == 1:
            return 0.5
        else:
            return -1  # Keine Treffer

    def choose_action(self, state_key):
        """Wählt Aktion (Strategie) mit Epsilon-Greedy"""
        if np.random.random() < self.epsilon:
            return np.random.randint(0, len(self.strategies))
        else:
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

        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )

        self.q_table[state_key][action] = new_q
        self.strategy_values[action] = (
            0.9 * self.strategy_values[action] + 0.1 * reward
        )

    def learn_from_result(self, prediction, actual_draw, features=None):
        """Lernt aus einem Vorhersage-Ergebnis"""
        predicted_main = set(prediction.get('numbers', []))
        actual_main = set(actual_draw.get('numbers', []))
        predicted_euro = set(prediction.get('eurozahlen', []))
        actual_euro = set(actual_draw.get('eurozahlen', []))

        main_matches = len(predicted_main & actual_main)
        euro_matches = len(predicted_euro & actual_euro)

        reward = self.calculate_reward(main_matches, euro_matches)

        # Finde welche Strategie verwendet wurde
        method = prediction.get('method', '')
        action = 0
        for i, strat in enumerate(self.strategies):
            if strat in method or method in strat:
                action = i
                break

        # Erstelle State-Key aus Features oder Standard
        if features is not None:
            state_key = self.get_state_key(features)
        else:
            state_key = f"default_{main_matches}_{euro_matches}"

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
            'main_matches': main_matches,
            'euro_matches': euro_matches,
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
            if strat == strategy_name or strat in strategy_name or strategy_name in strat:
                value = self.strategy_values[i]
                weight = 1.0 + (value / 100)
                return max(0.5, min(2.0, weight))
        return 1.0

    def predict(self, draws):
        """Gibt Strategie-Empfehlungen basierend auf Q-Learning"""
        best_strategies = self.get_best_strategies(5)

        return {
            'recommended_strategies': best_strategies,
            'strategy_values': {s: float(v) for s, v in zip(self.strategies, self.strategy_values)},
            'total_episodes': self.episodes,
            'total_rewards': self.total_rewards
        }

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
# EUROZAHL ML (Spezialisiert auf 2 aus 12)
# =====================================================

class EurozahlML:
    """
    Spezialisiertes ML-Modell für die 2 Eurozahlen (1-12).

    Verwendet:
    - Zeitreihen-Analyse
    - Pattern Recognition (Paare)
    - Bayesian Learning
    - Markov für Paar-Übergänge
    """

    MODEL_FILE = 'eurozahl_ml.json'

    def __init__(self):
        saved = load_model(self.MODEL_FILE)

        if saved and 'weights' in saved:
            self.weights = np.array(saved['weights'])
            self.pair_weights = np.array(saved.get('pair_weights', np.ones((12, 12)) / 144))
            self.transition_probs = np.array(saved.get('transition_probs', np.ones((12, 12)) / 12))
            self.alpha = np.array(saved.get('alpha', np.ones(12)))
            self.beta = np.array(saved.get('beta', np.ones(12)))
            self.observations = saved.get('observations', 0)
        else:
            self.weights = np.ones(12) / 12  # Gleichverteilung für 1-12
            self.pair_weights = np.ones((12, 12)) / 144  # Paar-Häufigkeiten
            self.transition_probs = np.ones((12, 12)) / 12
            self.alpha = np.ones(12)  # Bayesian Prior
            self.beta = np.ones(12)
            self.observations = 0

    def train(self, draws):
        """Trainiert mit historischen Eurozahlen"""
        freq = np.zeros(12)
        pair_freq = np.ones((12, 12))  # Laplace Smoothing
        transitions = np.ones((12, 12))

        prev_pair = None
        for draw in draws:
            eurozahlen = draw.get('eurozahlen', [])
            if len(eurozahlen) >= 2:
                ez1, ez2 = eurozahlen[0], eurozahlen[1]

                # Index 0-11 (Eurozahlen 1-12)
                if 1 <= ez1 <= 12 and 1 <= ez2 <= 12:
                    idx1, idx2 = ez1 - 1, ez2 - 1

                    freq[idx1] += 1
                    freq[idx2] += 1

                    # Bayesian Update
                    self.alpha[idx1] += 1
                    self.alpha[idx2] += 1
                    for i in range(12):
                        if i != idx1 and i != idx2:
                            self.beta[i] += 0.5

                    # Paar-Häufigkeit
                    pair_freq[idx1][idx2] += 1
                    pair_freq[idx2][idx1] += 1

                    # Transitions
                    if prev_pair is not None:
                        transitions[prev_pair[0]][idx1] += 1
                        transitions[prev_pair[1]][idx2] += 1

                    prev_pair = (idx1, idx2)

        # Normalisiere
        if np.sum(freq) > 0:
            self.weights = freq / np.sum(freq)

        pair_sum = np.sum(pair_freq)
        if pair_sum > 0:
            self.pair_weights = pair_freq / pair_sum

        row_sums = transitions.sum(axis=1, keepdims=True)
        self.transition_probs = transitions / row_sums

        self.observations = len(draws)
        self.save()

        return {'observations': self.observations}

    def train_on_new_draw(self, previous_euro, new_euro):
        """Inkrementelles Update"""
        if len(new_euro) < 2:
            return

        ez1, ez2 = new_euro[0], new_euro[1]
        if not (1 <= ez1 <= 12 and 1 <= ez2 <= 12):
            return

        idx1, idx2 = ez1 - 1, ez2 - 1

        # Update Bayesian
        self.alpha[idx1] += 1
        self.alpha[idx2] += 1
        for i in range(12):
            if i != idx1 and i != idx2:
                self.beta[i] += 0.1

        # Update Pair Weights
        self.pair_weights[idx1][idx2] += 0.1
        self.pair_weights[idx2][idx1] += 0.1
        self.pair_weights = self.pair_weights / np.sum(self.pair_weights)

        # Update Transition
        if previous_euro and len(previous_euro) >= 2:
            prev1, prev2 = previous_euro[0] - 1, previous_euro[1] - 1
            if 0 <= prev1 < 12 and 0 <= prev2 < 12:
                self.transition_probs[prev1][idx1] += 0.1
                self.transition_probs[prev2][idx2] += 0.1
                row_sums = self.transition_probs.sum(axis=1, keepdims=True)
                self.transition_probs = self.transition_probs / row_sums

        # Update Gewichte
        self.weights[idx1] = 0.95 * self.weights[idx1] + 0.05
        self.weights[idx2] = 0.95 * self.weights[idx2] + 0.05
        self.weights = self.weights / np.sum(self.weights)

        self.observations += 1
        self.save()

    def predict(self, last_euro=None):
        """Vorhersage der nächsten 2 Eurozahlen"""
        scores = np.zeros(12)

        # 1. Bayesian Posterior (35%)
        posterior = self.alpha / (self.alpha + self.beta)
        scores += 0.35 * posterior

        # 2. Transition Probability (25%)
        if last_euro and len(last_euro) >= 2:
            idx1, idx2 = last_euro[0] - 1, last_euro[1] - 1
            if 0 <= idx1 < 12 and 0 <= idx2 < 12:
                trans_scores = (self.transition_probs[idx1] + self.transition_probs[idx2]) / 2
                scores += 0.25 * trans_scores
        else:
            scores += 0.25 * np.mean(self.transition_probs, axis=0)

        # 3. Häufigkeitsgewichte (25%)
        scores += 0.25 * self.weights

        # 4. Thompson Sampling für Exploration (15%)
        thompson_samples = np.random.beta(self.alpha, self.beta)
        scores += 0.15 * thompson_samples

        # Normalisieren
        scores = scores / np.sum(scores)

        # Ranking für einzelne Eurozahlen
        ranking = [(i + 1, float(scores[i])) for i in range(12)]
        ranking.sort(key=lambda x: x[1], reverse=True)

        # Beste 2 Eurozahlen (nicht gleich!)
        best_euro = [ranking[0][0], ranking[1][0]]
        confidence = (ranking[0][1] + ranking[1][1]) * 50

        # Auch beste Paare analysieren
        pair_scores = []
        for i in range(12):
            for j in range(i + 1, 12):
                pair_score = self.pair_weights[i][j] + scores[i] + scores[j]
                pair_scores.append(((i + 1, j + 1), pair_score))

        pair_scores.sort(key=lambda x: x[1], reverse=True)
        top_pairs = pair_scores[:5]

        return sorted(best_euro), ranking, confidence, top_pairs

    def save(self):
        """Speichert das Modell"""
        data = {
            'weights': self.weights,
            'pair_weights': self.pair_weights,
            'transition_probs': self.transition_probs,
            'alpha': self.alpha,
            'beta': self.beta,
            'observations': self.observations,
            'last_updated': datetime.now().isoformat()
        }
        save_model(self.MODEL_FILE, data)


# =====================================================
# ZIFFERN-BASIERTE SPIELE ML (Spiel 77, Super 6, Glücksspirale)
# =====================================================

class DigitNeuralNetwork:
    """
    Neuronales Netz für ziffernbasierte Spiele.

    Arbeitet pro Position: Jede Position hat ihr eigenes Mini-Netz.
    """

    def __init__(self, game_name, num_digits):
        self.game_name = game_name
        self.num_digits = num_digits
        self.MODEL_FILE = f'{game_name}_neural_network.json'

        saved = load_model(self.MODEL_FILE)

        if saved and 'weights' in saved:
            self.position_weights = {}
            for pos in range(num_digits):
                pos_key = f'pos_{pos}'
                if pos_key in saved['weights']:
                    self.position_weights[pos] = {
                        'W1': np.array(saved['weights'][pos_key]['W1']),
                        'b1': np.array(saved['weights'][pos_key]['b1']),
                        'W2': np.array(saved['weights'][pos_key]['W2']),
                        'b2': np.array(saved['weights'][pos_key]['b2'])
                    }
                else:
                    self._init_position_weights(pos)
            self.epochs_trained = saved.get('epochs_trained', 0)
            self.learning_rate = saved.get('learning_rate', 0.01)
        else:
            self.position_weights = {}
            for pos in range(num_digits):
                self._init_position_weights(pos)
            self.epochs_trained = 0
            self.learning_rate = 0.01

    def _init_position_weights(self, pos):
        """Initialisiert Gewichte für eine Position (10 -> 16 -> 10)"""
        self.position_weights[pos] = {
            'W1': np.random.randn(10, 16) * np.sqrt(2.0 / 10),
            'b1': np.zeros((1, 16)),
            'W2': np.random.randn(16, 10) * np.sqrt(2.0 / 16),
            'b2': np.zeros((1, 10))
        }

    def relu(self, x):
        return np.clip(np.maximum(0, x), 0, 100)

    def softmax(self, x):
        x_clipped = np.clip(x, -100, 100)
        x_shifted = x_clipped - np.max(x_clipped, axis=1, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / (np.sum(exp_x, axis=1, keepdims=True) + 1e-10)

    def forward(self, X, pos):
        """Forward pass für eine Position"""
        w = self.position_weights[pos]
        z1 = np.dot(X, w['W1']) + w['b1']
        a1 = self.relu(z1)
        z2 = np.dot(a1, w['W2']) + w['b2']
        return self.softmax(z2)

    def get_digits(self, number_str):
        """Konvertiert Zahl in Ziffern-Liste"""
        return [int(d) for d in str(number_str).zfill(self.num_digits)]

    def train(self, draws, epochs=100):
        """Trainiert das Netzwerk"""
        if len(draws) < 10:
            return {'error': 'Nicht genug Daten'}

        # Erstelle Features pro Position
        for pos in range(self.num_digits):
            X = []  # Häufigkeiten der letzten N Ziehungen
            y = []  # Nächste Ziffer

            for i in range(len(draws) - 1):
                # Feature: Häufigkeit der Ziffern in letzten 20 Ziehungen für diese Position
                freq = np.zeros(10)
                for j in range(i, min(i + 20, len(draws))):
                    digits = self.get_digits(draws[j]['number'])
                    freq[digits[pos]] += 1

                if np.sum(freq) > 0:
                    freq = freq / np.sum(freq)
                X.append(freq)

                # Label: Ziffer der nächsten Ziehung
                next_digits = self.get_digits(draws[i]['number'])
                y_vec = np.zeros(10)
                y_vec[next_digits[pos]] = 1
                y.append(y_vec)

            X = np.array(X)
            y = np.array(y)

            if len(X) < 2:
                continue

            # Training
            w = self.position_weights[pos]
            for epoch in range(epochs):
                # Forward
                z1 = np.dot(X, w['W1']) + w['b1']
                a1 = self.relu(z1)
                z2 = np.dot(a1, w['W2']) + w['b2']
                output = self.softmax(z2)

                # Backward
                dz2 = output - y
                dW2 = np.clip(np.dot(a1.T, dz2) / len(X), -1, 1)
                db2 = np.clip(np.sum(dz2, axis=0, keepdims=True) / len(X), -1, 1)

                dz1 = np.dot(dz2, w['W2'].T) * (z1 > 0).astype(float)
                dW1 = np.clip(np.dot(X.T, dz1) / len(X), -1, 1)
                db1 = np.clip(np.sum(dz1, axis=0, keepdims=True) / len(X), -1, 1)

                w['W2'] -= self.learning_rate * dW2
                w['b2'] -= self.learning_rate * db2
                w['W1'] -= self.learning_rate * dW1
                w['b1'] -= self.learning_rate * db1

                # Clip weights
                w['W1'] = np.clip(w['W1'], -5, 5)
                w['W2'] = np.clip(w['W2'], -5, 5)

        self.epochs_trained += epochs
        self.save()

        return {'epochs': epochs, 'total_epochs': self.epochs_trained}

    def predict(self, draws):
        """Vorhersage der nächsten Zahl"""
        predicted_digits = []
        confidences = []

        for pos in range(self.num_digits):
            # Feature für diese Position
            freq = np.zeros(10)
            for j in range(min(20, len(draws))):
                digits = self.get_digits(draws[j]['number'])
                freq[digits[pos]] += 1

            if np.sum(freq) > 0:
                freq = freq / np.sum(freq)

            X = freq.reshape(1, -1)
            probs = self.forward(X, pos)[0]

            best_digit = np.argmax(probs)
            predicted_digits.append(best_digit)
            confidences.append(float(probs[best_digit]))

        number = ''.join(str(d) for d in predicted_digits)
        confidence = float(np.mean(confidences) * 100)

        return number, confidence

    def save(self):
        weights_data = {}
        for pos in range(self.num_digits):
            weights_data[f'pos_{pos}'] = self.position_weights[pos]

        data = {
            'weights': weights_data,
            'epochs_trained': self.epochs_trained,
            'learning_rate': self.learning_rate,
            'game_name': self.game_name,
            'num_digits': self.num_digits,
            'last_updated': datetime.now().isoformat()
        }
        save_model(self.MODEL_FILE, data)


class DigitBayesian:
    """Bayesian Predictor für ziffernbasierte Spiele"""

    def __init__(self, game_name, num_digits):
        self.game_name = game_name
        self.num_digits = num_digits
        self.MODEL_FILE = f'{game_name}_bayesian.json'

        saved = load_model(self.MODEL_FILE)

        if saved and 'alpha' in saved:
            self.alpha = {int(k): np.array(v) for k, v in saved['alpha'].items()}
            self.beta = {int(k): np.array(v) for k, v in saved['beta'].items()}
            self.observations = saved.get('observations', 0)
        else:
            self.alpha = {pos: np.ones(10) for pos in range(num_digits)}
            self.beta = {pos: np.ones(10) for pos in range(num_digits)}
            self.observations = 0

    def get_digits(self, number_str):
        return [int(d) for d in str(number_str).zfill(self.num_digits)]

    def train(self, draws):
        """Trainiert mit historischen Daten"""
        for draw in draws:
            digits = self.get_digits(draw['number'])
            for pos, digit in enumerate(digits):
                self.alpha[pos][digit] += 1
                for d in range(10):
                    if d != digit:
                        self.beta[pos][d] += 0.1

        self.observations = len(draws)
        self.save()
        return {'observations': self.observations}

    def predict(self, method='thompson'):
        """Vorhersage mit Thompson Sampling"""
        predicted_digits = []

        for pos in range(self.num_digits):
            if method == 'thompson':
                samples = [np.random.beta(self.alpha[pos][d], self.beta[pos][d]) for d in range(10)]
                best_digit = np.argmax(samples)
            else:  # MAP
                probs = self.alpha[pos] / (self.alpha[pos] + self.beta[pos])
                best_digit = np.argmax(probs)

            predicted_digits.append(best_digit)

        number = ''.join(str(d) for d in predicted_digits)
        confidence = 70 if method == 'thompson' else 75

        return number, confidence

    def save(self):
        data = {
            'alpha': {str(k): v.tolist() for k, v in self.alpha.items()},
            'beta': {str(k): v.tolist() for k, v in self.beta.items()},
            'observations': self.observations,
            'game_name': self.game_name,
            'last_updated': datetime.now().isoformat()
        }
        save_model(self.MODEL_FILE, data)


class DigitMarkov:
    """Markov-Kette für ziffernbasierte Spiele"""

    def __init__(self, game_name, num_digits):
        self.game_name = game_name
        self.num_digits = num_digits
        self.MODEL_FILE = f'{game_name}_markov.json'

        saved = load_model(self.MODEL_FILE)

        if saved and 'transitions' in saved:
            self.transitions = {int(k): np.array(v) for k, v in saved['transitions'].items()}
            self.observations = saved.get('observations', 0)
        else:
            self.transitions = {pos: np.ones((10, 10)) / 10 for pos in range(num_digits)}
            self.observations = 0

    def get_digits(self, number_str):
        return [int(d) for d in str(number_str).zfill(self.num_digits)]

    def train(self, draws):
        """Trainiert Übergangswahrscheinlichkeiten pro Position"""
        counts = {pos: np.ones((10, 10)) for pos in range(self.num_digits)}

        for i in range(len(draws) - 1):
            current_digits = self.get_digits(draws[i + 1]['number'])
            next_digits = self.get_digits(draws[i]['number'])

            for pos in range(self.num_digits):
                counts[pos][current_digits[pos]][next_digits[pos]] += 1

        for pos in range(self.num_digits):
            row_sums = counts[pos].sum(axis=1, keepdims=True)
            self.transitions[pos] = counts[pos] / row_sums

        self.observations = len(draws)
        self.save()
        return {'observations': self.observations}

    def predict(self, last_draw):
        """Vorhersage basierend auf letzter Ziehung"""
        last_digits = self.get_digits(last_draw['number'])
        predicted_digits = []

        for pos in range(self.num_digits):
            probs = self.transitions[pos][last_digits[pos]]
            # Reduziere Wahrscheinlichkeit für gleiche Ziffer
            probs[last_digits[pos]] *= 0.7
            probs = probs / np.sum(probs)

            best_digit = np.argmax(probs)
            predicted_digits.append(best_digit)

        number = ''.join(str(d) for d in predicted_digits)
        confidence = 65

        return number, confidence

    def save(self):
        data = {
            'transitions': {str(k): v.tolist() for k, v in self.transitions.items()},
            'observations': self.observations,
            'game_name': self.game_name,
            'last_updated': datetime.now().isoformat()
        }
        save_model(self.MODEL_FILE, data)


class DigitReinforcementLearner:
    """
    Q-Learning Reinforcement Learner für ziffernbasierte Spiele.

    Lernt welche Strategien/Ziffernmuster am besten funktionieren
    und passt Gewichte basierend auf Belohnungen an.
    """

    def __init__(self, game_name, num_digits):
        self.game_name = game_name
        self.num_digits = num_digits
        self.MODEL_FILE = f'{game_name}_reinforcement.json'

        # Strategien für Ziffernspiele
        self.strategies = [
            'hot_digits', 'cold_digits', 'balanced',
            'odd_even', 'high_low', 'sequential',
            'random_weighted', 'position_frequency',
            'transition_based', 'pattern_based'
        ]

        # Q-Learning Parameter
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.15  # Exploration rate

        saved = load_model(self.MODEL_FILE)

        if saved and 'q_table' in saved:
            self.q_table = np.array(saved['q_table'])
            self.strategy_stats = saved.get('strategy_stats', {})
            self.total_rewards = saved.get('total_rewards', 0)
            self.episodes = saved.get('episodes', 0)
        else:
            # Q-Table: States (10 Zustände) x Actions (10 Strategien)
            self.q_table = np.zeros((10, len(self.strategies)))
            self.strategy_stats = {s: {'uses': 0, 'rewards': 0} for s in self.strategies}
            self.total_rewards = 0
            self.episodes = 0

    def get_state(self, draws):
        """Berechnet aktuellen Zustand aus historischen Daten."""
        if not draws:
            return 0

        # Zustand basierend auf letzter Ziffer der letzten Ziehung
        last_draw = draws[0] if draws else {'number': '0' * self.num_digits}
        last_digit = int(str(last_draw.get('number', '0'))[-1])
        return last_digit

    def select_action(self, state):
        """Epsilon-Greedy Action Selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(len(self.strategies))
        return np.argmax(self.q_table[state])

    def calculate_reward(self, matches):
        """Berechnet Belohnung basierend auf Endziffern-Übereinstimmung."""
        # Ziffernspiele: Endziffern von rechts nach links
        rewards = {
            7: 100,   # Alle 7 Ziffern (Spiel77/Glücksspirale)
            6: 50,    # 6 Ziffern
            5: 20,    # 5 Ziffern
            4: 10,    # 4 Ziffern
            3: 5,     # 3 Ziffern
            2: 2,     # 2 Ziffern
            1: 1,     # 1 Ziffer
            0: -0.5   # Keine Übereinstimmung
        }
        return rewards.get(matches, 0)

    def update_q_value(self, state, action, reward, next_state):
        """Q-Learning Update."""
        current_q = self.q_table[state, action]
        max_next_q = np.max(self.q_table[next_state])
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        self.q_table[state, action] = new_q

    def learn_from_result(self, state, action, matches, next_state):
        """Lernt aus einem Ergebnis."""
        reward = self.calculate_reward(matches)
        self.update_q_value(state, action, reward, next_state)

        strategy_name = self.strategies[action]
        self.strategy_stats[strategy_name]['uses'] += 1
        self.strategy_stats[strategy_name]['rewards'] += reward
        self.total_rewards += reward
        self.episodes += 1

        # Decay epsilon
        self.epsilon = max(0.05, self.epsilon * 0.995)

        self.save()

    def generate_number(self, draws, strategy_idx=None):
        """Generiert eine Zahl basierend auf gewählter Strategie."""
        if strategy_idx is None:
            state = self.get_state(draws)
            strategy_idx = self.select_action(state)

        strategy = self.strategies[strategy_idx]

        # Berechne Ziffernhäufigkeiten pro Position
        position_freqs = {pos: np.zeros(10) for pos in range(self.num_digits)}
        for draw in draws[:100]:
            digits = [int(d) for d in str(draw.get('number', '0' * self.num_digits)).zfill(self.num_digits)]
            for pos, digit in enumerate(digits):
                position_freqs[pos][digit] += 1

        # Normalisieren
        for pos in position_freqs:
            total = position_freqs[pos].sum()
            if total > 0:
                position_freqs[pos] /= total

        result_digits = []

        for pos in range(self.num_digits):
            freqs = position_freqs[pos]

            if strategy == 'hot_digits':
                # Wähle häufigste Ziffer
                digit = np.argmax(freqs)
            elif strategy == 'cold_digits':
                # Wähle seltenste Ziffer
                digit = np.argmin(freqs + (freqs == 0) * 1000)
            elif strategy == 'balanced':
                # Wähle Ziffer nahe am Durchschnitt
                target = 4.5
                digit = min(range(10), key=lambda x: abs(x - target))
            elif strategy == 'odd_even':
                # Alternierend gerade/ungerade
                if pos % 2 == 0:
                    evens = [0, 2, 4, 6, 8]
                    weights = [freqs[d] for d in evens]
                    if sum(weights) > 0:
                        digit = random.choices(evens, weights=weights)[0]
                    else:
                        digit = random.choice(evens)
                else:
                    odds = [1, 3, 5, 7, 9]
                    weights = [freqs[d] for d in odds]
                    if sum(weights) > 0:
                        digit = random.choices(odds, weights=weights)[0]
                    else:
                        digit = random.choice(odds)
            elif strategy == 'high_low':
                # Alternierend hoch/niedrig
                if pos % 2 == 0:
                    lows = [0, 1, 2, 3, 4]
                    weights = [freqs[d] for d in lows]
                    if sum(weights) > 0:
                        digit = random.choices(lows, weights=weights)[0]
                    else:
                        digit = random.choice(lows)
                else:
                    highs = [5, 6, 7, 8, 9]
                    weights = [freqs[d] for d in highs]
                    if sum(weights) > 0:
                        digit = random.choices(highs, weights=weights)[0]
                    else:
                        digit = random.choice(highs)
            elif strategy == 'sequential':
                # Aufsteigende Sequenz mit Variation
                base = (pos * 2) % 10
                digit = (base + random.randint(-1, 1)) % 10
            elif strategy == 'random_weighted':
                # Zufällig mit Häufigkeitsgewichtung
                weights = freqs + 0.1  # Kleine Basis hinzufügen
                digit = random.choices(range(10), weights=weights)[0]
            elif strategy == 'position_frequency':
                # Häufigste für diese Position
                digit = np.argmax(freqs)
            elif strategy == 'transition_based':
                # Basierend auf Übergängen
                if draws and pos > 0:
                    last_digit = int(str(draws[0].get('number', '0' * self.num_digits)).zfill(self.num_digits)[pos-1])
                    digit = (last_digit + random.randint(1, 3)) % 10
                else:
                    digit = random.randint(0, 9)
            elif strategy == 'pattern_based':
                # Muster aus letzten Ziehungen
                if len(draws) >= 3:
                    recent_digits = []
                    for draw in draws[:3]:
                        d = int(str(draw.get('number', '0' * self.num_digits)).zfill(self.num_digits)[pos])
                        recent_digits.append(d)
                    # Trend fortsetzen
                    if len(set(recent_digits)) == 1:
                        digit = (recent_digits[0] + 1) % 10
                    else:
                        digit = (sum(recent_digits) // len(recent_digits)) % 10
                else:
                    digit = random.randint(0, 9)
            else:
                digit = random.randint(0, 9)

            result_digits.append(digit)

        number = ''.join(str(d) for d in result_digits)
        return number, strategy

    def predict(self, draws):
        """Generiert Vorhersage mit bester Strategie."""
        state = self.get_state(draws)
        best_action = np.argmax(self.q_table[state])
        number, strategy = self.generate_number(draws, best_action)

        return number, 70, strategy

    def get_best_strategies(self, top_n=3):
        """Gibt die besten Strategien zurück."""
        avg_rewards = []
        for s in self.strategies:
            stats = self.strategy_stats.get(s, {'uses': 0, 'rewards': 0})
            if stats['uses'] > 0:
                avg_rewards.append((s, stats['rewards'] / stats['uses']))
            else:
                avg_rewards.append((s, 0))

        avg_rewards.sort(key=lambda x: x[1], reverse=True)
        return avg_rewards[:top_n]

    def save(self):
        data = {
            'q_table': self.q_table.tolist(),
            'strategy_stats': self.strategy_stats,
            'total_rewards': self.total_rewards,
            'episodes': self.episodes,
            'epsilon': self.epsilon,
            'game_name': self.game_name,
            'num_digits': self.num_digits,
            'last_updated': datetime.now().isoformat()
        }
        save_model(self.MODEL_FILE, data)


class DigitGameEnsembleML:
    """
    Ensemble ML für ziffernbasierte Spiele (4 ML-Algorithmen).

    Kombiniert:
    1. Neural Network (Backpropagation pro Position)
    2. Markov Chain (Übergangswahrscheinlichkeiten)
    3. Bayesian Predictor (Thompson Sampling)
    4. Reinforcement Learner (Q-Learning)
    """

    def __init__(self, game_name, num_digits):
        self.game_name = game_name
        self.num_digits = num_digits
        self.MODEL_FILE = f'{game_name}_ensemble_ml.json'

        saved = load_model(self.MODEL_FILE)

        if saved and 'model_weights' in saved:
            self.model_weights = saved['model_weights']
        else:
            self.model_weights = {
                'neural_network': 1.0,
                'markov': 1.0,
                'bayesian': 1.0,
                'reinforcement': 1.0
            }

        # Initialisiere alle 4 ML-Modelle
        self.nn = DigitNeuralNetwork(game_name, num_digits)
        self.markov = DigitMarkov(game_name, num_digits)
        self.bayesian = DigitBayesian(game_name, num_digits)
        self.reinforcement = DigitReinforcementLearner(game_name, num_digits)

    def train_all(self, draws):
        """Trainiert alle 4 ML-Modelle"""
        results = {}

        print(f"   🧠 Training {self.game_name} Neural Network...")
        results['neural_network'] = self.nn.train(draws)

        print(f"   🔗 Training {self.game_name} Markov Chain...")
        results['markov'] = self.markov.train(draws)

        print(f"   📊 Training {self.game_name} Bayesian...")
        results['bayesian'] = self.bayesian.train(draws)

        print(f"   🎯 Training {self.game_name} Reinforcement Learner...")
        # RL lernt während der Vorhersage-Auswertung
        results['reinforcement'] = {'status': 'ready', 'episodes': self.reinforcement.episodes}

        self.save()
        return results

    def predict(self, draws):
        """Kombinierte Vorhersage aus allen 4 Modellen"""
        predictions = {}

        # 1. Neural Network
        nn_num, nn_conf = self.nn.predict(draws)
        predictions['neural_network'] = {
            'number': nn_num, 'confidence': nn_conf,
            'weight': self.model_weights.get('neural_network', 1.0)
        }

        # 2. Markov
        markov_num, markov_conf = self.markov.predict(draws[0] if draws else {'number': '0' * self.num_digits})
        predictions['markov'] = {
            'number': markov_num, 'confidence': markov_conf,
            'weight': self.model_weights.get('markov', 1.0)
        }

        # 3. Bayesian
        bayes_num, bayes_conf = self.bayesian.predict('thompson')
        predictions['bayesian'] = {
            'number': bayes_num, 'confidence': bayes_conf,
            'weight': self.model_weights.get('bayesian', 1.0)
        }

        # 4. Reinforcement Learner
        rl_num, rl_conf, rl_strategy = self.reinforcement.predict(draws)
        predictions['reinforcement'] = {
            'number': rl_num, 'confidence': rl_conf,
            'weight': self.model_weights.get('reinforcement', 1.0),
            'strategy': rl_strategy
        }

        # Gewichtetes Voting pro Position
        ensemble_digits = []
        for pos in range(self.num_digits):
            votes = Counter()
            for model_name, pred in predictions.items():
                weight = pred['weight']
                digit = int(pred['number'][pos])
                votes[digit] += weight

            best_digit = votes.most_common(1)[0][0]
            ensemble_digits.append(best_digit)

        ensemble_number = ''.join(str(d) for d in ensemble_digits)

        return {
            'ensemble': ensemble_number,
            'individual': predictions,
            'model_weights': self.model_weights
        }

    def update_weights(self, model_name, matches):
        """Aktualisiert Modellgewichte basierend auf Ergebnis."""
        if model_name in self.model_weights:
            if matches >= 3:
                self.model_weights[model_name] = min(2.0, self.model_weights[model_name] * 1.05)
            elif matches == 0:
                self.model_weights[model_name] = max(0.5, self.model_weights[model_name] * 0.98)
            self.save()

    def save(self):
        data = {
            'model_weights': self.model_weights,
            'game_name': self.game_name,
            'num_digits': self.num_digits,
            'last_updated': datetime.now().isoformat()
        }
        save_model(self.MODEL_FILE, data)


# =====================================================
# GLOBAL META-LEARNER - LERNT VON ALLEN SPIELEN
# =====================================================

class GlobalMetaLearner:
    """
    Globaler Meta-Learner der von ALLEN Spielen lernt.

    Dieses Modell analysiert die Performance aller Strategien und ML-Modelle
    über alle 5 Spiele hinweg und identifiziert globale Muster.

    Features:
    - Cross-Game Learning: Erkenntnisse von einem Spiel auf andere übertragen
    - Globale Strategie-Bewertung: Welche Ansätze funktionieren am besten?
    - Adaptive Gewichtung: Modelle die gut performen bekommen mehr Gewicht
    - Meta-Statistiken: Globale Trends und Muster erkennen

    Unterstützte Spiele:
    1. Lotto 6aus49 (6 ML-Modelle + 15 Strategien)
    2. Eurojackpot (5 ML-Modelle + 18 Strategien)
    3. Spiel 77 (Ziffernbasiert)
    4. Super 6 (Ziffernbasiert)
    5. Glücksspirale (Ziffernbasiert)
    """

    MODEL_FILE = 'global_meta_learner.json'

    def __init__(self):
        """Initialisiert den GlobalMetaLearner."""
        # Alle unterstützten Spiele
        self.games = ['lotto_6aus49', 'eurojackpot', 'spiel77', 'super6', 'gluecksspirale']

        # ML-Modelle pro Spiel
        self.ml_models = {
            'lotto_6aus49': ['neural_network', 'markov', 'bayesian', 'reinforcement', 'ensemble', 'superzahl_ml'],
            'eurojackpot': ['neural_network', 'markov', 'bayesian', 'reinforcement', 'eurozahl_ml', 'ensemble'],
            'spiel77': ['digit_frequency', 'pattern_analyzer'],
            'super6': ['digit_frequency', 'pattern_analyzer'],
            'gluecksspirale': ['digit_frequency', 'pattern_analyzer']
        }

        # Strategie-Kategorien (spielübergreifend)
        self.strategy_categories = {
            'frequency_based': ['hot', 'cold', 'hot_cold', 'hot_cold_mix', 'overdue'],
            'pattern_based': ['odd_even', 'low_high', 'decade_balance', 'no_consecutive'],
            'mathematical': ['sum_optimized', 'delta', 'prime_mix', 'position'],
            'stochastic': ['monte_carlo', 'bayesian', 'markov'],
            'learning': ['neural_network', 'reinforcement', 'ensemble']
        }

        # Performance-Tracking pro Spiel und Modell
        self.model_performance = {game: {} for game in self.games}

        # Globale Kategorie-Performance
        self.category_performance = {cat: {
            'total_predictions': 0,
            'successful_predictions': 0,
            'average_matches': 0.0,
            'trend': 0.0,  # Positiv = verbessernd, Negativ = verschlechternd
            'confidence': 0.5
        } for cat in self.strategy_categories}

        # Cross-Game Transfer Learning
        self.transfer_weights = np.ones((len(self.games), len(self.games))) / len(self.games)

        # Globale Meta-Statistiken
        self.global_stats = {
            'total_predictions': 0,
            'total_matches': 0,
            'best_performing_game': None,
            'best_performing_category': None,
            'learning_rate': 0.1,
            'last_update': None
        }

        # Lade gespeicherte Daten
        self._load_model()

    def _load_model(self):
        """Lädt das Modell aus der Datei."""
        data = load_model(self.MODEL_FILE)
        if data:
            self.model_performance = data.get('model_performance', self.model_performance)
            self.category_performance = data.get('category_performance', self.category_performance)
            self.global_stats = data.get('global_stats', self.global_stats)

            # Transfer-Gewichte wiederherstellen
            if 'transfer_weights' in data:
                tw = data['transfer_weights']
                self.transfer_weights = np.array(tw) if isinstance(tw, list) else tw

    def save_model(self):
        """Speichert das Modell."""
        data = {
            'model_performance': self.model_performance,
            'category_performance': self.category_performance,
            'transfer_weights': self.transfer_weights.tolist(),
            'global_stats': self.global_stats,
            'last_update': datetime.now().isoformat()
        }
        save_model(self.MODEL_FILE, data)

    def record_prediction(self, game: str, model_name: str, prediction: dict,
                          actual_result: dict = None, matches: int = 0):
        """
        Zeichnet eine Vorhersage auf und lernt daraus.

        Args:
            game: Name des Spiels (z.B. 'lotto_6aus49')
            model_name: Name des Modells/Strategie
            prediction: Die Vorhersage
            actual_result: Das tatsächliche Ergebnis (optional)
            matches: Anzahl der Treffer
        """
        if game not in self.games:
            return

        # Initialisiere Modell-Entry falls nicht vorhanden
        if model_name not in self.model_performance[game]:
            self.model_performance[game][model_name] = {
                'predictions': 0,
                'total_matches': 0,
                'average_matches': 0.0,
                'wins': 0,
                'weight': 1.0,
                'recent_performance': []  # Letzte 10 Ergebnisse
            }

        perf = self.model_performance[game][model_name]
        perf['predictions'] += 1
        perf['total_matches'] += matches
        perf['recent_performance'].append(matches)

        # Nur letzte 10 behalten
        if len(perf['recent_performance']) > 10:
            perf['recent_performance'] = perf['recent_performance'][-10:]

        # Average aktualisieren
        perf['average_matches'] = perf['total_matches'] / perf['predictions']

        # Gewicht anpassen basierend auf Performance
        if matches >= 3:  # Gewinnklasse erreicht
            perf['wins'] += 1
            perf['weight'] = min(2.0, perf['weight'] * 1.05)
        elif matches == 0:
            perf['weight'] = max(0.5, perf['weight'] * 0.98)

        # Globale Stats aktualisieren
        self.global_stats['total_predictions'] += 1
        self.global_stats['total_matches'] += matches

        # Kategorie-Performance aktualisieren
        self._update_category_performance(model_name, matches)

        # Transfer-Learning anwenden
        self._update_transfer_weights(game, model_name, matches)

    def _update_category_performance(self, model_name: str, matches: int):
        """Aktualisiert die Kategorie-Performance."""
        for category, models in self.strategy_categories.items():
            # Prüfe ob Modellname zur Kategorie passt
            if any(m in model_name.lower() for m in models):
                cat_perf = self.category_performance[category]
                cat_perf['total_predictions'] += 1

                # Gewichteter Durchschnitt für average_matches
                old_avg = cat_perf['average_matches']
                n = cat_perf['total_predictions']
                cat_perf['average_matches'] = (old_avg * (n - 1) + matches) / n

                if matches >= 3:
                    cat_perf['successful_predictions'] += 1

                # Trend berechnen (Exponentieller Moving Average)
                trend_factor = 0.1
                expected = cat_perf['average_matches']
                cat_perf['trend'] = cat_perf['trend'] * (1 - trend_factor) + (matches - expected) * trend_factor

                # Confidence basierend auf Datenmenge
                cat_perf['confidence'] = min(1.0, 0.5 + cat_perf['total_predictions'] / 100)
                break

    def _update_transfer_weights(self, source_game: str, model_name: str, matches: int):
        """
        Aktualisiert Transfer-Gewichte zwischen Spielen.

        Wenn ein Modell bei einem Spiel gut funktioniert, könnte das
        gleiche Prinzip auch bei anderen Spielen funktionieren.
        """
        if source_game not in self.games:
            return

        source_idx = self.games.index(source_game)

        # Gute Performance erhöht Transfer-Gewichte
        if matches >= 2:
            learning_rate = self.global_stats['learning_rate']

            # Erhöhe Gewichte von diesem Spiel zu allen anderen
            for target_idx in range(len(self.games)):
                if target_idx != source_idx:
                    # Ähnliche Spiele bekommen mehr Transfer
                    similarity = self._game_similarity(source_game, self.games[target_idx])
                    update = learning_rate * similarity * (matches / 5.0)
                    self.transfer_weights[source_idx, target_idx] += update

            # Normalisieren
            row_sum = self.transfer_weights[source_idx].sum()
            if row_sum > 0:
                self.transfer_weights[source_idx] /= row_sum

    def _game_similarity(self, game1: str, game2: str) -> float:
        """
        Berechnet Ähnlichkeit zwischen zwei Spielen.

        Ähnliche Spieltypen (z.B. beide zahlenbasiert) haben höhere Ähnlichkeit.
        """
        # Spieltypen definieren
        number_games = {'lotto_6aus49', 'eurojackpot'}
        digit_games = {'spiel77', 'super6', 'gluecksspirale'}

        if game1 in number_games and game2 in number_games:
            return 0.9  # Sehr ähnlich
        elif game1 in digit_games and game2 in digit_games:
            return 0.8  # Ähnlich
        else:
            return 0.3  # Unterschiedlich

    def get_best_strategy_for_game(self, game: str) -> dict:
        """
        Gibt die beste Strategie für ein Spiel zurück.

        Berücksichtigt:
        1. Lokale Performance (bei diesem Spiel)
        2. Transfer-Learning (von anderen Spielen)
        3. Kategorie-Trends
        """
        if game not in self.games:
            return {'strategy': None, 'confidence': 0}

        game_perf = self.model_performance.get(game, {})

        if not game_perf:
            # Keine Daten - empfehle basierend auf Kategorie-Trends
            best_cat = max(self.category_performance.items(),
                          key=lambda x: x[1]['average_matches'] * x[1]['confidence'])
            return {
                'strategy': best_cat[0],
                'confidence': best_cat[1]['confidence'] * 0.5,
                'reason': 'Basierend auf globalen Kategorie-Trends'
            }

        # Finde beste Strategie für dieses Spiel
        best_model = max(game_perf.items(),
                        key=lambda x: x[1]['weight'] * x[1]['average_matches'])

        # Transfer-Wissen einbeziehen
        game_idx = self.games.index(game)
        transfer_boost = 0
        for other_idx, other_game in enumerate(self.games):
            if other_idx != game_idx:
                other_perf = self.model_performance.get(other_game, {})
                for model, perf in other_perf.items():
                    if model == best_model[0]:
                        transfer_boost += self.transfer_weights[other_idx, game_idx] * perf['average_matches']

        confidence = min(1.0, (best_model[1]['predictions'] / 50) + 0.3)

        return {
            'strategy': best_model[0],
            'weight': best_model[1]['weight'],
            'average_matches': best_model[1]['average_matches'],
            'transfer_boost': transfer_boost,
            'confidence': confidence,
            'reason': 'Basierend auf lokaler Performance und Transfer-Learning'
        }

    def get_global_insights(self) -> dict:
        """
        Liefert globale Erkenntnisse über alle Spiele hinweg.
        """
        insights = {
            'total_predictions': self.global_stats['total_predictions'],
            'overall_average_matches': 0,
            'best_game': None,
            'best_category': None,
            'trending_up': [],
            'trending_down': [],
            'recommendations': []
        }

        if self.global_stats['total_predictions'] > 0:
            insights['overall_average_matches'] = (
                self.global_stats['total_matches'] / self.global_stats['total_predictions']
            )

        # Bestes Spiel finden
        best_game_score = 0
        for game, models in self.model_performance.items():
            if models:
                avg_weight = sum(m['weight'] for m in models.values()) / len(models)
                if avg_weight > best_game_score:
                    best_game_score = avg_weight
                    insights['best_game'] = game

        # Beste Kategorie finden
        best_cat = max(self.category_performance.items(),
                      key=lambda x: x[1]['average_matches'] * x[1]['confidence'])
        insights['best_category'] = best_cat[0]

        # Trends analysieren
        for cat, perf in self.category_performance.items():
            if perf['trend'] > 0.1:
                insights['trending_up'].append(cat)
            elif perf['trend'] < -0.1:
                insights['trending_down'].append(cat)

        # Empfehlungen generieren
        if insights['trending_up']:
            insights['recommendations'].append(
                f"Fokussiere auf {', '.join(insights['trending_up'])} - aktuell positive Trends"
            )
        if insights['trending_down']:
            insights['recommendations'].append(
                f"Reduziere Gewichtung von {', '.join(insights['trending_down'])} - negative Trends"
            )

        return insights

    def apply_transfer_learning(self, target_game: str) -> dict:
        """
        Wendet Transfer-Learning auf ein Zielspiel an.

        Returns Gewichtsanpassungen basierend auf Erkenntnissen anderer Spiele.
        """
        if target_game not in self.games:
            return {}

        target_idx = self.games.index(target_game)
        adjustments = {}

        for source_idx, source_game in enumerate(self.games):
            if source_idx == target_idx:
                continue

            source_perf = self.model_performance.get(source_game, {})
            transfer_weight = self.transfer_weights[source_idx, target_idx]

            for model, perf in source_perf.items():
                if model not in adjustments:
                    adjustments[model] = {
                        'base_weight': 1.0,
                        'transfer_adjustment': 0.0,
                        'sources': []
                    }

                # Übertrage gelernte Gewichte
                adjustment = (perf['weight'] - 1.0) * transfer_weight * 0.5
                adjustments[model]['transfer_adjustment'] += adjustment
                adjustments[model]['sources'].append(source_game)

        # Finale Gewichte berechnen
        for model, adj in adjustments.items():
            adj['final_weight'] = adj['base_weight'] + adj['transfer_adjustment']
            adj['final_weight'] = max(0.5, min(2.0, adj['final_weight']))

        return adjustments

    def get_model_ranking(self) -> list:
        """
        Gibt ein globales Ranking aller Modelle zurück.
        """
        rankings = []

        for game, models in self.model_performance.items():
            for model, perf in models.items():
                score = perf['weight'] * perf['average_matches'] * (1 + perf['wins'] * 0.1)
                rankings.append({
                    'game': game,
                    'model': model,
                    'score': round(score, 3),
                    'predictions': perf['predictions'],
                    'average_matches': round(perf['average_matches'], 2),
                    'wins': perf['wins']
                })

        # Sortiere nach Score
        rankings.sort(key=lambda x: x['score'], reverse=True)

        return rankings


def get_global_meta_learner() -> GlobalMetaLearner:
    """
    Factory-Funktion für den GlobalMetaLearner.

    Gibt eine initialisierte Instanz zurück.
    """
    return GlobalMetaLearner()


# =====================================================
# HELPER FUNKTIONEN FÜR ALLE SPIELE
# =====================================================

def to_native_types(obj):
    """Konvertiert numpy Typen zu Python nativen Typen"""
    # NumPy 2.0 kompatibel - np.float_ und np.int_ wurden entfernt
    if hasattr(np, 'integer') and isinstance(obj, np.integer):
        return int(obj)
    elif hasattr(np, 'floating') and isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, (np.int64, np.int32, np.intc)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return [to_native_types(x) for x in obj.tolist()]
    elif isinstance(obj, list):
        return [to_native_types(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: to_native_types(v) for k, v in obj.items()}
    return obj


def get_eurojackpot_ml_predictions(draws):
    """
    Holt ML-Vorhersagen für Eurojackpot.

    Verwendet 5 echte ML-Algorithmen:
    1. Neural Network (Backpropagation)
    2. Markov Chain (Übergangswahrscheinlichkeiten)
    3. Bayesian Predictor (Thompson Sampling)
    4. Eurozahl ML (Spezialisiert auf 2 aus 12)
    5. Reinforcement Learner (Q-Learning für Strategie-Optimierung)
    """
    ensemble = EurojackpotEnsembleML()
    result = ensemble.predict(draws)

    predictions = []

    # Ensemble Champion - Konvertiere zu nativen Python-Typen
    predictions.append({
        'numbers': to_native_types(result['ensemble']['numbers']),
        'eurozahlen': to_native_types(result['ensemble']['eurozahlen']),
        'method': 'ml_ensemble_real',
        'method_name': '🏆 ML Ensemble (5 Modelle)',
        'provider': 'ml_real',
        'strategy': 'Gewichtetes Voting aus Neural Network, Markov, Bayesian & EurozahlML',
        'confidence': 88,
        'is_real_ml': True,
        'is_champion': True
    })

    # Einzelne ML-Modelle
    name_map = {
        'neural_network': '🧠 Neuronales Netz',
        'markov': '🔗 Markov-Kette',
        'bayesian': '📊 Bayesian Thompson'
    }

    for model_name, pred in result['individual'].items():
        # Skip reinforcement und eurozahl_ml (haben keine Hauptzahlen)
        if model_name in ['reinforcement', 'eurozahl_ml']:
            continue

        if 'numbers' in pred and 'eurozahlen' in pred:
            predictions.append({
                'numbers': to_native_types(pred['numbers']),
                'eurozahlen': to_native_types(pred['eurozahlen']),
                'method': f'ml_{model_name}_real',
                'method_name': name_map.get(model_name, model_name),
                'provider': 'ml_real',
                'strategy': f'Echtes ML: {model_name}',
                'confidence': to_native_types(pred['confidence']),
                'is_real_ml': True
            })

    # Eurozahl ML Vorhersage (spezialisiert auf Eurozahlen)
    if 'eurozahl_ml' in result['individual']:
        ez_pred = result['individual']['eurozahl_ml']
        # Kombiniere mit besten Hauptzahlen aus Ensemble
        predictions.append({
            'numbers': to_native_types(result['ensemble']['numbers']),
            'eurozahlen': to_native_types(ez_pred['eurozahlen']),
            'method': 'ml_eurozahl_real',
            'method_name': '🎯 Eurozahl ML (Spezialist)',
            'provider': 'ml_real',
            'strategy': 'Spezialisiertes ML für 2 aus 12 mit Paar-Analyse',
            'confidence': to_native_types(ez_pred['confidence']),
            'is_real_ml': True,
            'top_pairs': to_native_types(ez_pred.get('top_pairs', []))
        })

    # Reinforcement Learning Empfehlungen
    if 'rl_strategies' in result:
        rl_strategies = result['rl_strategies']
        if rl_strategies:
            predictions.append({
                'numbers': to_native_types(result['ensemble']['numbers']),
                'eurozahlen': to_native_types(result['ensemble']['eurozahlen']),
                'method': 'ml_reinforcement_real',
                'method_name': '🎮 Q-Learning (RL)',
                'provider': 'ml_real',
                'strategy': f'Beste Strategie: {rl_strategies[0]["strategy"] if rl_strategies else "N/A"}',
                'confidence': 75,
                'is_real_ml': True,
                'recommended_strategies': to_native_types(rl_strategies[:3])
            })

    return predictions


def get_digit_game_ml_predictions(game_name, num_digits, draws):
    """
    Holt ML-Vorhersagen für ziffernbasierte Spiele.

    Verwendet 4 echte ML-Algorithmen:
    1. Neural Network (Backpropagation pro Position)
    2. Markov Chain (Übergangswahrscheinlichkeiten)
    3. Bayesian Predictor (Thompson Sampling)
    4. Reinforcement Learner (Q-Learning)
    """
    ensemble = DigitGameEnsembleML(game_name, num_digits)
    result = ensemble.predict(draws)

    predictions = []

    # Ensemble Champion
    predictions.append({
        'number': to_native_types(result['ensemble']),
        'method': 'ml_ensemble_real',
        'method_name': '🏆 ML Ensemble',
        'provider': 'ml_real',
        'strategy': 'Gewichtetes Voting aus Neural Network, Markov, Bayesian & Q-Learning',
        'confidence': 85,
        'is_real_ml': True,
        'is_champion': True
    })

    # Einzelne Modelle
    name_map = {
        'neural_network': '🧠 Neuronales Netz',
        'markov': '🔗 Markov-Kette',
        'bayesian': '📊 Bayesian',
        'reinforcement': '🎯 Q-Learning'
    }
    for model_name, pred in result['individual'].items():
        strategy = f'Echtes ML: {model_name}'
        if model_name == 'reinforcement' and 'strategy' in pred:
            strategy = f'Q-Learning: {pred["strategy"]}'

        predictions.append({
            'number': to_native_types(pred['number']),
            'method': f'ml_{model_name}_real',
            'method_name': name_map.get(model_name, model_name),
            'provider': 'ml_real',
            'strategy': strategy,
            'confidence': to_native_types(pred['confidence']),
            'is_real_ml': True
        })

    return predictions


def train_eurojackpot_ml(draws):
    """
    Trainiert alle Eurojackpot ML-Modelle.

    5 echte ML-Algorithmen:
    1. Neural Network (Backpropagation)
    2. Markov Chain (Übergangswahrscheinlichkeiten)
    3. Bayesian Predictor (Thompson Sampling)
    4. Eurozahl ML (Spezialisiert auf 2 aus 12)
    5. Reinforcement Learner (Q-Learning)
    """
    print("\n" + "=" * 60)
    print("🌟 EUROJACKPOT ML-TRAINING (5 Modelle)")
    print("=" * 60)

    ensemble = EurojackpotEnsembleML()
    results = ensemble.train_all(draws)

    print(f"✅ Eurojackpot ML-Training abgeschlossen")
    print(f"   📊 Trainierte Modelle: Neural Network, Markov, Bayesian, EurozahlML, RL")
    return results


def train_digit_game_ml(game_name, num_digits, draws):
    """
    Trainiert alle ML-Modelle für ziffernbasierte Spiele.

    4 echte ML-Algorithmen:
    1. Neural Network (Backpropagation)
    2. Markov Chain (Übergangswahrscheinlichkeiten)
    3. Bayesian Predictor (Thompson Sampling)
    4. Reinforcement Learner (Q-Learning)
    """
    print(f"\n{'=' * 60}")
    print(f"🎰 {game_name.upper()} ML-TRAINING (4 Modelle)")
    print("=" * 60)

    ensemble = DigitGameEnsembleML(game_name, num_digits)
    results = ensemble.train_all(draws)

    print(f"✅ {game_name} ML-Training abgeschlossen")
    print(f"   📊 Trainierte Modelle: Neural Network, Markov, Bayesian, Q-Learning")
    return results


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
