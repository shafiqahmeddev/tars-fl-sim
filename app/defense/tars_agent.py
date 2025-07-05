import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict, OrderedDict
from typing import Dict, Any, List, Tuple, Callable
from app.shared.interfaces import IAgent, IServer

class TARSAgent(IAgent):
    """Implements the Trust-Aware Reinforcement Selector agent."""
    def __init__(self, actions: Dict[int, Callable], num_actions: int, config: dict):
        self.actions = actions
        self.num_actions = num_actions
        
        # RL Hyperparameters
        self.learning_rate = config.get('learning_rate', 0.1)
        self.discount_factor = config.get('discount_factor', 0.9)
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.epsilon_min = config.get('epsilon_min', 0.01)
        
        # Trust mechanism parameters
        self.beta = config.get('trust_beta', 0.5) # For temporal smoothing
        self.trust_params = config.get('trust_params', {'w_sim': 0.5, 'w_loss': 0.3, 'w_norm': 0.2, 'norm_threshold': 10.0})
        self.reward_weights = config.get('reward_weights', {'alpha1': 1.0, 'alpha2': 0.5, 'alpha3': 0.2})

        self.q_table = defaultdict(lambda: np.zeros(self.num_actions))
        self.trust_memory = defaultdict(lambda: 1.0)

    def calculate_raw_trust_score(self, client_update_state: Dict[str, Any], global_model_state: Dict[str, Any], server: IServer) -> float:
        """Calculates the raw trust score based on the paper's criteria."""
        _, client_loss = server.evaluate_model(client_update_state)
        _, global_loss = server.evaluate_model(global_model_state)
        loss_divergence = client_loss - global_loss

        client_vec = torch.cat([p.view(-1) for p in client_update_state.values()])
        global_vec = torch.cat([p.view(-1) for p in global_model_state.values()])
        cosine_sim = F.cosine_similarity(client_vec, global_vec, dim=0).item()

        grad_vec = client_vec - global_vec
        grad_norm = torch.norm(grad_vec).item()

        score = (
            self.trust_params['w_sim'] * cosine_sim -
            self.trust_params['w_loss'] * max(0, loss_divergence) -
            self.trust_params['w_norm'] * (1 if grad_norm > self.trust_params['norm_threshold'] else 0)
        )
        normalized_score = (np.tanh(score) + 1) / 2
        return normalized_score

    def update_and_get_smoothed_trust(self, client_id: int, raw_trust_score: float) -> float:
        """Implements temporal trust smoothing."""
        prev_trust = self.trust_memory[client_id]
        smoothed_trust = self.beta * prev_trust + (1 - self.beta) * raw_trust_score
        self.trust_memory[client_id] = smoothed_trust
        return smoothed_trust

    def get_state(self, prev_accuracy: float, prev_loss: float, avg_trust_score: float) -> tuple:
        """Encodes continuous metrics into a discrete state for the Q-table."""
        acc_bin = int(prev_accuracy / 10)
        loss_bin = int(min(prev_loss, 2.0) * 5)
        trust_bin = int(avg_trust_score * 10)
        return (acc_bin, loss_bin, trust_bin)

    def choose_action(self, state: tuple, current_round: int) -> int:
        """Chooses an action using an epsilon-greedy policy."""
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.num_actions)
        return np.argmax(self.q_table[state])

    def calculate_reward(self, new_accuracy: float, new_loss: float, avg_smoothed_trust: float) -> float:
        """Calculates the trust-regularized reward."""
        reward = (
            self.reward_weights['alpha1'] * new_accuracy -
            self.reward_weights['alpha2'] * new_loss +
            self.reward_weights['alpha3'] * avg_smoothed_trust
        )
        return reward

    def update_q_table(self, state: tuple, action: int, reward: float, next_state: tuple):
        """Updates the Q-table using the Bellman equation."""
        old_value = self.q_table[state][action]
        next_max = np.max(self.q_table[next_state])
        new_value = old_value + self.learning_rate * (reward + self.discount_factor * next_max - old_value)
        self.q_table[state][action] = new_value

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
