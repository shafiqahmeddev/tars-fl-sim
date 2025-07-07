import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict, OrderedDict
from typing import Dict, Any, List, Tuple, Callable
from app.shared.interfaces import IAgent, IServer

class TARSAgent(IAgent):
    """Implements the Trust-Aware Reinforcement Selector agent with device consistency."""
    def __init__(self, actions: Dict[int, Callable], num_actions: int, config: dict):
        self.actions = actions
        self.num_actions = num_actions
        
        # Device management
        configured_device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        # Handle generic 'cuda' by converting to specific device
        if configured_device == 'cuda' and torch.cuda.is_available():
            self.device = 'cuda:0'
        else:
            self.device = configured_device
        
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

        # FIXED: Remove lambda functions to enable pickling
        self.q_table = defaultdict(self._default_q_values)
        self.trust_memory = defaultdict(self._default_trust_value)
    
    def _default_q_values(self):
        """Default Q-value initializer (replaces lambda)."""
        return np.zeros(self.num_actions)
    
    def _default_trust_value(self):
        """Default trust value initializer (replaces lambda)."""
        return 1.0

    def calculate_raw_trust_score(self, client_update_state: Dict[str, Any], global_model_state: Dict[str, Any], server: IServer) -> float:
        """Enhanced trust score calculation with improved metrics and device consistency."""
        # Extract client stats if available
        client_stats = client_update_state.get('_client_stats', {})
        
        # Remove stats from model state for evaluation
        clean_client_state = {k: v for k, v in client_update_state.items() if k != '_client_stats'}
        
        # Calculate model performance metrics
        _, client_loss = server.evaluate_model(clean_client_state)
        _, global_loss = server.evaluate_model(global_model_state)
        loss_divergence = client_loss - global_loss

        # Calculate parameter similarity with device consistency
        client_tensors = []
        global_tensors = []
        
        for key in clean_client_state.keys():
            if isinstance(clean_client_state[key], torch.Tensor) and isinstance(global_model_state[key], torch.Tensor):
                # Ensure both tensors are on the same device
                client_tensor = clean_client_state[key].to(device=self.device, dtype=torch.float32)
                global_tensor = global_model_state[key].to(device=self.device, dtype=torch.float32)
                
                client_tensors.append(client_tensor.view(-1))
                global_tensors.append(global_tensor.view(-1))
        
        if not client_tensors or not global_tensors:
            # Fallback if no valid tensors found
            return 0.5
        
        client_vec = torch.cat(client_tensors)
        global_vec = torch.cat(global_tensors)
        
        # Ensure vectors are on same device
        if client_vec.device != global_vec.device:
            global_vec = global_vec.to(client_vec.device)
        
        cosine_sim = F.cosine_similarity(client_vec, global_vec, dim=0).item()
        
        # Handle NaN from cosine similarity
        import math
        if math.isnan(cosine_sim):
            cosine_sim = 0.0

        # Calculate gradient/update magnitude with device consistency
        grad_vec = client_vec - global_vec
        grad_norm = torch.norm(grad_vec).item()
        
        # Normalize gradient norm by model size for better scaling
        model_size = torch.norm(global_vec).item()
        normalized_grad_norm = grad_norm / (model_size + 1e-8)
        
        # Handle NaN from gradient calculations
        if math.isnan(grad_norm) or math.isnan(model_size) or math.isnan(normalized_grad_norm):
            normalized_grad_norm = 1.0  # Default to moderate penalty
        
        # Handle NaN from loss calculations
        if math.isnan(client_loss) or math.isnan(global_loss) or math.isnan(loss_divergence):
            loss_divergence = 0.0
            global_loss = 1.0  # Safe default

        # Enhanced trust score calculation
        similarity_score = max(0, cosine_sim)  # Clamp to [0, 1]
        loss_score = max(0, 1 - abs(loss_divergence) / (abs(global_loss) + 1e-8))  # Relative loss difference
        norm_score = max(0, 1 - normalized_grad_norm / self.trust_params['norm_threshold'])  # Normalized gradient penalty
        
        # Weighted combination
        score = (
            self.trust_params['w_sim'] * similarity_score +
            self.trust_params['w_loss'] * loss_score +
            self.trust_params['w_norm'] * norm_score
        )
        
        # Additional penalty for extreme updates
        if normalized_grad_norm > self.trust_params['norm_threshold'] * 2:
            score *= 0.5  # Heavy penalty for very large updates
            
        # Use client training loss if available
        if 'avg_loss' in client_stats:
            client_training_loss = client_stats['avg_loss']
            if client_training_loss > global_loss * 3:  # Unusually high training loss
                score *= 0.8
        
        # Ensure score is in [0, 1] range and handle any remaining NaN
        if math.isnan(score):
            score = 0.5  # Default neutral score
        normalized_score = max(0, min(1, score))
        
        # Final NaN check
        if math.isnan(normalized_score):
            normalized_score = 0.5
            
        return normalized_score

    def update_and_get_smoothed_trust(self, client_id: int, raw_trust_score: float) -> float:
        """Implements temporal trust smoothing with NaN handling."""
        import math
        
        # Handle NaN input
        if math.isnan(raw_trust_score):
            raw_trust_score = 0.5  # Default neutral trust
            
        prev_trust = self.trust_memory[client_id]
        if math.isnan(prev_trust):
            prev_trust = 1.0  # Default initial trust
            
        smoothed_trust = self.beta * prev_trust + (1 - self.beta) * raw_trust_score
        
        # Handle NaN result
        if math.isnan(smoothed_trust):
            smoothed_trust = 0.5  # Fallback neutral trust
            
        self.trust_memory[client_id] = smoothed_trust
        return smoothed_trust

    def get_state(self, prev_accuracy: float, prev_loss: float, avg_trust_score: float) -> tuple:
        """Encodes continuous metrics into a discrete state for the Q-table with NaN handling."""
        import math
        
        # Handle NaN values by replacing with safe defaults
        if math.isnan(prev_accuracy) or prev_accuracy is None:
            prev_accuracy = 0.0
        if math.isnan(prev_loss) or prev_loss is None:
            prev_loss = 2.0  # High loss default
        if math.isnan(avg_trust_score) or avg_trust_score is None:
            avg_trust_score = 0.5  # Neutral trust default
        
        # Clamp values to reasonable ranges
        prev_accuracy = max(0.0, min(100.0, prev_accuracy))
        prev_loss = max(0.0, min(10.0, prev_loss))  # Expand loss range
        avg_trust_score = max(0.0, min(1.0, avg_trust_score))
        
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
        """Calculates the trust-regularized reward with NaN handling."""
        import math
        
        # Handle NaN values by replacing with safe defaults
        if math.isnan(new_accuracy) or new_accuracy is None:
            new_accuracy = 0.0
        if math.isnan(new_loss) or new_loss is None:
            new_loss = 2.0  # High loss default
        if math.isnan(avg_smoothed_trust) or avg_smoothed_trust is None:
            avg_smoothed_trust = 0.5  # Neutral trust default
        
        # Clamp values to reasonable ranges
        new_accuracy = max(0.0, min(100.0, new_accuracy))
        new_loss = max(0.0, min(10.0, new_loss))
        avg_smoothed_trust = max(0.0, min(1.0, avg_smoothed_trust))
        
        reward = (
            self.reward_weights['alpha1'] * new_accuracy -
            self.reward_weights['alpha2'] * new_loss +
            self.reward_weights['alpha3'] * avg_smoothed_trust
        )
        
        # Ensure reward is not NaN
        if math.isnan(reward):
            reward = 0.0
        
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
