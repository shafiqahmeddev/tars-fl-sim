import torch
import numpy as np
import os
import pickle
from typing import Dict, Any
from app.components.server import Server
from app.components.client import Client
from app.defense.tars_agent import TARSAgent
import app.defense.aggregation_rules as agg
import app.attacks.poisoning as attacks
from app.shared.data_utils import load_datasets, partition_data
from app.shared.models import MNIST_CNN, CIFAR10_CNN
from torch.utils.data import DataLoader, Subset, random_split

class Simulation:
    """Manages the entire Federated Learning simulation process."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Model persistence settings
        self.model_dir = "checkpoints"
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_path = f"{self.model_dir}/{config['dataset']}_global_model.pth"
        self.agent_path = f"{self.model_dir}/{config['dataset']}_tars_agent.pkl"
        
        self._setup()

    def _setup(self):
        """Initializes all components for the simulation."""
        # Load data
        train_mnist, test_mnist, train_cifar, test_cifar = load_datasets()
        self.train_dataset = train_mnist if self.config['dataset'] == 'mnist' else train_cifar
        self.test_dataset = test_mnist if self.config['dataset'] == 'mnist' else test_cifar
        
        # Split test set for validation
        val_size = int(0.2 * len(self.test_dataset))
        test_size = len(self.test_dataset) - val_size
        val_set, self.test_set = random_split(self.test_dataset, [val_size, test_size])
        self.val_loader = DataLoader(val_set, batch_size=64)
        self.test_loader = DataLoader(self.test_set, batch_size=64)

        # Create model
        self.global_model = MNIST_CNN() if self.config['dataset'] == 'mnist' else CIFAR10_CNN()
        
        # Create server
        self.server = Server(self.global_model, self.val_loader, self.device)

        # Create clients
        client_subsets, _ = partition_data(self.train_dataset, self.config['num_clients'], self.config['is_iid'])
        num_byzantine = int(self.config['num_clients'] * self.config['byzantine_pct'])
        
        self.clients = []
        for i in range(self.config['num_clients']):
            is_byzantine = i < num_byzantine
            attack = None
            if is_byzantine:
                if self.config['attack_type'] == 'sign_flipping':
                    attack = attacks.SignFlippingAttack()
                elif self.config['attack_type'] == 'gaussian':
                    attack = attacks.GaussianAttack(std_dev=self.config.get('attack_std_dev', 1.5))

            client_loader = DataLoader(client_subsets[i], batch_size=32, shuffle=True)
            client_model = MNIST_CNN() if self.config['dataset'] == 'mnist' else CIFAR10_CNN()
            self.clients.append(Client(i, client_model, client_loader, self.device, is_byzantine, attack))

        # Create TARS agent
        self.aggregation_rules = {
            0: agg.fed_avg,
            1: agg.krum,
            2: agg.trimmed_mean,
            3: agg.median,
            4: agg.fl_trust
        }
        self.agent = TARSAgent(self.aggregation_rules, len(self.aggregation_rules), self.config)
        self.history = []
        
        # Load pre-trained model if available
        self._load_pretrained_model()

    def _load_pretrained_model(self):
        """Load pre-trained global model and TARS agent if available."""
        # Skip loading if force retrain is enabled or pretrained models are disabled
        if (self.config.get('force_retrain', False) or 
            not self.config.get('use_pretrained', True)):
            print(f"📝 Model persistence disabled or force retrain enabled. Will train from scratch.")
            self.is_pretrained = False
            return
            
        if os.path.exists(self.model_path) and os.path.exists(self.agent_path):
            print(f"🔄 Loading pre-trained {self.config['dataset'].upper()} model...")
            
            # Load global model
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.global_model.load_state_dict(checkpoint['model_state_dict'])
            self.server.set_global_model_state(checkpoint['model_state_dict'])
            
            # Load TARS agent (Q-table and trust memory)
            with open(self.agent_path, 'rb') as f:
                agent_data = pickle.load(f)
                self.agent.q_table = agent_data['q_table']
                self.agent.trust_memory = agent_data['trust_memory']
                self.agent.epsilon = agent_data.get('epsilon', self.agent.epsilon_min)
            
            print(f"✅ Pre-trained model loaded successfully!")
            print(f"   - Final accuracy: {checkpoint.get('final_accuracy', 'N/A')}%")
            print(f"   - Training rounds: {checkpoint.get('rounds_trained', 'N/A')}")
            print(f"   - Q-table entries: {len(self.agent.q_table)}")
            
            self.is_pretrained = True
        else:
            print(f"📝 No pre-trained {self.config['dataset'].upper()} model found. Will train from scratch.")
            self.is_pretrained = False

    def save_trained_model(self, final_accuracy: float, final_loss: float, rounds_trained: int):
        """Save the trained global model and TARS agent."""
        print(f"💾 Saving trained model to {self.model_path}...")
        
        # Save global model with metadata
        checkpoint = {
            'model_state_dict': self.server.get_global_model_state(),
            'final_accuracy': final_accuracy,
            'final_loss': final_loss,
            'rounds_trained': rounds_trained,
            'config': self.config,
            'dataset': self.config['dataset'],
            'device': self.device
        }
        torch.save(checkpoint, self.model_path)
        
        # Save TARS agent (Q-table and trust memory)
        agent_data = {
            'q_table': self.agent.q_table,
            'trust_memory': self.agent.trust_memory,
            'epsilon': self.agent.epsilon,
            'config': self.config
        }
        with open(self.agent_path, 'wb') as f:
            pickle.dump(agent_data, f)
        
        print(f"✅ Model saved successfully!")
        print(f"   - Global model: {self.model_path}")
        print(f"   - TARS agent: {self.agent_path}")

    def run(self):
        """Executes the main simulation loop."""
        print("Starting simulation...")
        
        # Check if we should skip training
        if (self.is_pretrained and 
            self.config.get('use_pretrained', True) and 
            not self.config.get('force_retrain', False)):
            
            print("🚀 Using pre-trained model! Evaluating performance...")
            final_acc, final_loss = self.server.evaluate_model(self.server.get_global_model_state())
            print(f"Pre-trained Model Performance:")
            print(f"   - Test Accuracy: {final_acc:.2f}%")
            print(f"   - Test Loss: {final_loss:.4f}")
            
            # Return empty history since no new training was done
            return []
        
        # Train the model
        print("🏋️ Training federated learning model...")
        last_accuracy, last_loss = self.server.evaluate_model(self.server.get_global_model_state())
        
        for t in range(self.config['num_rounds']):
            print(f"\n--- Round {t+1}/{self.config['num_rounds']} ---")
            
            # Client training
            client_updates = []
            global_state = self.server.get_global_model_state()
            for client in self.clients:
                update = client.train(global_state, current_round=t)
                client_updates.append(update)

            # TARS Decision Making
            raw_trusts = [self.agent.calculate_raw_trust_score(up, global_state, self.server) for up in client_updates]
            smoothed_trusts = [self.agent.update_and_get_smoothed_trust(i, score) for i, score in enumerate(raw_trusts)]
            avg_trust = np.mean(smoothed_trusts)
            
            state = self.agent.get_state(last_accuracy, last_loss, avg_trust)
            action = self.agent.choose_action(state, t)
            chosen_rule = self.aggregation_rules[action]
            
            print(f"TARS chose rule: {chosen_rule.__name__}")

            # Aggregation
            agg_params = {
                'num_malicious': int(self.config['num_clients'] * self.config['byzantine_pct']),
                'trim_ratio': 0.1,
                'server_update': self.server.get_global_model_state() # For FLTrust
            }
            new_global_state = chosen_rule(client_updates, **agg_params)
            self.server.set_global_model_state(new_global_state)

            # Evaluation and Learning
            accuracy, loss = self.server.evaluate_model(new_global_state)
            print(f"Round {t+1} Accuracy: {accuracy:.2f}%, Loss: {loss:.4f}")
            
            reward = self.agent.calculate_reward(accuracy, loss, avg_trust)
            next_state = self.agent.get_state(accuracy, loss, avg_trust)
            self.agent.update_q_table(state, action, reward, next_state)
            
            last_accuracy, last_loss = accuracy, loss
            self.history.append({'round': t+1, 'accuracy': accuracy, 'loss': loss, 'chosen_rule': chosen_rule.__name__})
            
        print("\n--- Final Evaluation on Test Set ---")
        final_acc, final_loss = self.server.evaluate_model(self.server.get_global_model_state())
        print(f"Final Test Accuracy: {final_acc:.2f}%, Final Test Loss: {final_loss:.4f}")
        
        # Save the trained model if configured to do so
        if self.config.get('save_model', True):
            self.save_trained_model(final_acc, final_loss, self.config['num_rounds'])
        
        return self.history
