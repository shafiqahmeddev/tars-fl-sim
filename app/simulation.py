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
    """Manages the entire Federated Learning simulation process with enhanced device management."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Enhanced device management - use config device if specified
        config_device = config.get('device', 'auto')
        if config_device == 'auto':
            if torch.cuda.is_available():
                # Use specific GPU device for multi-GPU environments
                gpu_count = torch.cuda.device_count()
                if gpu_count > 1:
                    print(f"🎮 Detected {gpu_count} GPUs - using cuda:0 for TARS simulation")
                self.device = "cuda:0"
            else:
                self.device = "cpu"
        elif config_device == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            self.device = "cpu"
        elif config_device == 'cuda' and torch.cuda.is_available():
            # Convert generic cuda to specific device
            self.device = "cuda:0"
        else:
            self.device = config_device
            
        print(f"Using device: {self.device}")
        
        # Ensure config has device information for components
        self.config['device'] = self.device
        
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
        
        # GPU-optimized data loaders
        num_workers = self.config.get('num_workers', 0)
        pin_memory = self.config.get('pin_memory', False)
        prefetch_factor = self.config.get('prefetch_factor', 2)
        
        self.val_loader = DataLoader(
            val_set, 
            batch_size=self.config.get('batch_size', 64),
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            prefetch_factor=prefetch_factor if num_workers > 0 else 2
        )
        self.test_loader = DataLoader(
            self.test_set, 
            batch_size=self.config.get('batch_size', 64),
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            prefetch_factor=prefetch_factor if num_workers > 0 else 2
        )

        # Create model and ensure device placement
        self.global_model = MNIST_CNN() if self.config['dataset'] == 'mnist' else CIFAR10_CNN()
        self.global_model = self.global_model.to(self.device)
        
        # Create server
        self.server = Server(self.global_model, self.val_loader, self.device)

        # Create clients with enhanced configuration
        client_subsets, _ = partition_data(self.train_dataset, self.config['num_clients'], self.config['is_iid'])
        num_byzantine = int(self.config['num_clients'] * self.config['byzantine_pct'])
        
        batch_size = self.config.get('batch_size', 32)
        
        self.clients = []
        for i in range(self.config['num_clients']):
            is_byzantine = i < num_byzantine
            attack = None
            if is_byzantine:
                if self.config['attack_type'] == 'sign_flipping':
                    attack = attacks.SignFlippingAttack()
                elif self.config['attack_type'] == 'gaussian':
                    attack = attacks.GaussianAttack(std_dev=self.config.get('attack_std_dev', 1.5))

            # GPU-optimized client data loaders
            client_loader = DataLoader(
                client_subsets[i], 
                batch_size=batch_size, 
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory and torch.cuda.is_available(),
                prefetch_factor=prefetch_factor if num_workers > 0 else 2
            )
            client_model = MNIST_CNN() if self.config['dataset'] == 'mnist' else CIFAR10_CNN()
            client_model = client_model.to(self.device)  # Ensure client model is on correct device
            self.clients.append(Client(i, client_model, client_loader, self.device, is_byzantine, attack, self.config))

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
        
        # Performance monitoring
        self.best_accuracy = 0.0
        self.best_model_state = None
        self.patience_counter = 0
        self.early_stop = False
        
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
        
        # GPU memory monitoring
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"💾 Initial GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB / {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f}GB")
        
        last_accuracy, last_loss = self.server.evaluate_model(self.server.get_global_model_state())
        
        for t in range(self.config['num_rounds']):
            print(f"\n--- Round {t+1}/{self.config['num_rounds']} ---")
            
            # Clear GPU cache periodically
            if torch.cuda.is_available() and t % self.config.get('empty_cache_every', 5) == 0:
                torch.cuda.empty_cache()
                gpu_memory = torch.cuda.memory_allocated() / 1024**3
                max_memory = torch.cuda.max_memory_allocated() / 1024**3
                print(f"🔧 GPU Memory: {gpu_memory:.2f}GB (Peak: {max_memory:.2f}GB)")
            
            # Client training with device consistency
            client_updates = []
            global_state = self.server.get_global_model_state()
            
            # Ensure global state tensors are on correct device
            for key, tensor in global_state.items():
                if isinstance(tensor, torch.Tensor):
                    global_state[key] = tensor.to(self.device)
            
            for client in self.clients:
                update = client.train(global_state, current_round=t)
                
                # Ensure client update tensors are on correct device
                for key, tensor in update.items():
                    if isinstance(tensor, torch.Tensor):
                        update[key] = tensor.to(self.device)
                
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
            
            # Ensure aggregated state tensors are on correct device
            for key, tensor in new_global_state.items():
                if isinstance(tensor, torch.Tensor):
                    new_global_state[key] = tensor.to(self.device)
            
            self.server.set_global_model_state(new_global_state)

            # Evaluation and Learning
            accuracy, loss = self.server.evaluate_model(new_global_state)
            
            # Enhanced logging with device and convergence monitoring
            print(f"Round {t+1} Accuracy: {accuracy:.2f}%, Loss: {loss:.4f}, Avg Trust: {avg_trust:.3f}")
            
            # Debug device consistency
            if t % 10 == 0:  # Every 10 rounds
                self._debug_device_consistency(new_global_state, t+1)
            
            # Monitor convergence
            if t > 5:  # After first few rounds
                recent_accuracies = [h['accuracy'] for h in self.history[-5:]]
                if recent_accuracies:
                    acc_trend = recent_accuracies[-1] - recent_accuracies[0] if len(recent_accuracies) > 1 else 0
                    if abs(acc_trend) < 0.5:  # Very slow progress
                        print(f"⚠️ Slow convergence detected (trend: {acc_trend:+.2f}%)")
                        if accuracy < 30:  # Very low accuracy
                            print(f"💡 Suggestion: Check device placement and learning rates")
            
            # Performance monitoring and early stopping
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_model_state = new_global_state.copy()
                self.patience_counter = 0
                print(f"🎯 New best accuracy: {accuracy:.2f}%")
            else:
                self.patience_counter += 1
            
            # Early stopping check
            if self.config.get('early_stopping', False):
                patience = self.config.get('patience', 10)
                if self.patience_counter >= patience:
                    print(f"🛑 Early stopping triggered after {patience} rounds without improvement")
                    self.early_stop = True
            
            reward = self.agent.calculate_reward(accuracy, loss, avg_trust)
            next_state = self.agent.get_state(accuracy, loss, avg_trust)
            self.agent.update_q_table(state, action, reward, next_state)
            
            last_accuracy, last_loss = accuracy, loss
            
            # Enhanced history tracking
            round_stats = {
                'round': t+1, 
                'accuracy': accuracy, 
                'loss': loss, 
                'avg_trust': avg_trust,
                'chosen_rule': chosen_rule.__name__,
                'client_losses': [up.get('_client_stats', {}).get('avg_loss', 0) for up in client_updates],
                'trust_scores': smoothed_trusts.copy(),
                'epsilon': self.agent.epsilon
            }
            self.history.append(round_stats)
            
            # Break if early stopping triggered
            if self.early_stop:
                break
            
        print("\n--- Final Evaluation on Test Set ---")
        
        # Use best model if available and early stopping was used
        if self.best_model_state is not None and self.config.get('early_stopping', False):
            print("🏆 Using best model from training...")
            self.server.set_global_model_state(self.best_model_state)
            final_acc, final_loss = self.server.evaluate_model(self.best_model_state)
        else:
            final_acc, final_loss = self.server.evaluate_model(self.server.get_global_model_state())
        
        print(f"Final Test Accuracy: {final_acc:.2f}%, Final Test Loss: {final_loss:.4f}")
        print(f"Best Accuracy Achieved: {self.best_accuracy:.2f}%")
        
        # Performance summary
        if final_acc >= 97.0:
            print("🎉 TARGET ACHIEVED: 97%+ accuracy reached!")
        elif final_acc >= 95.0:
            print("✅ EXCELLENT: 95%+ accuracy achieved!")
        elif final_acc >= 90.0:
            print("👍 GOOD: 90%+ accuracy achieved")
        else:
            print("⚠️  Accuracy below 90% - may need further tuning")
        
        # Save the trained model if configured to do so
        if self.config.get('save_model', True):
            actual_rounds = len(self.history)
            self.save_trained_model(final_acc, final_loss, actual_rounds)
        
        return self.history

    def _debug_device_consistency(self, model_state: Dict[str, Any], round_num: int):
        """Debug device consistency and tensor information."""
        print(f"\n🔍 Device Consistency Debug (Round {round_num})")
        print("-" * 40)
        
        device_summary = {}
        tensor_info = {}
        
        for key, tensor in model_state.items():
            if isinstance(tensor, torch.Tensor):
                device = str(tensor.device)
                dtype = str(tensor.dtype)
                shape = tuple(tensor.shape)
                
                if device not in device_summary:
                    device_summary[device] = 0
                device_summary[device] += 1
                
                tensor_info[key] = {
                    'device': device,
                    'dtype': dtype,
                    'shape': shape,
                    'requires_grad': tensor.requires_grad,
                    'is_cuda': tensor.is_cuda
                }
        
        # Print device summary
        print(f"📊 Tensor Device Distribution:")
        for device, count in device_summary.items():
            print(f"  {device}: {count} tensors")
        
        # Check for device mismatches using device compatibility
        expected_device = torch.device(self.device)
        mismatched_tensors = []
        for key, info in tensor_info.items():
            tensor_device = torch.device(info['device'])
            # Check device compatibility rather than exact string match
            if tensor_device != expected_device:
                mismatched_tensors.append(f"{key} ({info['device']})")
        
        if mismatched_tensors:
            print(f"⚠️  Device Mismatches (expected {expected_device}):")
            for tensor_name in mismatched_tensors[:5]:  # Show first 5
                print(f"  - {tensor_name}")
            if len(mismatched_tensors) > 5:
                print(f"  ... and {len(mismatched_tensors) - 5} more")
        else:
            print(f"✅ All tensors on correct device ({expected_device})")
        
        # Memory usage if CUDA
        if torch.cuda.is_available() and expected_device.type == 'cuda':
            memory_used = torch.cuda.memory_allocated() / 1024**3
            memory_cached = torch.cuda.memory_reserved() / 1024**3
            print(f"🎮 GPU Memory: {memory_used:.2f}GB used, {memory_cached:.2f}GB cached")
        
        print("-" * 40)
