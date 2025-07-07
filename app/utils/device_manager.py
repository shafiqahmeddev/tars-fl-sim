"""
Device Management Utility for TARS Federated Learning
Provides device selection and consistency management for Kaggle compatibility.
"""

import torch
import psutil
from typing import Dict, Any, Tuple, Optional


class DeviceManager:
    """Manages device selection and consistency for federated learning."""
    
    def __init__(self, force_device: Optional[str] = None):
        """
        Initialize device manager.
        
        Args:
            force_device: Force specific device ('cpu', 'cuda', or None for auto)
        """
        self.force_device = force_device
        self.selected_device = self._select_optimal_device()
        self.device_info = self._gather_device_info()
    
    def _select_optimal_device(self) -> str:
        """Select optimal device based on availability and force settings."""
        if self.force_device == 'cpu':
            return 'cpu'
        elif self.force_device == 'cuda':
            if torch.cuda.is_available():
                return 'cuda'
            else:
                print("Warning: CUDA forced but not available. Falling back to CPU.")
                return 'cpu'
        else:
            # Auto-detection
            return 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def _gather_device_info(self) -> Dict[str, Any]:
        """Gather comprehensive device information."""
        info = {
            'selected_device': self.selected_device,
            'cuda_available': torch.cuda.is_available(),
            'cpu_count': psutil.cpu_count(),
            'ram_gb': psutil.virtual_memory().total / 1024**3
        }
        
        if torch.cuda.is_available():
            info.update({
                'gpu_name': torch.cuda.get_device_name(0),
                'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3,
                'cuda_version': torch.version.cuda,
                'cudnn_version': torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
            })
        
        return info
    
    def get_optimal_config(self, dataset: str = 'mnist') -> Dict[str, Any]:
        """
        Get optimal configuration based on device and environment.
        
        Args:
            dataset: Dataset name ('mnist' or 'cifar10')
            
        Returns:
            Optimized configuration dictionary
        """
        if self.selected_device == 'cuda':
            return self._get_gpu_config(dataset)
        else:
            return self._get_cpu_config(dataset)
    
    def _get_gpu_config(self, dataset: str) -> Dict[str, Any]:
        """Get GPU-optimized configuration."""
        gpu_memory = self.device_info.get('gpu_memory_gb', 8)
        
        # Base configuration for GPU
        if dataset == 'mnist':
            config = {
                'batch_size': 256 if gpu_memory >= 8 else 128,
                'num_clients': 20 if gpu_memory >= 8 else 15,
                'local_epochs': 5,
                'num_workers': 4,  # Kaggle optimal
                'use_amp': True,
                'pin_memory': True,
                'prefetch_factor': 2
            }
        else:  # cifar10
            config = {
                'batch_size': 128 if gpu_memory >= 8 else 64,
                'num_clients': 20 if gpu_memory >= 8 else 15,
                'local_epochs': 5,
                'num_workers': 4,  # Kaggle optimal
                'use_amp': True,
                'pin_memory': True,
                'prefetch_factor': 2
            }
        
        # High-end GPU optimizations
        if gpu_memory >= 15:  # 16GB GPU
            config['batch_size'] = int(config['batch_size'] * 1.5)
            config['num_clients'] = min(30, config['num_clients'] + 10)
        
        return config
    
    def _get_cpu_config(self, dataset: str) -> Dict[str, Any]:
        """Get CPU-optimized configuration."""
        cpu_count = self.device_info.get('cpu_count', 4)
        ram_gb = self.device_info.get('ram_gb', 8)
        
        # Base configuration for CPU
        if dataset == 'mnist':
            config = {
                'batch_size': 64,
                'num_clients': 10,
                'local_epochs': 3,
                'num_workers': min(4, cpu_count),
                'use_amp': False,  # No mixed precision on CPU
                'pin_memory': False,
                'prefetch_factor': 2
            }
        else:  # cifar10
            config = {
                'batch_size': 32,
                'num_clients': 8,
                'local_epochs': 3,
                'num_workers': min(4, cpu_count),
                'use_amp': False,  # No mixed precision on CPU
                'pin_memory': False,
                'prefetch_factor': 2
            }
        
        # High-RAM optimizations
        if ram_gb >= 16:
            config['batch_size'] = int(config['batch_size'] * 1.5)
            config['num_clients'] = min(15, config['num_clients'] + 5)
        
        return config
    
    def create_full_config(self, dataset: str, byzantine_pct: float = 0.1) -> Dict[str, Any]:
        """
        Create complete TARS configuration with device optimization.
        
        Args:
            dataset: Dataset name ('mnist' or 'cifar10')
            byzantine_pct: Percentage of Byzantine clients
            
        Returns:
            Complete configuration dictionary
        """
        base_config = self.get_optimal_config(dataset)
        
        full_config = {
            # Dataset and federated learning settings
            'dataset': dataset,
            'num_clients': base_config['num_clients'],
            'byzantine_pct': byzantine_pct,
            'attack_type': 'sign_flipping',
            'is_iid': False,
            'num_rounds': 50 if dataset == 'mnist' else 60,
            'local_epochs': base_config['local_epochs'],
            
            # Training parameters
            'client_lr': 0.01,  # Conservative for stability
            'client_optimizer': 'adam',
            'batch_size': base_config['batch_size'],
            'weight_decay': 1e-4,
            
            # Device and optimization settings
            'device': self.selected_device,
            'use_amp': base_config['use_amp'],
            'amp_dtype': 'float16' if base_config['use_amp'] else 'float32',
            'grad_clip': 1.0,
            
            # Data loading settings
            'num_workers': base_config['num_workers'],
            'pin_memory': base_config['pin_memory'],
            'prefetch_factor': base_config['prefetch_factor'],
            
            # Memory management
            'empty_cache_every': 5,
            'max_grad_norm': 1.0,
            
            # Q-learning parameters
            'learning_rate': 0.1,
            'discount_factor': 0.9,
            'epsilon_start': 1.0,
            'epsilon_decay': 0.995,
            'epsilon_min': 0.01,
            
            # Trust mechanism parameters
            'trust_beta': 0.5,
            'trust_params': {
                'w_sim': 0.4,
                'w_loss': 0.4,
                'w_norm': 0.2,
                'norm_threshold': 5.0
            },
            
            # Training enhancements
            'use_scheduler': True,
            'early_stopping': True,
            'patience': 15 if dataset == 'mnist' else 20,
            'save_model': True,
            'use_pretrained': False,
            'force_retrain': True
        }
        
        return full_config
    
    def print_device_info(self):
        """Print comprehensive device information."""
        print("🔍 DEVICE MANAGER ANALYSIS")
        print("=" * 50)
        
        print(f"Selected Device: {self.selected_device}")
        print(f"Force Setting: {self.force_device or 'Auto-detect'}")
        
        if self.selected_device == 'cuda':
            print(f"\n🎮 GPU INFORMATION:")
            print(f"  GPU Name: {self.device_info['gpu_name']}")
            print(f"  GPU Memory: {self.device_info['gpu_memory_gb']:.1f} GB")
            print(f"  CUDA Version: {self.device_info['cuda_version']}")
            if self.device_info['cudnn_version']:
                print(f"  cuDNN Version: {self.device_info['cudnn_version']}")
        
        print(f"\n💻 CPU INFORMATION:")
        print(f"  CPU Cores: {self.device_info['cpu_count']}")
        print(f"  RAM: {self.device_info['ram_gb']:.1f} GB")
        
        print(f"\n🎯 OPTIMIZATION STRATEGY:")
        if self.selected_device == 'cuda':
            print("  Using GPU acceleration with mixed precision")
            print("  Optimized batch sizes and parallel processing")
            print("  Expected 70-90% GPU utilization")
        else:
            print("  Using CPU with conservative settings")
            print("  Disabled mixed precision for stability")
            print("  Optimized for CPU cores and RAM")


def create_device_configs(force_device: Optional[str] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Create optimized MNIST and CIFAR-10 configurations.
    
    Args:
        force_device: Force specific device ('cpu', 'cuda', or None for auto)
        
    Returns:
        Tuple of (mnist_config, cifar_config)
    """
    device_manager = DeviceManager(force_device=force_device)
    device_manager.print_device_info()
    
    mnist_config = device_manager.create_full_config('mnist')
    cifar_config = device_manager.create_full_config('cifar10')
    
    return mnist_config, cifar_config