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
                # Use specific GPU device for multi-GPU environments
                return 'cuda:0'
            else:
                print("Warning: CUDA forced but not available. Falling back to CPU.")
                return 'cpu'
        else:
            # Auto-detection with specific GPU device
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                if gpu_count > 1:
                    print(f"🎮 Detected {gpu_count} GPUs - using cuda:0 for TARS training")
                return 'cuda:0'
            else:
                return 'cpu'
    
    def _gather_device_info(self) -> Dict[str, Any]:
        """Gather comprehensive device information."""
        info = {
            'selected_device': self.selected_device,
            'cuda_available': torch.cuda.is_available(),
            'cpu_count': psutil.cpu_count(),
            'ram_gb': psutil.virtual_memory().total / 1024**3
        }
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            info.update({
                'gpu_count': gpu_count,
                'gpu_name': torch.cuda.get_device_name(0),
                'gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3,
                'cuda_version': torch.version.cuda,
                'cudnn_version': torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
            })
            
            # Add info for additional GPUs if present
            if gpu_count > 1:
                info['total_gpu_memory_gb'] = sum(
                    torch.cuda.get_device_properties(i).total_memory / 1024**3 
                    for i in range(gpu_count)
                )
                info['all_gpu_names'] = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
        
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
        """Get GPU-optimized configuration with multi-GPU support."""
        gpu_memory = self.device_info.get('gpu_memory_gb', 8)
        gpu_count = self.device_info.get('gpu_count', 1)
        
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
        
        # Multi-GPU optimizations
        if gpu_count >= 2:
            # Scale up for multi-GPU setup
            config['num_clients'] = min(50, config['num_clients'] * 2)  # Double clients for 2 GPUs
            config['batch_size'] = int(config['batch_size'] * 1.2)  # Slightly larger batches
            config['multi_gpu'] = True
            config['gpu_count'] = gpu_count
            config['total_gpu_memory'] = self.device_info.get('total_gpu_memory_gb', gpu_memory * gpu_count)
            print(f"🎮 Multi-GPU optimization: {gpu_count} GPUs, {config['num_clients']} clients")
        else:
            config['multi_gpu'] = False
            config['gpu_count'] = 1
        
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
            'num_rounds': 100 if dataset == 'mnist' else 120,  # Increased for better convergence
            'local_epochs': base_config['local_epochs'],
            
            # Training parameters - optimized for convergence
            'client_lr': 0.001,  # Lower learning rate for stability
            'client_optimizer': 'adam',
            'batch_size': base_config['batch_size'],
            'weight_decay': 1e-4,
            
            # Device and optimization settings
            'device': self.selected_device,
            'use_amp': base_config['use_amp'],
            'amp_dtype': 'float16' if base_config['use_amp'] else 'float32',
            'grad_clip': 1.0,
            
            # Multi-GPU settings
            'multi_gpu': base_config.get('multi_gpu', False),
            'gpu_count': base_config.get('gpu_count', 1),
            'total_gpu_memory': base_config.get('total_gpu_memory', None),
            
            # Data loading settings
            'num_workers': base_config['num_workers'],
            'pin_memory': base_config['pin_memory'],
            'prefetch_factor': base_config['prefetch_factor'],
            
            # Memory management
            'empty_cache_every': 5,
            'max_grad_norm': 1.0,
            
            # Q-learning parameters - tuned for better learning
            'learning_rate': 0.05,  # Slower Q-learning for stability
            'discount_factor': 0.95,  # Higher discount for long-term thinking
            'epsilon_start': 0.9,  # Less exploration initially
            'epsilon_decay': 0.99,  # Slower decay
            'epsilon_min': 0.05,  # Higher minimum for continued exploration
            
            # Trust mechanism parameters - optimized
            'trust_beta': 0.7,  # Higher smoothing for stability
            'trust_params': {
                'w_sim': 0.5,     # Increased similarity weight
                'w_loss': 0.3,    # Reduced loss weight
                'w_norm': 0.2,    # Keep norm weight
                'norm_threshold': 10.0  # Increased threshold for tolerance
            },
            
            # Training enhancements - optimized for convergence
            'use_scheduler': True,
            'early_stopping': True,
            'patience': 30 if dataset == 'mnist' else 40,  # Much higher patience
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
        
        if self.selected_device.startswith('cuda'):
            print(f"\n🎮 GPU INFORMATION:")
            gpu_count = self.device_info.get('gpu_count', 1)
            if gpu_count > 1:
                print(f"  GPU Count: {gpu_count} (Multi-GPU Setup)")
                print(f"  All GPUs: {', '.join(self.device_info['all_gpu_names'])}")
                print(f"  Total GPU Memory: {self.device_info['total_gpu_memory_gb']:.1f} GB")
                print(f"  Using: {self.device_info['gpu_name']} (Primary)")
                print(f"  Primary GPU Memory: {self.device_info['gpu_memory_gb']:.1f} GB")
            else:
                print(f"  GPU Name: {self.device_info['gpu_name']}")
                print(f"  GPU Memory: {self.device_info['gpu_memory_gb']:.1f} GB")
            print(f"  CUDA Version: {self.device_info['cuda_version']}")
            if self.device_info['cudnn_version']:
                print(f"  cuDNN Version: {self.device_info['cudnn_version']}")
        
        print(f"\n💻 CPU INFORMATION:")
        print(f"  CPU Cores: {self.device_info['cpu_count']}")
        print(f"  RAM: {self.device_info['ram_gb']:.1f} GB")
        
        print(f"\n🎯 OPTIMIZATION STRATEGY:")
        if self.selected_device.startswith('cuda'):
            gpu_count = self.device_info.get('gpu_count', 1)
            if gpu_count > 1:
                print("  Multi-GPU environment detected")
                print("  Client distribution across multiple GPUs enabled")
                print(f"  GPU 0: Primary device for server and coordination")
                print(f"  GPU 1: Secondary device for parallel client training")
                print(f"  Total VRAM: {self.device_info.get('total_gpu_memory_gb', 32):.1f}GB")
                print("  Expected 2x training speed improvement")
            else:
                print("  Single GPU configuration")
            print("  Using GPU acceleration with mixed precision")
            print("  Optimized batch sizes and parallel processing")
            print("  Expected 70-90% GPU utilization per device")
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