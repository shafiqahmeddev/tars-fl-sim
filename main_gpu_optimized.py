from app.simulation import Simulation
import pandas as pd
import torch

if __name__ == "__main__":
    # Check GPU availability and memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🚀 GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {gpu_memory:.1f} GB")
        
        # Optimize for GPU utilization based on available memory
        if gpu_memory >= 12:  # High-end GPU
            batch_size = 128
            num_clients = 20
            print("📈 Using high-performance configuration")
        elif gpu_memory >= 8:  # Mid-range GPU  
            batch_size = 64
            num_clients = 15
            print("⚡ Using optimized configuration")
        else:  # Lower-end GPU
            batch_size = 32
            num_clients = 10
            print("🔧 Using standard configuration")
    else:
        batch_size = 32
        num_clients = 10
        print("⚠️  GPU not available, using CPU configuration")
    
    # --- GPU-Optimized Configuration ---
    config = {
        "dataset": "mnist",  # 'mnist' or 'cifar10'
        "num_clients": num_clients,  # More clients = more GPU utilization
        "byzantine_pct": 0.2,
        "attack_type": "sign_flipping",
        "is_iid": False,
        "num_rounds": 50,
        "local_epochs": 5,  # Increased for more GPU work per round
        
        # GPU-optimized training parameters
        "client_lr": 0.001,
        "client_optimizer": "adam",
        "batch_size": batch_size,  # Larger batches = better GPU utilization
        "weight_decay": 1e-4,
        
        # Enhanced data loading for GPU
        "num_workers": 4,  # Parallel data loading
        "pin_memory": True,  # Faster GPU transfer
        "prefetch_factor": 2,  # Prefetch batches
        
        # Mixed precision training for better GPU utilization
        "use_amp": True,  # Automatic Mixed Precision
        "amp_dtype": "float16",  # Use half precision
        
        # Parallel training options
        "parallel_clients": True,  # Train multiple clients in parallel
        "max_parallel": 4,  # Max clients training simultaneously
        
        # Q-learning parameters
        "learning_rate": 0.1,
        "discount_factor": 0.9,
        "epsilon_start": 1.0,
        "epsilon_decay": 0.995,
        "epsilon_min": 0.01,
        
        # Trust mechanism parameters
        "trust_beta": 0.5,
        "trust_params": {
            "w_sim": 0.4,
            "w_loss": 0.4,
            "w_norm": 0.2,
            "norm_threshold": 5.0
        },
        
        # Model persistence options
        "use_pretrained": True,
        "save_model": True,
        "force_retrain": False,
        
        # Training enhancements
        "use_scheduler": True,
        "early_stopping": True,
        "patience": 10,
        
        # GPU memory optimization
        "gradient_accumulation_steps": 1,  # Accumulate gradients
        "max_grad_norm": 1.0,  # Gradient clipping
        "empty_cache_every": 5,  # Clear GPU cache every N rounds
    }
    
    print(f"🎯 Training Configuration:")
    print(f"   Clients: {num_clients}")
    print(f"   Batch Size: {batch_size}")
    print(f"   Local Epochs: {config['local_epochs']}")
    print(f"   Mixed Precision: {config.get('use_amp', False)}")
    print(f"   Parallel Training: {config.get('parallel_clients', False)}")

    # --- Run GPU-Optimized Simulation ---
    print("\n🚀 Starting GPU-optimized TARS training...")
    simulation = Simulation(config)
    history = simulation.run()

    # --- Save Results ---
    if history:
        df = pd.DataFrame(history)
        df.to_csv("gpu_optimized_results.csv", index=False)
        print(f"\n💾 Results saved to gpu_optimized_results.csv")
        
        # Print final performance
        final_acc = df['accuracy'].iloc[-1]
        best_acc = df['accuracy'].max()
        print(f"📊 Final Accuracy: {final_acc:.2f}%")
        print(f"🏆 Best Accuracy: {best_acc:.2f}%")
        
        if best_acc >= 97.0:
            print("🎉 TARGET ACHIEVED: 97%+ accuracy reached!")
    else:
        print("⚠️  No training history available")