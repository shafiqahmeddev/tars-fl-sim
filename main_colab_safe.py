from app.simulation import Simulation
import pandas as pd
import torch

if __name__ == "__main__":
    # Safe Colab Configuration - Won't trigger termination
    print("🛡️ SAFE COLAB CONFIGURATION - No Session Termination")
    print("=" * 60)
    
    # Check GPU availability
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🚀 GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {gpu_memory:.1f} GB")
        print("🔧 Using conservative settings to avoid termination")
    else:
        print("⚠️  GPU not available, using CPU configuration")
    
    # Conservative configuration that won't trigger Colab's abuse detection
    config = {
        "dataset": "mnist",  # 'mnist' or 'cifar10'
        "num_clients": 15,   # Conservative client count
        "byzantine_pct": 0.2,
        "attack_type": "sign_flipping",
        "is_iid": False,
        "num_rounds": 50,
        "local_epochs": 3,   # Safe epoch count
        
        # Safe GPU utilization parameters
        "client_lr": 0.001,
        "client_optimizer": "adam",
        "batch_size": 128,   # Safe batch size (was 32, now 4x larger)
        "weight_decay": 1e-4,
        
        # Conservative data loading to avoid RAM issues
        "num_workers": 2,    # Conservative worker count
        "pin_memory": True,  # Keep this optimization
        "prefetch_factor": 2,
        
        # GPU optimizations that are safe
        "use_amp": True,     # Mixed precision is safe and helpful
        "amp_dtype": "float16",
        "grad_clip": 1.0,
        
        # Conservative memory management
        "empty_cache_every": 5,  # More frequent cache clearing
        "max_grad_norm": 1.0,
        
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
    }
    
    print(f"🎯 Safe Configuration Summary:")
    print(f"   Clients: {config['num_clients']} (conservative)")
    print(f"   Batch Size: {config['batch_size']} (4x original)")
    print(f"   Local Epochs: {config['local_epochs']} (safe)")
    print(f"   Mixed Precision: {config['use_amp']} (memory efficient)")
    print(f"   Workers: {config['num_workers']} (conservative)")
    print(f"   Expected GPU Usage: 6-8GB (40-50%)")
    print(f"   Termination Risk: VERY LOW ✅")
    print(f"   Expected Time: 20-25 minutes")

    # --- Run Safe Simulation ---
    print("\n🚀 Starting SAFE TARS training (won't terminate)...")
    simulation = Simulation(config)
    history = simulation.run()

    # --- Save Results ---
    if history:
        df = pd.DataFrame(history)
        df.to_csv("safe_colab_results.csv", index=False)
        print(f"\n💾 Results saved to safe_colab_results.csv")
        
        # Print performance summary
        final_acc = df['accuracy'].iloc[-1]
        best_acc = df['accuracy'].max()
        print(f"📊 Final Accuracy: {final_acc:.2f}%")
        print(f"🏆 Best Accuracy: {best_acc:.2f}%")
        
        if best_acc >= 97.0:
            print("🎉 TARGET ACHIEVED: 97%+ accuracy reached!")
        elif best_acc >= 95.0:
            print("✅ EXCELLENT: 95%+ accuracy achieved!")
        else:
            print("👍 GOOD: Training completed successfully")
            
        # GPU utilization summary
        if torch.cuda.is_available():
            max_memory = torch.cuda.max_memory_allocated() / 1024**3
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            utilization = (max_memory / total_memory) * 100
            print(f"💾 Peak GPU Usage: {max_memory:.1f}GB / {total_memory:.1f}GB ({utilization:.1f}%)")
            
            if utilization < 60:
                print("✅ SAFE: Low GPU usage - no termination risk")
            elif utilization < 80:
                print("⚠️  MODERATE: Monitor for potential issues")
            else:
                print("🚨 HIGH: Consider reducing batch size next time")
    else:
        print("⚠️  No training history available")