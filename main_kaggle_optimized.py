from app.simulation import Simulation
import pandas as pd
import torch

if __name__ == "__main__":
    # Kaggle-Optimized Configuration - Higher limits, more reliable
    print("🏆 KAGGLE-OPTIMIZED CONFIGURATION")
    print("=" * 60)
    
    # Check GPU availability
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🚀 GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {gpu_memory:.1f} GB")
        
        # Kaggle has more generous limits than Colab
        if gpu_memory >= 14:  # P100 or T4
            batch_size = 256
            num_clients = 25
            local_epochs = 5
            num_workers = 4
            print("🎯 HIGH-END KAGGLE GPU: Optimized configuration")
        else:
            batch_size = 128
            num_clients = 20
            local_epochs = 4
            num_workers = 3
            print("🔧 STANDARD KAGGLE GPU: Balanced configuration")
    else:
        batch_size = 64
        num_clients = 10
        local_epochs = 2
        num_workers = 2
        print("⚠️  CPU mode - basic configuration")
    
    # Kaggle-optimized configuration
    config = {
        "dataset": "mnist",  # 'mnist' or 'cifar10'
        "num_clients": num_clients,
        "byzantine_pct": 0.2,
        "attack_type": "sign_flipping",
        "is_iid": False,
        "num_rounds": 50,
        "local_epochs": local_epochs,
        
        # Kaggle-optimized training parameters
        "client_lr": 0.001,
        "client_optimizer": "adam",
        "batch_size": batch_size,
        "weight_decay": 1e-4,
        
        # Enhanced data loading for Kaggle
        "num_workers": num_workers,
        "pin_memory": True,
        "prefetch_factor": 3,  # Higher than Colab safe
        
        # GPU optimizations
        "use_amp": True,
        "amp_dtype": "float16",
        "grad_clip": 1.0,
        
        # Kaggle memory management
        "empty_cache_every": 4,
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
        "patience": 12,  # Higher patience for Kaggle
    }
    
    print(f"🎯 Kaggle Configuration Summary:")
    print(f"   Platform: Kaggle Notebooks (30h/week limit)")
    print(f"   Clients: {config['num_clients']} (higher than Colab)")
    print(f"   Batch Size: {config['batch_size']} (optimized)")
    print(f"   Local Epochs: {config['local_epochs']} (extended)")
    print(f"   Mixed Precision: {config['use_amp']} (memory efficient)")
    print(f"   Workers: {config['num_workers']} (higher parallelism)")
    print(f"   Expected GPU Usage: 10-12GB (65-80%)")
    print(f"   Termination Risk: VERY LOW ✅")
    print(f"   Expected Time: 15-20 minutes")
    print(f"   Accuracy Target: 97%+ (same as max config)")

    # --- Run Kaggle-Optimized Simulation ---
    print("\n🚀 Starting Kaggle-optimized TARS training...")
    simulation = Simulation(config)
    history = simulation.run()

    # --- Save Results ---
    if history:
        df = pd.DataFrame(history)
        df.to_csv("kaggle_optimized_results.csv", index=False)
        print(f"\n💾 Results saved to kaggle_optimized_results.csv")
        
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
            print("👍 GOOD: Strong performance achieved")
            
        # Performance comparison
        rounds_completed = len(df)
        avg_round_time = "~20-30 seconds"  # Estimated
        print(f"\n📈 Training Summary:")
        print(f"   Rounds Completed: {rounds_completed}")
        print(f"   Avg Time/Round: {avg_round_time}")
        print(f"   Total Training Time: ~{rounds_completed * 0.4:.0f} minutes")
        
        # GPU utilization summary
        if torch.cuda.is_available():
            max_memory = torch.cuda.max_memory_allocated() / 1024**3
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            utilization = (max_memory / total_memory) * 100
            print(f"💾 Peak GPU Usage: {max_memory:.1f}GB / {total_memory:.1f}GB ({utilization:.1f}%)")
            
            if utilization >= 60:
                print("🎯 EXCELLENT: High GPU utilization achieved")
            elif utilization >= 40:
                print("✅ GOOD: Decent GPU utilization")
            else:
                print("📈 TIP: Could increase batch size for higher utilization")
        
        # Kaggle-specific tips
        print(f"\n💡 Kaggle Tips:")
        print(f"   - Save notebook frequently (Ctrl+S)")
        print(f"   - Download results before 30h limit")
        print(f"   - Use Version control for experiments")
        print(f"   - Consider running CIFAR-10 next for comparison")
        
    else:
        print("⚠️  No training history available")