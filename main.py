from app.simulation import Simulation
import pandas as pd

if __name__ == "__main__":
    # --- Configuration ---
    config = {
        "dataset": "mnist",  # 'mnist' or 'cifar10'
        "num_clients": 10,
        "byzantine_pct": 0.2,
        "attack_type": "sign_flipping", # 'sign_flipping' or 'gaussian'
        "is_iid": False,  # Non-IID for realistic federated learning
        "num_rounds": 50,  # Full training for 97%+ accuracy
        "local_epochs": 3,  # Multiple local epochs per round
        
        # Client training parameters
        "client_lr": 0.001,  # Client learning rate
        "client_optimizer": "adam",  # 'adam' or 'sgd'
        "batch_size": 32,
        "weight_decay": 1e-4,
        
        # Q-learning parameters
        "learning_rate": 0.1,
        "discount_factor": 0.9,
        "epsilon_start": 1.0,
        "epsilon_decay": 0.995,
        "epsilon_min": 0.01,
        
        # Trust mechanism parameters
        "trust_beta": 0.5,
        "trust_params": {
            "w_sim": 0.4,    # Weight for cosine similarity
            "w_loss": 0.4,   # Weight for loss divergence  
            "w_norm": 0.2,   # Weight for gradient norm
            "norm_threshold": 5.0  # Gradient norm threshold
        },
        
        # Model persistence options
        "use_pretrained": True,
        "save_model": True,
        "force_retrain": False,
        
        # Training enhancements
        "use_scheduler": True,  # Learning rate scheduling
        "early_stopping": True,  # Early stopping based on validation
        "patience": 10,  # Patience for early stopping
    }

    # --- Run Simulation ---
    simulation = Simulation(config)
    history = simulation.run()

    # --- Save Results ---
    df = pd.DataFrame(history)
    df.to_csv("simulation_results.csv", index=False)
    print("\nResults saved to simulation_results.csv")
