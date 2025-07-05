from app.simulation import Simulation
import pandas as pd

if __name__ == "__main__":
    # --- Configuration ---
    config = {
        "dataset": "mnist",  # 'mnist' or 'cifar10'
        "num_clients": 10,
        "byzantine_pct": 0.2,
        "attack_type": "sign_flipping", # 'sign_flipping' or 'gaussian'
        "is_iid": True,
        "num_rounds": 3,
        "learning_rate": 0.1,
        "discount_factor": 0.9,
        "epsilon_start": 1.0,
        "epsilon_decay": 0.99,
        "epsilon_min": 0.01,
        "trust_beta": 0.5,
        
        # Model persistence options
        "use_pretrained": True,    # Load pre-trained model if available
        "save_model": True,        # Save model after training
        "force_retrain": False,    # Force retraining even if model exists
    }

    # --- Run Simulation ---
    simulation = Simulation(config)
    history = simulation.run()

    # --- Save Results ---
    df = pd.DataFrame(history)
    df.to_csv("simulation_results.csv", index=False)
    print("\nResults saved to simulation_results.csv")
