import numpy as np
import matplotlib.pyplot as plt

def main():

    # Load logs
    timesteps = np.load("/home/ritwik/Desktop/MuJoCo_Quant/logs/results/128_128/evaluations.npz")["timesteps"]  # shape: (evals,)
    results = np.load("/home/ritwik/Desktop/MuJoCo_Quant/logs/results/128_128/evaluations.npz")["results"]  # shape: (evals, n_eval_episodes)

    # Mean and std
    mean_rewards = results.mean(axis=1)
    std_rewards = results.std(axis=1)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(timesteps, mean_rewards, label="Mean reward")
    plt.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.3)
    plt.xlabel("Timesteps")
    plt.ylabel("Evaluation Reward")
    plt.title("Policy Evaluation Over Time")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()