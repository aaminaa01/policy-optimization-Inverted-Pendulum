from imports import *

def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def plot_returns(configuration, callback):
    # Ensure the directory exists
    create_dir(f"logs/ppo/{configuration}/train")

    # Plot Returns (Episode Rewards)
    plt.figure(figsize=(12, 6))
    plt.plot(callback.returns, label='Episode Reward')
    plt.xlabel('Episode')
    plt.ylabel('Training Step')
    plt.title('Episode Rewards Over Time')
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig(f"logs/ppo/{configuration}/train/training_returns.png")

def plot_training_losses(configuration, callback):
    # Ensure the directory exists
    create_dir(f"logs/ppo/{configuration}/train")

    # Plot Losses (Network Losses)
    plt.figure(figsize=(12, 6))
    plt.plot(callback.losses, label='Loss', color='red')
    plt.xlabel('Update Step')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig(f"logs/ppo/{configuration}/train/training_losses.png")

def plot_value_deltas(configuration, callback):
    # Ensure the directory exists
    create_dir(f"logs/ppo/{configuration}/train")

    plt.figure(figsize=(12, 6))
    plt.plot(callback.value_losses, label='Delta in Value Estimate')
    plt.xlabel('Update Step')
    plt.ylabel('Delta')
    plt.title('Delta in Value Estimates Across Updates')
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig(f"logs/ppo/{configuration}/train/training_value_deltas.png")

# TO-DO: Add a function to plot the training policy loss of the agent
# TO-DO: Add a function to plot the training entropy loss of the agent  
