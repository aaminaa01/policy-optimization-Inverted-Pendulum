from imports import *

# Define the parameter space
learning_rate_range = [1e-5, 1e-4, 1e-3]
gamma_range = [0.9, 0.99, 0.999]
clip_range_range = [0.1, 0.2, 0.3]
ent_coef_range = [0.0, 0.01]
n_steps_range = [128, 256, 512, 1024]

# Create the environment
vec_env = make_vec_env('InvertedPendulum-v5', n_envs=4)

# Perform Random Search
best_score = -float('inf')
best_params = None
for _ in range(10):  # Try 10 random combinations
    params = {
        'learning_rate': random.choice(learning_rate_range),
        'gamma': random.choice(gamma_range),
        'clip_range': random.choice(clip_range_range),
        'ent_coef': random.choice(ent_coef_range),
        'n_steps': random.choice(n_steps_range)
    }
    model = PPO('MlpPolicy', vec_env, **params, verbose=1)
    model.learn(total_timesteps=100000)
    mean_reward, _ = evaluate_policy(model, vec_env, n_eval_episodes=10)
    if mean_reward > best_score:
        best_score = mean_reward
        best_params = params

print(f"Best parameters: {best_params}")
print(f"Best reward: {best_score}")
