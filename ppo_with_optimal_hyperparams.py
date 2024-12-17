# Import the necessary libraries
from imports import *
from callback_functions import ProgressCallback
from config import *
from plots import *

# Instantiate the callback
callback = ProgressCallback()

# Create a Gym environment
# Parallel environments to speed up and stabilize training
vec_env = make_vec_env("InvertedPendulum-v5", n_envs=4)

# Initialize the PPO model
model = PPO("MlpPolicy", vec_env, n_steps=optimal_n_steps, verbose=1, learning_rate=optimal_learning_rate, gamma=optimal_gamma, clip_range=optimal_clip_range, ent_coef=optimal_ent_coef)

"""Evaluating the agent before training it"""

# Random Agent, before training
mean_reward, std_reward = evaluate_policy(
    model,
    vec_env,
    n_eval_episodes=N_EVAL_EPISODES,
)
print(f"Stats before training the agent:\nmean_reward={mean_reward:.2f} +/- {std_reward}\n\n")

"""Training the agent"""

# Train the model for a specified number of timesteps
model.learn(total_timesteps=TRAINING_TIMESTEPS, progress_bar=True, callback=callback)
print("Training complete\n\n")

"""Evaluating the agent after training it"""

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=N_EVAL_EPISODES)
print(f"Stats after training the agent:\nMean reward: {mean_reward} +/- {std_reward}\n\n")

"""Plotting training stats"""

# Plot both on separate subplots
plot_returns('optimal', callback)
plot_training_losses('optimal', callback)
plot_value_deltas('optimal', callback)

"""Recording a video of the trained agent"""
# obs = vec_env.reset()
# video_folder = "logs/videos/"
# video_length = 100

# ev_env = DummyVecEnv([lambda: gym.make("InvertedPendulum-v5", render_mode="rgb_array")])


# # Record the video starting at the first step
# ev_env = VecVideoRecorder(ev_env, video_folder,
#                        record_video_trigger=lambda x: x == 0, video_length=video_length,
#                        name_prefix=f"random-agent-InvertedPendulum-v5")



