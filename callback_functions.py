from imports import *

class ProgressCallback(BaseCallback):
  # Callback function to store returns and losses during training

    def __init__(self):
        super(ProgressCallback, self).__init__()
        self.returns = []  # Store returns (episode rewards)
        self.losses = []   # Store network loss (if available)
        self.value_losses = [] # Store delta between MC estimate and actual value fucntion

    def _on_step(self) -> bool:
        # Access the `infos` dictionary for reward tracking, tracks rewards per epsiode
        if "episode" in self.locals["infos"][0]:
            episode_reward = self.locals["infos"][0]["episode"]["r"]
            self.returns.append(episode_reward)
        # Tracks training loss per update
        if hasattr(self.model, "logger") and self.model.logger:
          loss = self.model.logger.name_to_value['train/loss']
          if loss is not None:
             self.losses.append(loss)
        # Tracks value loss per update
        if hasattr(self.model, "logger") and self.model.logger:
          loss = self.model.logger.name_to_value['train/value_loss']
          if loss is not None:
             self.value_losses.append(loss)
        
        return True