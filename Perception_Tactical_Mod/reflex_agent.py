import torch
import torch.nn as nn
import numpy as np

# A lightweight MLP for tactical execution
class CaRLNetwork(nn.Module):
    def __init__(self, input_dim=130, action_dim=2): # 64 waypoints (x,y) + speed + yaw
        super(CaRLNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh() # Outputs normalized [-1, 1]
        )

    def forward(self, x):
        return self.net(x)

class ReflexAgent:
    def __init__(self, weights_path="weights/active_reflex.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CaRLNetwork().to(self.device)
        self.model.load_state_dict(torch.load(weights_path))
        self.model.eval()

    def get_action(self, waypoints, ego_dynamics):
        """
        waypoints: List of [x, y] from Governor (64 points)
        ego_dynamics: dict containing {'speed': float, 'yaw': float}
        """
        if waypoints is None:
            return 0.0, 0.0 # Emergency fallback

        # 1. Pre-process: Flatten waypoints and add ego-state
        wp_flat = np.array(waypoints).flatten() # 128 values
        state = np.append(wp_flat, [ego_dynamics['speed'], ego_dynamics['yaw']])
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # 2. Inference
        with torch.no_grad():
            action = self.model(state_tensor).cpu().numpy()[0]

        # 3. Rescale Output: [target_speed (m/s), target_steer (-1 to 1)]
        target_speed = (action[0] + 1.0) * 10.0 # Scaled to 0-20 m/s
        target_steer = action[1] 

        return target_speed, target_steer

    def reload_weights(self, path):
        """Used by the Rewind Script to hot-swap weights after an audit failure."""
        self.model.load_state_dict(torch.load(path))
        print(f"Reflex Agent: Weights reverted to {path}")