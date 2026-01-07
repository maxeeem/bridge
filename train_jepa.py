import gymnasium as gym
import minigrid
import torch
import torch.optim as optim
import numpy as np
from jepa_components import JEPALight
import random

def collect_data(steps=2000):
    print(f"Collecting {steps} steps of data...")
    env = gym.make("MiniGrid-Empty-5x5-v0", render_mode=None)
    data = []
    
    obs, _ = env.reset()
    # obs is dict, we want obs['image'] which is (7, 7, 3)
    current_image = obs['image']
    
    for _ in range(steps):
        # random policy
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        
        next_image = next_obs['image']
        data.append((current_image, action, next_image))
        
        current_image = next_image
        
        if terminated or truncated:
            obs, _ = env.reset()
            current_image = obs['image']
            
    env.close()
    return data

def train():
    # settings
    batch_size = 64
    epochs = 500
    tau = 0.99
    lr = 1e-3
    
    # 1. Data Collection
    raw_data = collect_data(2000)
    
    # Pre-process data into tensors
    # Observations: (N, 7, 7, 3) -> (N, 3, 7, 7)
    obs_list = [d[0] for d in raw_data]
    act_list = [d[1] for d in raw_data]
    next_obs_list = [d[2] for d in raw_data]
    
    obs_tensor = torch.tensor(np.array(obs_list), dtype=torch.float32).permute(0, 3, 1, 2)
    act_tensor = torch.tensor(np.array(act_list), dtype=torch.float32) # will be cast to long for index
    next_obs_tensor = torch.tensor(np.array(next_obs_list), dtype=torch.float32).permute(0, 3, 1, 2)
    
    dataset_size = len(raw_data)
    
    # 2. Initialize JEPA
    jepa = JEPALight()
    optimizer = optim.Adam(jepa.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    
    print("Starting training...")
    for epoch in range(epochs):
        # Sample batch
        indices = np.random.choice(dataset_size, batch_size, replace=True)
        
        batch_s = obs_tensor[indices]
        batch_a = act_tensor[indices]
        batch_s_next = next_obs_tensor[indices]
        
        # Forward
        pred_next_s, target_next_s = jepa(batch_s, batch_a, batch_s_next)
        
        loss = loss_fn(pred_next_s, target_next_s)
        
        # Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # EMA Update
        with torch.no_grad():
            for param_online, param_target in zip(jepa.online_encoder.parameters(), jepa.target_encoder.parameters()):
                param_target.data = tau * param_target.data + (1 - tau) * param_online.data
                
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
            
    # 3. Save
    torch.save(jepa.online_encoder.state_dict(), "jepa_retina.pth")
    print("Model saved to jepa_retina.pth")

if __name__ == "__main__":
    train()
