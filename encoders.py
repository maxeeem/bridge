import numpy as np
import os
import pickle
import torch
from jepa_components import Encoder

class VisualEncoder:
    def __init__(self):
        # 147 input features (7x7x3), 64 output dimensions
        # Create a fixed, random projection matrix (shape 147 x 64) using np.random.normal.
        self.projection_matrix = np.random.normal(size=(147, 64))

    def encode(self, obs_image):
        """
        Convert the 7x7x3 MiniGrid pixel grid into a 64-dim vector.
        """
        # Flatten the 7x7x3 image to a 147-vector.
        flat_obs = obs_image.flatten().astype(np.float32)
        
        # Multiply by the projection matrix.
        encoded_vector = np.dot(flat_obs, self.projection_matrix)
        
        # Normalize: Divide by the L2 norm (magnitude) so the vector lies on the unit sphere.
        norm = np.linalg.norm(encoded_vector)
        if norm > 0:
            encoded_vector = encoded_vector / norm
            
        return encoded_vector

    def save(self, filename="encoder.pkl"):
        with open(filename, 'wb') as f:
            pickle.dump(self.projection_matrix, f)
        print(f"Encoder saved to {filename}")

    def load(self, filename="encoder.pkl"):
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                self.projection_matrix = pickle.load(f)
            print(f"Encoder loaded from {filename}")
        else:
            print(f"Encoder file {filename} not found, keeping random initialization.")

class JEPAVisualEncoder:
    def __init__(self, model_path="jepa_retina.pth"):
        self.encoder = Encoder(embedding_dim=64)
        if os.path.exists(model_path):
            self.encoder.load_state_dict(torch.load(model_path))
            print(f"JEPA Encoder loaded from {model_path}")
        else:
            print(f"Warning: {model_path} not found. Using random initialization.")
        self.encoder.eval()

    def encode(self, obs_image):
        # input obs_image is numpy (7, 7, 3)
        # Convert to tensor (1, 3, 7, 7) for the model
        obs_tensor = torch.tensor(obs_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        
        with torch.no_grad():
            emb = self.encoder(obs_tensor)
            
        # Convert back to numpy
        encoded_vector = emb.squeeze(0).numpy()
        
        # L2 Normalize
        norm = np.linalg.norm(encoded_vector)
        if norm > 0:
            encoded_vector = encoded_vector / norm
            
        return encoded_vector
