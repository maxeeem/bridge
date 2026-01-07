import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, embedding_dim=64):
        super(Encoder, self).__init__()
        # Input is (B, 3, 7, 7)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        # Using 32 channels, 7x7 image -> 32 * 7 * 7 = 1568
        self.flatten_dim = 32 * 7 * 7
        self.fc = nn.Linear(self.flatten_dim, embedding_dim)
        
    def forward(self, x):
        # x: (B, 3, 7, 7)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class Predictor(nn.Module):
    def __init__(self, embedding_dim=64, action_dim=7):
        super(Predictor, self).__init__()
        # Concatenate state embedding and one-hot action
        self.net = nn.Sequential(
            nn.Linear(embedding_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim)
        )
        self.action_dim = action_dim
        self.embedding_dim = embedding_dim

    def forward(self, state_embedding, action):
        # state_embedding: (B, 64)
        # action: (B,) containing integer action indices
        
        # Create one-hot encoding for action
        action_one_hot = F.one_hot(action.long(), num_classes=self.action_dim).float()
        
        # Concatenate
        x = torch.cat([state_embedding, action_one_hot], dim=1)
        return self.net(x)

class JEPALight(nn.Module):
    def __init__(self, embedding_dim=64, action_dim=7):
        super(JEPALight, self).__init__()
        self.online_encoder = Encoder(embedding_dim)
        self.target_encoder = Encoder(embedding_dim)
        
        # Initialize target encoder with same weights as online encoder
        self.target_encoder.load_state_dict(self.online_encoder.state_dict())
        
        # Disable gradient computation for target encoder
        for param in self.target_encoder.parameters():
            param.requires_grad = False
            
        self.predictor = Predictor(embedding_dim, action_dim)
        
    def forward(self, s_t, a_t, s_next=None):
        """
        If s_next is provided, returns (pred_next_s, target_next_s).
        Otherwise returns pred_next_s.
        """
        # Online encoding of current state
        state_embed = self.online_encoder(s_t)
        
        # Prediction of next state embedding
        pred_next_s = self.predictor(state_embed, a_t)
        
        if s_next is not None:
             with torch.no_grad():
                target_next_s = self.target_encoder(s_next)
             return pred_next_s, target_next_s
        
        return pred_next_s
