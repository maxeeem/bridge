import gymnasium as gym
import minigrid
import torch
import numpy as np
import pickle
import re
import os
from encoders import JEPAVisualEncoder
from quantizer import DynamicEventMap

# --- Configuration ---
LOG_FILE = "ona_minigrid.log"
KNOWLEDGE_FILE = "knowledge.pkl"
MODEL_PATH = "jepa_retina.pth"

def ascii_render(image):
    """
    Render a 7x7x3 MiniGrid image as ASCII art.
    Channel 0: Object ID
    Channel 1: Color ID
    """
    # Map Object IDs to characters
    # 1: Empty, 2: Wall, 8: Goal, 9: Lava, 10: Agent
    # We can also use colors if needed. 
    # Prompt: Green=G, Grey=#, Empty=.
    
    # Standard Minigrid Mappings (approximate)
    # TYPE_MAP = {
    #     1: '.', # Empty
    #     2: '#', # Wall
    #     8: 'G', # Goal
    # }
    
    DISPLAY_MAP = {
        1: '.',  # Empty
        2: '#',  # Wall
        3: '.',  # Floor
        8: 'G',  # Goal
        9: 'L',  # Lava
        10: 'A', # Agent
    }
    
    H, W, C = image.shape
    lines = []
    
    # Minigrid obs are usually (W, H, C) or (H, W, C)?
    # Observations are (7, 7, 3).
    # Typically encoded as [col][row].
    
    for y in range(H):
        line = ""
        for x in range(W):
            obj_id = image[x, y, 0] # Minigrid uses [col][row]
            char = DISPLAY_MAP.get(obj_id, '?')
            
            # Refine based on color if it's '?' or generic
            # Color 1 is Green (Goal)
            # Color 5 is Grey (Wall)
            color_id = image[x, y, 1]
            
            if char == '?' and color_id == 5:
                char = '#'
            if char == '?' and color_id == 1:
                char = 'G'
                
            line += " " + char
        lines.append(line)
        
    return "\n".join(lines)


def main():
    # Step 1: Load the Brain
    print("Loading Brain...")
    
    # Load Quantizer
    if not os.path.exists(KNOWLEDGE_FILE):
        print(f"Error: {KNOWLEDGE_FILE} not found.")
        return
    
    event_map = DynamicEventMap()
    event_map.load(KNOWLEDGE_FILE)
    prototypes = event_map.prototypes
    print(f"Loaded {len(prototypes)} prototypes.")
    
    # Load JEPA Visual Encoder
    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found.")
        return
    encoder = JEPAVisualEncoder(MODEL_PATH)
    
    # Step 2: Generate Reference Data
    print("Generating Reference Data (The Rosetta Stone)...")
    env = gym.make("MiniGrid-Empty-5x5-v0", render_mode=None)
    
    reference_data = [] # List of (image, vector)
    
    obs, _ = env.reset()
    for _ in range(1000):
        # We need diverse observations. Random actions help.
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        
        image = obs['image'] # (7,7,3)
        vector = encoder.encode(image)
        reference_data.append((image, vector))
        
        if terminated or truncated:
            obs, _ = env.reset()
            
    env.close()
    print(f"Generated {len(reference_data)} reference samples.")
    
    # Step 3: Extract Rules from Logic Log
    print(f"Scanning {LOG_FILE} for rules...")
    
    if not os.path.exists(LOG_FILE):
        print(f"Error: {LOG_FILE} not found.")
        return

    # We are looking for implications ending in <goal --> seen>
    # Regex for Narsese implication: <(... concept ...) =/> <goal --> seen>>
    # Note: ONA logs might look like: 
    # Derived: dt=... <(... antecedent ...) =/> <goal --> seen>>. Priority=... Truth: ...
    
    best_rule = None
    best_confidence = -1.0
    target_event = None
    
    # Simple regex to capture antecedent of <... =/> <goal --> seen>>
    # Assumes no nested =/> in antecedent for simplicity, or just greedy match
    regex = re.compile(r"<(?P<ant>.*?) =/> <goal --> seen>>.*confidence=(?P<conf>0\.\d+)")
    
    with open(LOG_FILE, 'r') as f:
        for line in f:
            match = regex.search(line)
            if match:
                ant = match.group('ant')
                conf = float(match.group('conf'))
                
                # We want the strongest rule (highest confidence)
                if conf > best_confidence:
                    best_confidence = conf
                    best_rule = ant
    
    if best_rule:
        print(f"Strongest Rule found (conf={best_confidence}): {best_rule} => Goal")
        
        # Extract event ID from antecedent
        # Antecedent could be "<event_5 --> seen>" or "(&/, <event_5 --> seen>, ...)"
        # Let's simple search for "event_(\d+)"
        event_match = re.search(r"event_(\d+)", best_rule)
        if event_match:
            event_idx = int(event_match.group(1))
            target_event = f"event_{event_idx}"
            print(f"Antecedent Concept Identified: {target_event}")
            
            # Check if we have a prototype for this event
            if event_idx < len(prototypes):
                target_vector = prototypes[event_idx]
            else:
                print(f"Error: Prototype for {target_event} not found (index {event_idx} >= {len(prototypes)}).")
                target_event = None
        else:
            print("Could not extract event ID from rule.")
            
    else:
        print("No rule leading to <goal --> seen> found.")
        
    # Step 4: Decode and Visualize
    if target_event is not None:
        print(f"Visualizing {target_event}...")
        
        # Calculate Cosine Similarity
        # Sim(A, B) = dot(A, B) / (norm(A)*norm(B))
        # Vectors are already normalized by Encoder/Quantizer, but let's be safe.
        
        scores = []
        target_norm = np.linalg.norm(target_vector)
        
        for img, vec in reference_data:
            vec_norm = np.linalg.norm(vec)
            if target_norm > 0 and vec_norm > 0:
                sim = np.dot(target_vector, vec) / (target_norm * vec_norm)
            else:
                sim = 0
            scores.append(sim)
            
        # Get Top 3
        top_indices = np.argsort(scores)[::-1][:3]
        
        print(f"\nNARS believes {target_event} leads to Goal.")
        print(f"Top 3 Visual Matches (Similarity > 0.9 suggests strong match):")
        
        for i, idx in enumerate(top_indices):
            print(f"\n--- Match {i+1} (Score: {scores[idx]:.4f}) ---")
            print(ascii_render(reference_data[idx][0]))
            
if __name__ == "__main__":
    main()
