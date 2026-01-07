import pickle
import numpy as np
import re
import math
import os

def analyze():
    # 1. Mining the Log
    log_path = "ona_minigrid.log"
    if not os.path.exists(log_path):
        print(f"Error: {log_path} not found.")
        return

    print("Mining ONA logs...")
    
    # Regex to find Implications where consequent is <goal --> seen>
    # We look for lines starting with "Derived:" or "Revised:"
    # Pattern: ... <... =/> <goal --> seen>> ... confidence=0.xyz
    
    # We want to capture the condition part. 
    # Example: <<event_0 --> seen> =/> <goal --> seen>>
    # Example: <(<event_1 --> seen> &/ <event_0 --> seen>) =/> <goal --> seen>>
    # Example procedural: <((<event_0 --> seen> &/ ^left) =/> <goal --> seen>>
    
    # We'll use a fairly broad regex and then process the condition
    # Capture group 1: condition
    # Capture group 2: confidence
    pattern = re.compile(r"<(.*) =/> <goal --> seen>>.*confidence=([0-9\.]+)")
    
    best_rule = None
    best_conf = -1.0
    best_event = None
    
    with open(log_path, 'r') as f:
        for line in f:
            if "=/> <goal --> seen>" in line:
                match = pattern.search(line)
                if match:
                    condition = match.group(1)
                    conf = float(match.group(2))
                    
                    if conf > best_conf:
                        # Extract basic event ID from condition, e.g. "event_0" from "<event_0 --> seen>"
                        # or from complex conditions, pick the first event found
                        event_match = re.search(r"event_([0-9]+)", condition)
                        if event_match:
                            best_rule = line.strip()
                            best_conf = conf
                            best_event = f"event_{event_match.group(1)}"

    if not best_event:
        print("No strong rules found leading to the goal.")
        return

    print(f"\nTop Rule Found:\n{best_rule}")
    print(f"Confidence: {best_conf}")
    print(f"Extracted Trigger Event: {best_event}")

    # 2. Decoding the Concept
    print("\nDecoding Concept...")
    
    try:
        with open("knowledge.pkl", 'rb') as f:
            quantizer_data = pickle.load(f)
            # quantizer save format: {"prototypes": [], ...}
            prototypes = quantizer_data.get("prototypes", [])
            
        with open("encoder.pkl", 'rb') as f:
            projection_matrix = pickle.load(f)
            
    except FileNotFoundError as e:
        print(f"Error loading pickle files: {e}")
        return

    # Get event index
    event_idx = int(best_event.split('_')[1])
    
    if event_idx >= len(prototypes):
        print(f"Error: {best_event} not found in prototypes (len={len(prototypes)})")
        return
        
    vector = prototypes[event_idx] # Shape (64,)
    
    # 3. Visual Reconstruction
    # I approx v @ E^dagger
    # E is (147, 64)
    # v is (64,)
    # pinv(E) is (64, 147)
    # But wait: v = I @ E
    # v @ E.T ?? No.
    # v = I @ E
    # I @ E @ E_pinv = v @ E_pinv
    # If E has linearly independent columns (64 < 147), E_pinv @ E = I_64? No.
    # E_pinv = (E.T @ E)^-1 @ E.T  (Left inverse? No, E is tall)
    # For tall matrix A (m > n), A x = b. x = (A^T A)^-1 A^T b.
    # Here I (1, 147) @ E (147, 64) = v (1, 64)
    # We want to solve for I.
    # This is an underdetermined system if we treat columns of E as equations? No.
    # We have 64 equations and 147 unknowns.
    # reconstruction = v @ pinv(E.T) ?
    # Let's use numpy's pinv.
    
    # E is (147, 64)
    # E_pinv is (64, 147)
    # We want I such that I @ E = v
    # Transpose: E.T @ I.T = v.T
    # E.T is (64, 147). I.T is (147, 1). v.T is (64, 1).
    # This is underdetermined (more unknowns than equations).
    # pinv will give the minimum norm solution.
    # I.T = pinv(E.T) @ v.T
    # I = (I.T).T = v @ pinv(E.T).T = v @ pinv(E) is not quite right dimensionally if not careful
    
    # Correct math:
    # I (1x147) * P (147x64) = V (1x64)
    # I = V * P_pseudo_inverse ?
    # Let P_dag = pinv(P) -> shape (64, 147)
    # Result = V (1x64) * P_dag (64, 147) -> (1x147)
    
    E_pinv = np.linalg.pinv(projection_matrix.T) # (147, 64)
    # Wait, np.linalg.pinv(A) returns A+
    # If A is (M, N), A+ is (N, M).
    # P is (147, 64). pinv(P) is (64, 147).
    # V is (64,).
    # I = V @ pinv(P) -> (147,)
    
    P_inv = np.linalg.pinv(projection_matrix) # (64, 147)
    reconstruction = np.dot(vector, P_inv) # (147,)
    
    re_image = reconstruction.reshape(7, 7, 3)
    
    # 4. Visualization (ASCII Heatmap of intensity)
    # We sum absolute values across RGB channels to get intensity
    intensity_map = np.linalg.norm(re_image, axis=2)
    
    # Normalize to 0-1
    _min = intensity_map.min()
    _max = intensity_map.max()
    if _max > _min:
        intensity_map = (intensity_map - _min) / (_max - _min)
    else:
        intensity_map = np.zeros((7,7))

    print("\nVisual Reconstruction of the Concept (Heatmap):")
    chars = " .:-=+*#%@"
    
    print("+" + "---" * 7 + "+")
    for row in range(7):
        line_str = "|"
        for col in range(7):
            val = intensity_map[row, col]
            char_idx = int(val * (len(chars) - 1))
            char = chars[char_idx]
            line_str += f" {char} "
        print(line_str + "|")
    print("+" + "---" * 7 + "+")
    
    print(f"\nInterpretation: High intensity regions (statistically) correspond to the Visual pattern that triggers the Goal.")

if __name__ == "__main__":
    analyze()
