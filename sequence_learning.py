
import time
import random
from opennars_interface import OpenNarsBackend
from physics import LatentWorld

def run_experiment():
    # 1. Setup
    print("Initializing OpenNARS...")
    nars = OpenNarsBackend()
    if not nars.running:
        print("Failure: Could not start OpenNARS.")
        return

    world = LatentWorld()
    
    # 2. Training Phase
    print("\n--- Training Phase (300 steps) ---")
    print("Feeding A -> B -> C sequence...")
    
    for i in range(300):
        obs = world.step()
        # Feed observation to NARS
        # Format: <obs --> seen>. :|:
        inp = f"<{obs} --> seen>. :|:"
        if i % 50 == 0:
            print(f"  Step {i}: Input {inp}")
        nars.send_input(inp)
        # OpenNARS generally runs continuously in shell, so we just wait
        
        # Give OpenNARS a moment to process (Java is slower)
        time.sleep(0.1) 
        
        # Clear buffers
        nars.get_derived()
        nars.get_anticipations()

    # 3. Verification Phase
    print("\n--- Verification Phase ---")
    print("Ground Truth Mapping: A -> B, B -> C, C -> A")
    
    test_transitions = [
        ("A", "B"),
        ("B", "C"),
        ("C", "A")
    ]
    
    for input_sym, expected_sym in test_transitions:
        # Input Token(A)
        inp = f"<{input_sym} --> seen>. :|:"
        nars.send_input(inp)
        # Allow deeper inference time
        
        # Wait for inference (poll for result)
        derived_events = []
        anticipations = []
        for _ in range(20): # Poll for 2 seconds
             time.sleep(0.1)
             derived_events.extend(nars.get_derived())
             anticipations.extend(nars.get_anticipations())
        
        # Summarize derived for cleaner output
        derived_summary = [d[:50] + "..." if len(d) > 50 else d for d in derived_events]
        print(f"Input: <{input_sym}>. Derived: {len(derived_events)} events. Anticipated: {anticipations}. Expected: <{expected_sym}>.")
        
        # Pass Condition: Check derivations AND anticipations
        found_in_derived = False
        found_in_anticipated = False
        
        for d in derived_events:
            if expected_sym in d:
                found_in_derived = True
        
        for a in anticipations:
            if expected_sym in a:
                found_in_anticipated = True

        if found_in_derived or found_in_anticipated:
            source = []
            if found_in_derived: source.append("Derived")
            if found_in_anticipated: source.append("Anticipated")
            print(f"  -> PASS: Prediction found in {source}.")
        else:
            print(f"  -> FAIL: Expected symbol not found in derivations or anticipations.")

    # 4. Cleanup
    nars.process.kill()
    print("Experiment finished.")

if __name__ == "__main__":
    run_experiment()
