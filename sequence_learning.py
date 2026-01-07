
import time
import random
from nars_interface import OnaBackend
from physics import LatentWorld

def run_experiment():
    # 1. Setup
    print("Initializing ONA...")
    nars = OnaBackend(executable_path="OpenNARS-for-Applications/NAR")
    if not nars.running:
        print("Failure: Could not start ONA.")
        return

    world = LatentWorld()
    
    # 2. Training Phase
    print("\n--- Training Phase (300 steps) ---")
    print("Feeding A -> B -> C sequence...")
    
    for i in range(300):
        obs = world.step()
        # Feed observation to NARS
        # Format: <obs>. :|:
        inp = f"<{obs}>. :|:"
        if i % 50 == 0:
            print(f"  Step {i}: Input {inp}")
        nars.send_input(inp)
        nars.send_input("10") # Run 10 cycles to process
        
        # Give ONA a moment to process
        time.sleep(0.05) 
        
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
        inp = f"<{input_sym}>. :|:"
        nars.send_input(inp)
        nars.send_input("100") # Run 100 cycles to allow deeper inference
        
        # Wait for inference (poll for result)
        derived_events = []
        for _ in range(20): # Poll for 2 seconds
             time.sleep(0.1)
             derived_events.extend(nars.get_derived())
        
        print(f"Input: <{input_sym}>. ONA Derived: {derived_events}. Expected: <{expected_sym}>.")
        
        # Pass Condition: Does derived_events contain Token(B)?
        found = False
        for d in derived_events:
            if expected_sym in d:
                found = True
                break
        
        if found:
            print(f"  -> PASS: Prediction derived.")
        else:
            print(f"  -> FAIL: Expected symbol not found in derivations.")

    # 4. Cleanup
    nars.process.kill()
    print("Experiment finished.")

if __name__ == "__main__":
    run_experiment()
