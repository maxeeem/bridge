import time
import sys
from nars_interface import OnaBackend, ActionMapper
from quantizer import DynamicEventMap
import numpy as np

def verify_patches():
    print("=== Verification 1: Mapping Test (Static) ===")
    mapper = ActionMapper()
    
    op = "^left"
    mapped = mapper.map_action(op)
    print(f"Mapping {op} -> {mapped}")
    
    if mapped == 0:
        print("PASS: Forward map correct")
    else:
        print(f"FAIL: Expected 0, got {mapped}")

    op = "^activate"
    mapped = mapper.map_action(op)
    print(f"Mapping {op} -> {mapped}")
    if mapped == 5:
        print("PASS: Custom map correct")
    else:
        print(f"FAIL: Expected 5, got {mapped}")

    print("\n=== Verification 2: Surprise Logic (Live ONA) ===")
    nars = OnaBackend(executable_path="OpenNARS-for-Applications/NAR")
    if not nars.running:
        print("FAIL: Could not start ONA")
        return

    nars.send_input("*volume=100")
    
    # Train strict rule A->B
    print("Training A->B with high confidence...")
    for i in range(10):
        nars.send_input("<tick --> seen>. :|:")
        nars.send_input("5")
        nars.send_input("<tock --> seen>. :|:")
        nars.send_input("10")
        
    print("Injecting Betrayal (A -> Z)...")
    nars.get_prediction_error() # Clear old error
    
    nars.send_input("<tick --> seen>. :|:")
    nars.send_input("5") # Wait for B expectation
    
    # Betrayal!
    nars.send_input("<tack --> seen>. :|:")
    time.sleep(0.5)
    
    # Wait steps to ensure processing
    nars.send_input("5")
    time.sleep(0.5)
    
    err = nars.get_prediction_error()
    print(f"Prediction Error: {err}")
    
    if err > 0.0:
        print(f"PASS: Error signal detected ({err})")
    else:
        print("FAIL: Still zero error.")
        
    # Cleanup detection
    derived = nars.get_derived()
    if len(derived) > 0:
        print(f"Info: System derived {len(derived)} items.")
        
    nars.process.kill()

    print("\n=== Verification 3: Quantizer Persistence ===")
    q = DynamicEventMap()
    vec = np.array([0.5, 0.5])
    t1 = q.quantize(vec, current_step=10)
    print(f"Created {t1}")
    
    q.save("test_q.pkl")
    
    q2 = DynamicEventMap()
    q2.load("test_q.pkl")
    
    if len(q2.prototypes) == 1:
        print("PASS: Loaded 1 prototype")
    else:
        print(f"FAIL: Loaded {len(q2.prototypes)}")
        
    if q2.last_used[0] == 10:
        print("PASS: Steps preserved")
    else:
        print(f"FAIL: Steps lost ({q2.last_used[0]})")
        
    print("Testing Pruning...")
    # Prune stuff older than 5 steps
    q2.prune(current_step=20, age_threshold=5)
    if len(q2.prototypes) == 0:
        print("PASS: Pruned successfully")
    else:
        print("FAIL: Pruning failed")

if __name__ == "__main__":
    verify_patches()
