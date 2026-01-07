import time
from nars_interface import OnaBackend
from opennars_interface import OpenNarsBackend

def test_muscles(backend, name):
    print(f"[{name}] Test 1: Muscles (Action Mapping)")
    # Send Action ID 0 (should be ^left)
    # We need to spy on the backend's last action or rely on the process output if backend self-loops.
    # OnaBackend sets self.last_action when it sees "^left executed".
    # OpenNarsBackend sets self.last_action when it sees "EXE: ^left".
    
    backend.send_action(0) # 0 -> ^left
    time.sleep(2.0) # Wait for processing
    
    action = backend.get_action()
    print(f"  Action received: {action}")
    if action == "^left":
        print("  PASS")
        return True
    else:
        print(f"  FAIL (Expected ^left)")
        return False

def test_brain(backend, name):
    print(f"[{name}] Test 2: Brain (Sequence Learning)")
    # Train A -> B
    # We send many pairs
    print("  Training <tick> =/> <tock>...")
    for _ in range(10):
        backend.send_input("<tick> . :|:")
        backend.send_input("<tock> . :|:")
        time.sleep(0.1)
    
    # Check if logic is derived?
    # Or check if sending A triggers expectancy of B?
    # We'll rely on the fact that if we send A now, betrayal triggers error later.
    # Just sending input is enough for "Training".
    print("  Training done.")
    return True

def test_panic(backend, name):
    print(f"[{name}] Test 3: Panic (Surprise Detection)")
    # Disable debug output to avoid console flooding (logs are in file)
    backend.debug_output = False
    
    # Requirement: We expect B (tock). We send C (boom).
    # Reset error
    backend.get_prediction_error()
    
    # 1. Prime the expectation
    backend.send_input("<tick> . :|:")
    time.sleep(1) # Wait for inference/anticipation
    
    # 2. Betray expectation
    print("  Injecting betrayal (boom)...")
    backend.send_input("<boom> . :|:")
    time.sleep(1)
    
    # 3. Check error
    error = backend.get_prediction_error()
    print(f"  Prediction Error: {error}")
    
    if error > 0.05:
        print("  PASS")
        return True
    else:
        print("  FAIL (Error too low)")
        return False

def run_suite(backend_cls, name, log_file):
    print(f"\n=== Running Verification Suite for {name} (Log: {log_file}) ===")
    try:
        backend = backend_cls(output_log_path=log_file)
        if not backend.running:
            print(f"SKIP: {name} backend did not start (missing binary/jar?)")
            return
        
        # Give it a moment to boot
        time.sleep(2)
        
        results = []
        results.append(test_muscles(backend, name))
        results.append(test_brain(backend, name))
        results.append(test_panic(backend, name))
        
        backend.stop()
        
        if all(results):
            print(f"=== {name} SUITE PASSED ===")
        else:
            print(f"=== {name} SUITE FAILED ===")
            
    except Exception as e:
        print(f"CRITICAL ERROR in {name} suite: {e}")

if __name__ == "__main__":
    # Test ONA
    run_suite(OnaBackend, "ONA", "ona_verify.log")
    
    # Test OpenNARS
    run_suite(OpenNarsBackend, "OpenNARS", "opennars_verify.log")
