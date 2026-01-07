import time
import sys
from nars_interface import OnaBackend

def test_a_muscle_check(nars):
    print("\n=== Test A: The Muscle Check (Native Operators) ===")
    operators = ["^left", "^right", "^forward", "^move", "^pick", "^drop"]
    supported = []
    unsupported = []

    for op in operators:
        # Clear previous actions
        nars.get_action()
        
        cmd = f"<(*, {{SELF}}) --> {op}>. :|:"
        print(f"Testing {op}...", end="")
        nars.send_input(cmd)
        # Give it a cycle to process
        nars.send_input("1") 
        
        # Give it a moment to flush logic/stdout
        time.sleep(0.1)
        
        action = nars.get_action()
        if action == op:
            print(" EXE: OK")
            supported.append(op)
        else:
            print(f" No Response (Got: {action})")
            unsupported.append(op)
            
    print("-" * 20)
    print(f"Supported: {supported}")
    print(f"Unsupported: {unsupported}")

def test_b_pulse_check(nars):
    print("\n=== Test B: The Pulse Check (Anticipation Visibility) ===")
    # Train A -> B
    # A = <tick --> seen>
    # B = <tock --> seen>
    print("Training A -> B (Sequence)...")
    for i in range(5):
        nars.send_input("<tick --> seen>. :|:")
        nars.send_input("5")
        nars.send_input("<tock --> seen>. :|:")
        nars.send_input("10") # Reset/Gap behavior
        
    print("Triggering A...")
    nars.send_input("<tick --> seen>. :|:")
    nars.send_input("1") # Immediate tick
    
    # Wait for the anticipated window (which was ~5 steps)
    time.sleep(0.2)
    nars.send_input("5")
    time.sleep(0.2)
    
    found_anticipation = False
    
    # Check explicitly captured anticipations
    anticipations = nars.get_anticipations()
    print(f"Captured Anticipations: {anticipations}")
    for score, content in anticipations:
        if "tock" in content or "received" in content: # 'tock' is the content we learned
            found_anticipation = True
            
    # Also check derived rules (showing it learned the connection)
    derived = nars.get_derived()
    # print(f"Captured Derived: {derived}")
    for d in derived:
        if "tick" in d and "tock" in d:
             print(f"Found derived rule: {d}")
             found_anticipation = True
             
    if found_anticipation:
        print("RESULT: PASS")
    else:
        print("RESULT: FAIL (No evidence of B prediction)")

def test_c_alarm_check(nars):
    print("\n=== Test C: The Alarm Check (Surprise Reliability) ===")
    # Train X -> Y
    print("Training X -> Y (Sequence)...")
    for i in range(5):
        nars.send_input("<ping --> seen>. :|:")
        nars.send_input("5")
        nars.send_input("<pong --> seen>. :|:")
        nars.send_input("10")
        
    print("Triggering Betrayal (X -> Z)...")
    # Clear error before betrayal
    nars.get_prediction_error()
    
    input_str = "<ping --> seen>. :|:"
    nars.send_input(input_str)
    nars.send_input("5") # Wait for the moment Y is expected
    
    # Betrayal! We send something else or nothing, but sending something else <pang> ensures noise
    nars.send_input("<pang --> seen>. :|:")
    time.sleep(0.2)
    
    # Wait a few more steps to ensure the mismatch is registered
    nars.send_input("5")
    time.sleep(0.2)
    
    error = nars.get_prediction_error()
    print(f"Prediction Error Value: {error}")
    
    if error > 0:
        print("RESULT: PASS")
    else:
        print("RESULT: FAIL (No error reported)")

def main():
    nars = OnaBackend(executable_path="OpenNARS-for-Applications/NAR")
    if not nars.running:
        print("Could not start ONA.")
        return

    # Ensure volume is high
    nars.send_input("*volume=100")
    
    try:
        test_a_muscle_check(nars)
        nars.send_input("*reset")
        time.sleep(1) 
        
        test_b_pulse_check(nars)
        nars.send_input("*reset")
        time.sleep(1)
        
        test_c_alarm_check(nars)
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        if nars.process:
            nars.process.kill()

if __name__ == "__main__":
    main()
