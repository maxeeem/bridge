import time
from nars_interface import OnaBackend
import os

def stress_test():
    # Setup
    nar_path = os.path.join(os.path.dirname(__file__), "OpenNARS-for-Applications", "NAR")
    nars = OnaBackend(executable_path=nar_path)
    if not nars.running:
        print("Test Skipped: NAR not found.")
        return

    print("Step 1: Training (<tick> -> <tock>)")
    for i in range(10):
        nars.send_input("<tick> . :|:")
        time.sleep(0.05)
        nars.send_input("<tock> . :|:")
        time.sleep(0.05)
        
        # Clear error to avoid accumulator noise
        _ = nars.get_prediction_error()

    # Step 2: The Betrayal
    print("Step 2: The Betrayal (<tick> -> <boom>)")
    
    # Send tick - Likely triggers reinforcement revision of tick->tock
    nars.send_input("<tick> . :|:")
    time.sleep(0.1) 
    
    # Send boom
    nars.send_input("<boom> . :|:")
    time.sleep(0.2) # Allow time for ONA to think and print

    # Step 3: Assertion
    error = nars.get_prediction_error()
    print(f"Prediction Error: {error}")

    if error > 0:
        print("PASS: Surprise/Revision detected.")
    else:
        print("FAIL: No surprise detected.")

    if nars.process:
        nars.process.terminate()

if __name__ == "__main__":
    stress_test()