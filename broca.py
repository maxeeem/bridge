import time
import numpy as np
import random
from nars_interface import OnaBackend
from quantizer import DynamicEventMap
import os

def main():
    # 1. Initialize Components
    # Assuming NAR executable is in the OpenNARS-for-Applications directory
    nar_path = os.path.join(os.path.dirname(__file__), "OpenNARS-for-Applications", "NAR")
    
    print(f"Initializing Broca Bridge...")
    nars = OnaBackend(executable_path=nar_path)
    if not nars.running and nars.process is None:
        print("CRITICAL: Failed to start NAR. Exiting mock loop.")
        # Proceeding might be useless but we'll try to run logic to show flow
        # In this case NarsBackend checks running flag and ignores sending.

    event_map = DynamicEventMap(input_dim=64)

    print("Starting Main Loop (50 steps)...")

    for step in range(50):
        # 2. Mock Physics Layer (Delta Vector)
        # Generate random vector. 
        # To simulate *some* structure, let's pick from a few 'hidden' true clusters + noise
        cluster_centers = [
            np.zeros(64),
            np.ones(64),
            np.array([1 if i % 2 == 0 else 0 for i in range(64)])
        ]
        chosen_center = random.choice(cluster_centers)
        noise = np.random.normal(0, 0.1, 64)
        delta_vector = chosen_center + noise

        # 3. Quantizer Layer
        token = event_map.quantize(delta_vector)
        
        # 4. Logic Layer (ONA)
        narsese_input = f"<{token}> . :|:"
        # print(f"Step {step}: Input {narsese_input}")
        nars.send_input(narsese_input)

        # Give ONA a moment to process (since we are communicating via pipes)
        time.sleep(0.1) 

        # 5. Read Output / Feedback
        action = nars.get_action()
        if action:
            print(f"Step {step}: ONA Executed Action -> {action}")
        else:
             # Just for debug visualization, maybe print nothing or dot
             pass

        error = nars.get_prediction_error()
        if error > 0:
            print(f"Step {step}: Surprise detected! (Error={error})")
            event_map.adjust_vigilance(error)

    print("Broca Loop Complete.")
    
    # Clean up (OnaBackend kills process on destruct or we can let it die)
    # nars.process.kill() is handled in __init__? No, let's rely on OS cleanup or explicit close if we added it.
    if nars.process:
        nars.process.terminate()

if __name__ == "__main__":
    main()
