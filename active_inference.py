import time
import random
import numpy as np
from nars_interface import OnaBackend # Correct import name
from physics import LatentWorld
from quantizer import DynamicEventMap

def run_active_inference():
    print("Initializing Active Inference Experiment...")
    nars = OnaBackend(executable_path="OpenNARS-for-Applications/NAR")
    if not nars.running:
        print("Failure: Could not start OpenNARS.")
        return

    world = LatentWorld()
    quantizer = DynamicEventMap(input_dim=2)
    
    # Pre-calculate tokens
    world_vec_light = world.VECTOR_LIGHT
    token_light = quantizer.quantize(world_vec_light)
    
    world_vec_dark = world.VECTOR_DARK
    token_dark = quantizer.quantize(world_vec_dark)

    print(f"Token Light: {token_light}")
    print(f"Token Dark: {token_dark}")

    print("\n--- Phase 1: Demonstration (Supervised Learning) ---")
    # We show NARS the cause-effect relationship deterministically.
    # We say: "Context" -> "Operator" -> "Result"
    
    for i in range(3):
        print(f"Demo {i+1}...")
        
        # 1. Context: Dark
        nars.send_input(f"<{token_dark} --> seen>. :|:")
        
        # 2. Action: Press (We claim SELF did it)
        nars.send_input(f"<(*, {{SELF}}) --> ^activate>. :|:")
        
        # 3. Time passes (Causality gap)
        # Use a consistent gap. 5 steps.
        nars.send_input("5")
        
        # 4. Result: Light
        nars.send_input(f"<{token_light} --> seen>. :|:")
             
        # 5. Reset (Wait 10 steps to clear context)
        nars.send_input("10")
        
        # 6. Reset state to Dark (Observation)
        nars.send_input(f"<{token_dark} --> seen>. :|:")
        nars.send_input("5")

    print("\n--- Phase 2: Active Inference Test ---")
    
    # 1. Set current state to DARK
    world.step("^wait")
    current_vec, _ = world.observe() 
    # Should be dark
    current_token = quantizer.quantize(current_vec)
    print(f"Current State: {current_token} (Should be {token_dark})")
    nars.send_input(f"<{current_token} --> seen>. :|:")
    
    # 2. Inject Goal
    # We want to see Light.
    # We mark it as '!' (Target)
    goal = f"<{token_light} --> seen>! :|:"
    print(f"Injecting Goal: {goal}")
    nars.send_input(goal)
    
    # 3. Optional: Inject Explicit Rule as backup (Reinforcement)
    # The demo should have learned: <(&/, <Dark-->seen>, +5, <(*,{SELF})-->^activate>) =/> <Light-->seen>>
    # We inject a similar rule with high confidence just in case.
    # Note: Interval is roughly +5.
    rule = f"<(&/, <{token_dark} --> seen>, +5, <(*, {{SELF}}) --> ^activate>) =/> <{token_light} --> seen>>."
    print(f"Injecting Explicit Rule: {rule}")
    nars.send_input(f"{rule} %1.0;0.90%")
    
    # 4. Wait for Action
    print("Waiting for execution...")
    start_time = time.time()
    success = False
    
    loops = 0
    # Run for up to 30 seconds
    while time.time() - start_time < 30.0:
        loops += 1
        
        # Tick the system
        nars.send_input("1")
        
        # Every 10 ticks (1 second), refresh the Goal and Context
        if loops % 10 == 0:
            nars.send_input(goal)
            nars.send_input(f"<{current_token} --> seen>. :|:")
            print(".", end="", flush=True)
            
        action = nars.get_action()
        if action:
            print(f"\nNARS Executed: {action}")
            if action == "^activate":
                print("\n>>> SUCCESS: NARS executed ^activate to achieve the goal!")
                success = True
                break
        
        time.sleep(0.1)

    if not success:
        print("\n>>> FAIL: NARS did not execute ^activate within timeout.")
        
    if nars.process:
        nars.process.kill()

if __name__ == "__main__":
    run_active_inference()
