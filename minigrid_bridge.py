import gymnasium as gym
import minigrid
import time
import os
import argparse
from nars_interface import OnaBackend, ActionMapper
from opennars_interface import OpenNarsBackend
from quantizer import DynamicEventMap
from encoders import VisualEncoder

def main():
    parser = argparse.ArgumentParser(description="Minigrid Agent with NARS Backend")
    parser.add_argument("--backend", type=str, choices=["ona", "opennars"], default="ona", help="Choose NARS backend: ona or opennars")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to run")
    parser.add_argument("--jar", type=str, default="opennars.jar", help="Path to OpenNARS JAR file (if backend is opennars)")
    args = parser.parse_args()

    # Setup
    # Initialize gym.make("MiniGrid-Empty-5x5-v0", render_mode="human")
    env = gym.make("MiniGrid-Empty-5x5-v0", render_mode="human")

    # Initialize Backend
    if args.backend == "ona":
        print("Initializing ONA Backend...")
        nars = OnaBackend(output_log_path="ona_minigrid.log")
    else:
        print(f"Initializing OpenNARS Backend with {args.jar}...")
        nars = OpenNarsBackend(jar_path=args.jar, output_log_path="opennars_minigrid.log")

    # Initialize VisualEncoder and DynamicEventMap
    mapper = ActionMapper()
    encoder = VisualEncoder()
    event_map = DynamicEventMap()

    # Load knowledge if it exists, else start fresh
    if os.path.exists("knowledge.pkl"):
        event_map.load("knowledge.pkl")
        # We also need to load the encoder to ensure the vector space matches the learned prototypes
        encoder.load("encoder.pkl")
    else:
        print("Starting fresh (no knowledge.pkl found)")

    # The Episode Loop
    num_episodes = args.episodes
    step_cnt = 0

    try:
        for episode in range(num_episodes):
            obs, _ = env.reset()
            print(f"\n--- Episode {episode + 1} Start ---")

            # Goal Injection: Send <goal --> seen>! :|: to NARS.
            nars.send_input("<goal --> seen>! :|:")

            done = False
            truncated = False

            while not (done or truncated):
                step_cnt += 1
                
                # Step Cycle:
                
                # See: Encode obs['image'] -> vector.
                vector = encoder.encode(obs['image'])

                # Concept: Quantize vector -> token (e.g., event_12).
                token = event_map.quantize(vector, current_step=step_cnt)

                # Input: Send <token --> seen>. :|: to NARS.
                nars.send_input(f"<{token} --> seen>. :|:")

                # Small delay to allow NARS to process the input and produce an output
                time.sleep(0.1)

                # Decide: op = nars.get_action().
                op = nars.get_action()

                # Act:
                action_id = -1
                if op is not None:
                    # If op exists: action_id = mapper.map_action(op).
                    print(f"NARS decided: {op}")
                    mapped_id = mapper.map_action(op)
                    if mapped_id != -1:
                        action_id = mapped_id
                    else:
                        # Fallback if mapping fails
                        action_id = env.action_space.sample()
                else:
                    # If op is None: action_id = env.action_space.sample() (Motor Babbling)
                    # OR 2 (Forward bias) to encourage exploring. Let's use random sample for now to test learning.
                    action_id = env.action_space.sample()

                # Execute: obs, reward, done, _, _ = env.step(action_id).
                obs, reward, done, truncated, _ = env.step(action_id)

                # Feedback (The Critical Link):
                if reward > 0:
                    print(f"Goal Achieved! Reward: {reward}")
                    # Send <goal --> seen>. :|: (Result!)
                    nars.send_input("<goal --> seen>. :|:")

                # Optional: slight render delay
                # time.sleep(0.05)

            # Cleanup: Save event_map.save('knowledge.pkl') after every episode.
            print("Episode complete. Saving knowledge...")
            event_map.save("knowledge.pkl")
            encoder.save("encoder.pkl")
            
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        nars.stop()
        env.close()

if __name__ == "__main__":
    main()
