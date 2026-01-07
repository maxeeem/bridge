
import numpy as np

class LatentWorld:
    """
    A simple specialized physics simulator.
    Can operate in "Sequence Mode" (A->B->C) or "Active Inference Mode" (Light/Dark via Action).
    """
    def __init__(self):
        self.states = ["A", "B", "C"]
        self.current_index = 0
        self.tick_count = 0
        
        # New Active Inference state
        self.light_on = False
        
        # Define Ground Truth vectors (using simple numpy arrays)
        # We can pretend these are high-dim, but for simulation 2D is distinct enough.
        self.VECTOR_DARK = np.array([0.1, 0.1])
        self.VECTOR_LIGHT = np.array([0.9, 0.9])

    def step(self, action=None):
        """
        Advances the world one step.
        Returns the new observation.
        """
        self.tick_count += 1
        
        # Standard Sequence Logic (Preserved if action is None/Wait, or we can mix them)
        # For the Active Inference requirement: "If action_name == ^wait (or None): Set self.light_on = False"
        
        if action == "^press":
            self.light_on = True
            return self.VECTOR_LIGHT, "Light"
        else:
            # ^wait or None
            # Maintain sequence logic if needed, but for active inference we prioritize Light state
            self.current_index = (self.current_index + 1) % len(self.states)
            
            self.light_on = False
            return self.VECTOR_DARK, "Dark"

    def observe(self):
        if self.light_on:
             return self.VECTOR_LIGHT, "Light"
        return self.VECTOR_DARK, "Dark"

    def reset(self):
        self.current_index = 0
        self.tick_count = 0
        self.light_on = False
        return self.observe()
