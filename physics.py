
class LatentWorld:
    """
    A simple specialized physics simulator for the sequence learning experiment.
    It cycles through states: A -> B -> C -> A ...
    It ignores actions for now (autonomous evolution), or we could make it reactive.
    For the A->B->C experiment, it's usually just a time-based sequence.
    """
    def __init__(self):
        self.states = ["A", "B", "C"]
        self.current_index = 0
        self.tick_count = 0

    def step(self, action=None):
        """
        Advances the world one step.
        Returns the new observation.
        """
        self.tick_count += 1
        
        # Simple cyclic deterministic transition
        self.current_index = (self.current_index + 1) % len(self.states)
        
        return self.observe()

    def observe(self):
        return self.states[self.current_index]

    def reset(self):
        self.current_index = 0
        self.tick_count = 0
        return self.observe()
