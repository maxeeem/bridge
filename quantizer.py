import numpy as np

class DynamicEventMap:
    def __init__(self, input_dim=64):
        self.prototypes = [] # List of numpy arrays
        # Each prototype will be associated with an ID based on its index: event_0, event_1, ...
        self.vigilance = 0.5
        self.input_dim = input_dim
        self.learning_rate = 0.1

    def _get_id(self, index):
        return f"event_{index}"

    def quantize(self, vector: np.array) -> str:
        """
        Maps a vector to a discrete token.
        - If distance to nearest prototype < vigilance, return existing ID and adapt.
        - Else, create new prototype.
        """
        if len(self.prototypes) == 0:
            self._add_prototype(vector)
            return self._get_id(0)

        # Find nearest
        distances = [np.linalg.norm(p - vector) for p in self.prototypes]
        min_dist_idx = np.argmin(distances)
        min_dist = distances[min_dist_idx]

        if min_dist < self.vigilance:
            # Match found
            # Adapt the prototype: p = p + alpha * (v - p)
            self.prototypes[min_dist_idx] += self.learning_rate * (vector - self.prototypes[min_dist_idx])
            return self._get_id(min_dist_idx)
        else:
            # No match within vigilance, create new
            new_idx = self._add_prototype(vector)
            return self._get_id(new_idx)

    def _add_prototype(self, vector):
        self.prototypes.append(vector.copy())
        return len(self.prototypes) - 1

    def adjust_vigilance(self, error: float):
        """
        If error is high (surprise), tighten vigilance (lower threshold) to make finer distinctions.
        If error is low, we could relax it (optional, but requirement says 'If error is high ... lower threshold').
        """
        if error > 0:
            # Check for high error
            # Simple adaptive logic: e.g., decrease vigilance by 10% of current value or fixed step
            # Requirement: "If error is high ... lower the threshold"
            
            # We decay vigilance
            decay = 0.05 * error
            self.vigilance = max(0.01, self.vigilance - decay)
            # print(f"High error ({error}), adjusting vigilance to {self.vigilance:.4f}")
        else:
            # Optional: Relax vigilance slowly if no error? 
            # The prompt only specified dealing with high error.
            pass
