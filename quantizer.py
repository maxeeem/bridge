import numpy as np
import pickle
import os

class DynamicEventMap:
    def __init__(self, input_dim=64):
        self.prototypes = [] # List of numpy arrays
        # Each prototype will be associated with an ID based on its index: event_0, event_1, ...
        self.vigilance = 0.5
        self.input_dim = input_dim
        self.learning_rate = 0.1
        
        # Pruning / Usage tracking
        self.usage_counts = []
        self.last_used = [] # Updates with step count

    def _get_id(self, index):
        return f"event_{index}"

    def save(self, filename="quantizer.pkl"):
        """Serialize state to file."""
        data = {
            "prototypes": self.prototypes,
            "vigilance": self.vigilance,
            "usage_counts": self.usage_counts,
            "last_used": self.last_used
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"Quantizer saved to {filename}")

    def load(self, filename="quantizer.pkl"):
        """Load state from file."""
        if not os.path.exists(filename):
            print(f"Warning: {filename} not found.")
            return
        
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            
        self.prototypes = data.get("prototypes", [])
        self.vigilance = data.get("vigilance", 0.5)
        self.usage_counts = data.get("usage_counts", [0]*len(self.prototypes))
        self.last_used = data.get("last_used", [0]*len(self.prototypes))
        print(f"Quantizer loaded from {filename} ({len(self.prototypes)} prototypes)")

    def quantize(self, vector: np.array, current_step=0) -> str:
        """
        Maps a vector to a discrete token.
        - If distance to nearest prototype < vigilance, return existing ID and adapt.
        - Else, create new prototype.
        """
        if len(self.prototypes) == 0:
            idx = self._add_prototype(vector, current_step)
            return self._get_id(idx)

        # Find nearest
        distances = [np.linalg.norm(p - vector) for p in self.prototypes]
        min_dist_idx = np.argmin(distances)
        min_dist = distances[min_dist_idx]

        if min_dist < self.vigilance:
            # Match found
            # Adapt the prototype: p = p + alpha * (v - p)
            self.prototypes[min_dist_idx] += self.learning_rate * (vector - self.prototypes[min_dist_idx])
            
            # Update Statistics
            self.usage_counts[min_dist_idx] += 1
            self.last_used[min_dist_idx] = current_step
            
            return self._get_id(min_dist_idx)
        else:
            # No match within vigilance, create new
            new_idx = self._add_prototype(vector, current_step)
            return self._get_id(new_idx)

    def _add_prototype(self, vector, step=0):
        self.prototypes.append(vector.copy())
        self.usage_counts.append(1)
        self.last_used.append(step)
        return len(self.prototypes) - 1

    def prune(self, current_step, age_threshold=1000):
        """
        Removes prototypes not used for `age_threshold` steps.
        Rebuilds lists to keep indices valid, forcing ID shift (Event 5 -> Event 4).
        NARS is expected to relearn the new IDs.
        """
        keep_indices = []
        removed_count = 0
        
        for i, last in enumerate(self.last_used):
            if current_step - last <= age_threshold:
                keep_indices.append(i)
            else:
                removed_count += 1
        
        if removed_count > 0:
            # Rebuild all lists
            self.prototypes = [self.prototypes[i] for i in keep_indices]
            self.usage_counts = [self.usage_counts[i] for i in keep_indices]
            self.last_used = [self.last_used[i] for i in keep_indices]
            print(f"Pruned {removed_count} prototypes. New total: {len(self.prototypes)}")

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
