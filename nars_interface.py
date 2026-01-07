import abc
import subprocess
import threading
import queue
import time
import re
import os

class NarsBackend(abc.ABC):
    @abc.abstractmethod
    def send_input(self, narsese: str):
        pass

    @abc.abstractmethod
    def get_action(self):
        """Returns the last executed operation or None."""
        pass

    @abc.abstractmethod
    def get_prediction_error(self) -> float:
        """Returns error score based on surprise/revision."""
        pass

class OnaBackend(NarsBackend):
    def __init__(self, executable_path="./NAR"):
        if not os.path.exists(executable_path):
             # minimal mock for testing when binary check fails, though we prefer real
             print(f"Warning: {executable_path} not found. Running in mock mode internally if launch fails.")
        
        try:
            # We use "shell" argument to enter interactive mode
            self.process = subprocess.Popen(
                [executable_path, "shell"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1,
                universal_newlines=True
            )
            self.running = True

            # Increase ONA volume to get details
            self.send_input("*volume=100")
        except FileNotFoundError:
            print("NAR executable not found, entering pure mock mode (no subprocess)")
            self.process = None
            self.running = False

        self.last_action = None
        self.last_error = 0.0
        self.last_input_term = "" # Track content for relevance filtering
        self.anticipations = [] # List of (score, implication)
        self.last_derived = [] # List of derived terms
        
        # Thread to read stdout
        if self.running:
            self.thread = threading.Thread(target=self._monitor_output, daemon=True)
            self.thread.start()

    def send_input(self, narsese: str):
        if not self.running:
            return
        
        # Extract a simplified "topic" to check against derivations
        # E.g. "<tick> . :|:" -> "tick"
        # We strip outer < > and punctuation
        clean = narsese.strip().replace("<","").replace(">","").split(" ")[0]
        if len(clean) > 0:
            self.last_input_term = clean

        # Ensure input ends with newline
        if not narsese.endswith("\n"):
            narsese += "\n"
        
        try:
            self.process.stdin.write(narsese)
            self.process.stdin.flush()
        except OSError:
            self.running = False

    def get_action(self):
        action = self.last_action
        self.last_action = None # Reset after reading
        return action

    def get_anticipations(self):
        """Returns list of recent anticipations found in stdout."""
        ants = self.anticipations[:]
        self.anticipations = []
        return ants

    def get_derived(self):
        """Returns list of derived terms since last call."""
        derived = self.last_derived[:]
        self.last_derived = []
        return derived

    def get_prediction_error(self) -> float:
        error = self.last_error
        self.last_error = 0.0 # Reset after reading
        return error

    def _monitor_output(self):
        """
        Reads stdout line by line.
        Detects:
        - Executed operations: "^op executed with args"
        - Surprise: "Anticipation failed" or "Revision" (needs heuristics)
        """
        # Regex to detect reinforcement vs contradiction (simple heuristic)
        # Truth: frequency=0.9... confidence=...
        # If frequency drops significantly from 1.0, it's a negative revision (Correction)
        # If frequency stays near 1.0, it's positive (Reinforcement)
        
        while self.running:
            try:
                line = self.process.stdout.readline()
                if not line:
                    break
                
                line = line.strip()
                if not line:
                    continue

                # Debug print to see what's happening
                # print(f"ONA: {line}") 

                # 1. Detect Operations
                # Example: "^left executed with args"
                if " executed with args" in line:
                    # Extract the operation name
                    # Format is usually: <Term> executed with args <Args>
                    # We might need to split carefully.
                    parts = line.split(" executed with args")
                    if len(parts) > 0:
                        op_candidate = parts[0].strip()
                        # Verify it starts with ^ (usually Narsese operations do)
                        if op_candidate.startswith("^"):
                            self.last_action = op_candidate

                # 1.5 Detect Anticipations
                # "decision expectation=0.543 implication: ..."
                if "decision expectation=" in line:
                    try:
                        match = re.search(r"decision expectation=([-0-9\.]+) implication: (.*)", line)
                        if match:
                            score = float(match.group(1))
                            imp = match.group(2).strip()
                            self.anticipations.append((score, imp))
                    except:
                        pass
                
                # "Anticipating: <...>" (as requested by user, though not found in grep)
                if "Anticipating:" in line:
                    self.anticipations.append((0.5, line.strip()))

                # 2. Detect Surprise / Novelty / Revision
                # We want to detect if the system is "surprised" or "learning new things".
                # 1. Negative Revision (Anticipation failure): Frequency drops.
                # 2. Novelty (New Rule): "Derived" with reasonable confidence.
                
                # Check for Revision
                if "Revis" in line:
                    freq_match = re.search(r"frequency=([0-9\.]+)", line)
                    if freq_match:
                        freq = float(freq_match.group(1))
                        # If frequency is low or dropped, it's surprise!
                        if freq < 0.9:
                             self.last_error = 0.8
                    else:
                        # Fallback if parsing fails
                        self.last_error = 0.5
                
                # Check for Derivation (Novelty)
                # If we derive a new strong implication, that's a "Surprise" to the previous model (Schema adaptation)
                if "Derived" in line:
                     # Debug print
                     # print(f"DEBUG ONA: {line}")
                     
                     # Capture derived terms (e.g., used for sequence learning verification)
                     if line.startswith("Derived:"):
                        # Greedy match to capture terms containing '>' like implications =/>
                        match = re.search(r"<(.+)>", line)
                        if match:
                            term = match.group(1)
                            self.last_derived.append(term)

                     # Filter: The derivation must be relevant to the recent input.
                     # If last input was "boom", the derivation should contain "boom".
                     # This prevents background noise (if any) from triggering high vigilance.
                     if self.last_input_term and self.last_input_term not in line:
                         continue

                     # Check confidence to avoid noise
                     # Derived: ... confidence=0.28...
                     try:
                         conf_match = re.search(r"confidence=([0-9\.]+)", line)
                         if conf_match:
                             conf = float(conf_match.group(1))
                             # If we derived something with significant confidence, it's a signal
                             if conf > 0.2: 
                                 # We set a lower error for derivation than direct failure
                                 self.last_error = max(self.last_error, 0.3 * conf)
                     except:
                         pass


            except ValueError:
                continue
            except Exception as e:
                print(f"Error in ONA monitor: {e}")
                break
        
        if self.process:
            self.process.kill()
