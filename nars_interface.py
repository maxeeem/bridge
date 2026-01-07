import abc
import subprocess
import threading
import queue
import time
import re
import os

class ActionMapper:
    """
    Maps NARS operations (strings) to System Action IDs (integers).
    Handles standard ops (^left, ^right) and generic fallbacks.
    """
    def __init__(self):
        self.mapping = {
            "^left": 0,
            "^right": 1,
            "^forward": 2,
            "^move": 2,
            "^pick": 3,
            "^drop": 4,
            "^toggle": 5,
            "^activate": 5, # Mapping ^activate to same as toggle/interact
            "^say": 6,
            "^wait": 7
        }
        # Create reverse mapping (first occurrence wins)
        self.id_to_op = {}
        for op, action_id in self.mapping.items():
            if action_id not in self.id_to_op:
                self.id_to_op[action_id] = op

    def map_action(self, nars_op: str) -> int:
        """
        Returns integer ID for a given NARS operation string.
        Returns -1 if unknown.
        """
        # Clean up string just in case
        op = nars_op.strip()
        return self.mapping.get(op, -1)

    def get_op_for_id(self, action_id: int) -> str:
        """Returns NARS op string for an ID."""
        return self.id_to_op.get(action_id, "^wait")


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

    @abc.abstractmethod
    def stop(self):
        """Stops the backend process."""
        pass

class OnaBackend(NarsBackend):
    def __init__(self, executable_path="./OpenNARS-for-Applications/NAR", output_log_path="ona.log"):
        self.action_mapper = ActionMapper()
        self.output_log_path = output_log_path
        self._log_file = None
        
        if output_log_path:
            try:
                self._log_file = open(output_log_path, "w")
            except Exception as e:
                print(f"Warning: Could not open log file {output_log_path}: {e}")

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

            # Increase ONA volume needed for verify to see output
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

    def stop(self):
        self.running = False
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                self.process.kill()
        
        # Close log file
        if self._log_file:
            try:
                self._log_file.close()
            except:
                pass
        
        # Wait for monitor thread
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def send_action(self, action_id: int):
        """Executes an action by ID."""
        op = self.action_mapper.get_op_for_id(action_id)
        # ONA requires valid Narsese punctuation
        # We send it as an input event/goal
        # Use full self format for consistency with OpenNARS
        self.send_input(f"<(*,{{SELF}}) --> {op}>! :|:")

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
        - Executed operations: "^op executed with args" OR "OUT: (^op"
        - Surprise: "Anticipation failed" or "Revision" or new "OUT:"
        """
        while self.running:
            try:
                line = self.process.stdout.readline()
                if not line:
                    break
                
                line = line.strip()
                if not line:
                    continue

                # Log to file instead of console
                if self._log_file:
                    self._log_file.write(f"{line}\n")
                    self._log_file.flush()

                # Debug print to see what's happening
                # print(f"ONA RAW: {line}") 

                # 1. Detect Operations
                # Case A: Old format or debug "executed with args"
                if " executed with args" in line:
                    parts = line.split(" executed with args")
                    if len(parts) > 0:
                        op_candidate = parts[0].strip()
                        if op_candidate.startswith("^"):
                            self.last_action = op_candidate
                
                # Case A.5: Selected input (Verification for "Muscle Test")
                # If we see "Selected: ^left", it means the system accepted the op input
                # This counts as "checking the wires" even if not executed logic
                if "Selected: " in line:
                    if "^" in line:
                         match = re.search(r"\^([a-zA-Z0-9_]+)", line)
                         if match:
                             self.last_action = "^" + match.group(1)

                # Case B: ONA Shell Format "OUT: (^left,#1)!"
                if line.startswith("OUT:"):
                    # Extract content after OUT:
                    content = line[4:].strip()
                    # Check if it looks like an operation tuple (^op, args)!
                    # Relaxed Regex for ^op anywhere in the content
                    op_match = re.search(r"\^([a-zA-Z0-9_]+)", content)
                    if op_match:
                        op_name = "^" + op_match.group(1)
                        # print(f"DEBUG ONA PARSED ACTION: {op_name} from {line}")
                        self.last_action = op_name
                    else:
                        pass
                        # print(f"DEBUG ONA NO ACTION MATCH in {line}")
                    
                    # 2. Detect Surprise / Novelty / Revision
                    
                    # Capture confidence
                    # Format: %Frequency;Confidence%
                    # e.g. %1.00;0.58%
                    conf_match = re.search(r";([0-9\.]+)%", content)
                    if conf_match:
                        conf = float(conf_match.group(1))
                        # print(f"DEBUG ONA CONF: {conf} from {line}")
                        if conf > 0.3: # Threshold
                            self.last_error = max(self.last_error, 0.3)
                    
                    # Store as derived
                    self.last_derived.append(content)

                # 1.5 Detect Anticipations
                if "decision expectation=" in line:
                    try:
                        match = re.search(r"decision expectation=([-0-9\.]+) implication: (.*)", line)
                        if match:
                            score = float(match.group(1))
                            imp = match.group(2).strip()
                            self.anticipations.append((score, imp))
                    except:
                        pass
                
                if "Anticipating:" in line:
                    self.anticipations.append((0.5, line.strip()))

                # Legacy "Derived:" check
                if "Derived" in line:
                     if line.startswith("Derived:"):
                        match = re.search(r"<(.+)>", line)
                        if match:
                            term = match.group(1)
                            self.last_derived.append(term)
                     
                     if self.last_input_term and self.last_input_term not in line:
                         continue

                     try:
                         conf_match = re.search(r"confidence=([0-9\.]+)", line)
                         if conf_match:
                             conf = float(conf_match.group(1))
                             if conf > 0.1: 
                                 self.last_error = max(self.last_error, 0.3)
                     except:
                         pass

            except ValueError:
                continue
            except Exception as e:
                print(f"Error in ONA monitor: {e}")
                break
        
        if self.process:
            try:
                self.process.kill()
            except:
                pass
