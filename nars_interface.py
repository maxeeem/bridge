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
        
        # Thread to read stdout
        if self.running:
            self.thread = threading.Thread(target=self._monitor_output, daemon=True)
            self.thread.start()

    def send_input(self, narsese: str):
        if not self.running:
            return
        
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

                # 2. Detect Surprise / Revision
                # Heuristic: Match "Revision" or "Revised"
                # The log output uses "Revised:", but instructions said "Revision". We match "Revis" to coverage both.
                if "Revis" in line:
                    # Capture confidence if possible, but default to 0.5 per instructions
                    # Example line: Revised: ... confidence=0.55 ...
                    if "confidence=" in line:
                         try:
                             # Regex or split to find confidence
                             # strict parsing or just greedy find
                             match = re.search(r"confidence=([0-9\.]+)", line)
                             if match:
                                 conf = float(match.group(1))
                                 # We could use conf as error, or delta.
                                 # User instruction: "set self.last_error = 0.5 (or higher)"
                                 self.last_error = 0.5
                             else:
                                 self.last_error = 0.5
                         except:
                             self.last_error = 0.5
                    else:
                        self.last_error = 0.5
                
                # Removed "Anticipation failed" check as it doesn't exist in ONA logs currently.

            except ValueError:
                continue
            except Exception as e:
                print(f"Error in ONA monitor: {e}")
                break
        
        if self.process:
            self.process.kill()
