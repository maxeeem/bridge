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
                # Heuristic: Match "Anticipation failed" or "Revision" (if printed)
                # Note: ONA might not print "Anticipation failed" explicitly in all versions.
                # Use "decision expectation" as a sign of activity, but "Revision" implies belief change.
                if "Anticipation failed" in line:
                    self.last_error = 1.0
                elif "Revision" in line: 
                    # If "Revision" is explicitly logged (user heuristic request)
                    self.last_error = 0.5 # Arbitrary "high" score for revision
                
                # Check for "CONFIRM" vs "DISCONFIRM" if available, or just ignore for now 
                # as ONA default logging is sparse on this.

            except ValueError:
                continue
            except Exception as e:
                print(f"Error in ONA monitor: {e}")
                break
        
        if self.process:
            self.process.kill()
