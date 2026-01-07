import subprocess
import threading
import queue
import time
import re
import os
from nars_interface import NarsBackend

class OpenNarsBackend(NarsBackend):
    def __init__(self, jar_path="opennars.jar"):
        self.jar_path = jar_path
        if not os.path.exists(self.jar_path):
             print(f"Warning: {self.jar_path} not found.")
        
        try:
            # Launch OpenNARS
            # -Xmx1024m is from instructions
            self.process = subprocess.Popen(
                ["java", "-Xmx1024m", "-jar", self.jar_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=1,
                universal_newlines=True
            )
            self.running = True

        except FileNotFoundError:
            print("Java or JAR not found, entering pure mock mode (no subprocess)")
            self.process = None
            self.running = False
        except Exception as e:
            print(f"Error launching OpenNARS: {e}")
            self.process = None
            self.running = False

        self.last_action = None
        self.last_error = 0.0
        self.last_derived = []
        self.anticipations = []
        
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
        self.last_action = None 
        return action
    
    def get_derived(self):
        derived = self.last_derived[:]
        self.last_derived = []
        return derived
    
    def get_anticipations(self):
        ants = self.anticipations[:]
        self.anticipations = []
        return ants

    def get_prediction_error(self) -> float:
        error = self.last_error
        self.last_error = 0.0
        return error

    def _monitor_output(self):
        while self.running:
            try:
                line = self.process.stdout.readline()
                if not line:
                    break
                
                line = line.strip()
                if not line:
                    continue

                # 1. Capture Derived and Anticipations
                if line.startswith("OUT:"):
                    # We store the raw content after OUT: to ensure we don't miss anything due to regex failure
                    # The consumer (sequence_learning.py) just checks substring presence "B" in string.
                    content = line[4:].strip()
                    self.last_derived.append(content)
                elif line.startswith("Answer:"):
                    content = line[7:].strip()
                    self.last_derived.append(content)
                elif "ANTICIPATE:" in line:
                    parts = line.split("ANTICIPATE:")
                    if len(parts) > 1:
                        content = parts[1].strip()
                        self.anticipations.append(content)
                        # Also try to parse anticipations specifically if needed
                        # <(Term) =/> (Term)>.
                        if "=/>" in content:
                            self.anticipations.append((0.0, content)) # Dummy score for now
                
                # 2. Operations (EXE:)
                # Example: EXE: ^left
                if line.startswith("EXE:"):
                    parts = line.split(":")
                    if len(parts) > 1:
                        op = parts[1].strip()
                        self.last_action = op

            except Exception as e:
                print(f"Error reading OpenNARS output: {e}")
                self.running = False
