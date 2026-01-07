import subprocess
import time

def explore_betrayal():
    process = subprocess.Popen(
        ["./OpenNARS-for-Applications/NAR", "shell"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
        universal_newlines=True
    )

    process.stdin.write("*volume=100\n")
    
    # 1. Train tick->tock strongly
    for i in range(10):
        process.stdin.write("<tick> . :|:\n")
        time.sleep(0.01)
        process.stdin.write("<tock> . :|:\n")
        time.sleep(0.01)
        process.stdin.flush()

    time.sleep(0.5)
    # Drain buffer crudely
    import fcntl
    import os
    fd = process.stdout.fileno()
    fl = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

    try:
        while True:
            chunk = process.stdout.read(1024)
            if not chunk: break
    except:
        pass
    
    print("\n--- BETRAYAL START ---")
    process.stdin.write("<tick> . :|:\n")
    process.stdin.flush()
    time.sleep(0.1)
    process.stdin.write("<boom> . :|:\n")
    process.stdin.flush()
    time.sleep(1)
    
    process.terminate()

    # Read remaining
    print("Capturing output...")
    while True:
        line = process.stdout.readline()
        if not line: break
        if "REVISION" in line.upper() or "DERIVED" in line.upper() or "INPUT" in line.upper() or "ANTICIPATION" in line.upper():
             print(line.strip())

if __name__ == "__main__":
    explore_betrayal()