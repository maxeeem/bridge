import subprocess
import time

def explore():
    process = subprocess.Popen(
        ["./OpenNARS-for-Applications/NAR", "shell"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
        universal_newlines=True
    )

    # Set volume
    process.stdin.write("*volume=100\n")
    process.stdin.flush()

    # Train sequence A->B
    cmds = []
    for _ in range(5):
        cmds.append("<A> . :|:")
        cmds.append("<B> . :|:")
    
    # Trigger expectation A->?
    cmds.append("<A> . :|:")
    # Betrayal B is missing, C appears
    cmds.append("<C> . :|:")

    for cmd in cmds:
        process.stdin.write(cmd + "\n")
        process.stdin.flush()
        time.sleep(0.1)

    # Allow buffer to flush
    time.sleep(1)
    process.terminate()

    print("--- OUTPUT START ---")
    while True:
        line = process.stdout.readline()
        if not line: break
        print(line.strip())
    print("--- OUTPUT END ---")

if __name__ == "__main__":
    explore()