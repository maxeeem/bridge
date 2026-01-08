# Neuro-Symbolic Bridge (NARS + JEPA)

## Description
This project implements a **Concept-Centered Knowledge Representation (CCKR)** system. It bridges the gap between Sub-symbolic Perception (Neural Networks) and Symbolic Reasoning (NARS) to create an agent capable of learning, reasoning, and tool use in the MiniGrid environment.

## The "Middle-Out" Architecture
1.  **The Retina (JEPA):** A Joint Embedding Predictive Architecture that compresses 7x7x3 pixel grids into "Physics Vectors" based on affordances/predictability.
2.  **The Bridge (Quantizer):** A Growing Neural Gas (GNG) that clusters continuous vectors into discrete symbols (e.g., `event_0`, `event_1`).
3.  **The Brain (NARS):** A logic engine (ONA or OpenNARS) that learns causal rules (`<event_0 --> seen> =/> <goal --> seen>`) and drives decision-making.

---

## 1. Installation

### Prerequisites
* Python 3.10+
* [OpenNARS-for-Applications (ONA)](https://github.com/opennars/OpenNARS-for-Applications) binary (compiled as `NAR`) OR [OpenNARS 3.1.0 JAR](https://github.com/opennars/opennars/releases).

### Python Dependencies
```bash
pip install gymnasium minigrid torch numpy

```

### File Setup

Ensure your directory structure looks like this:

```text
/bridge
  ├── NAR                   # ONA Executable (optional)
  ├── opennars.jar          # OpenNARS JAR (optional)
  ├── minigrid_bridge.py    # Main Agent Loop
  ├── train_jepa.py         # Retina Training Script
  ├── jepa_components.py    # PyTorch Neural Definitions
  ├── encoders.py           # Neural Encoder Wrapper
  ├── quantizer.py          # Dynamic Event Map (GNG)
  ├── nars_interface.py     # ONA Interface
  ├── opennars_interface.py # OpenNARS Interface
  └── analyze_jepa.py       # Visualization/Debugging Tool

```

---

## 2. Usage Guide

### Phase 1: The School (Train Vision)

Before the agent can reason, it must learn to see. This script collects random gameplay data and trains the JEPA to understand the "Physics" of the grid (e.g., walls stop movement).

```bash
python3 train_jepa.py

```

* **Output:** Generates `jepa_retina.pth` (The trained weights).

### Phase 2: The Mission (Run Agent)

Launch the agent into the environment. It will start with "blank" concepts but a "trained" eye.

**Basic Run (Empty Room):**

```bash
python3 minigrid_bridge.py --backend ona --episodes 10

```

**The Final Boss (Door & Key):**
Test the agent's ability to transfer knowledge and use tools.

```bash
python3 minigrid_bridge.py --env MiniGrid-DoorKey-5x5-v0 --episodes 15

```

* **Flags:**
* `--backend`: `ona` (Fast C++) or `opennars` (Java 3.0/3.1).
* `--env`: Any MiniGrid environment ID.
* `--jar`: Path to jar file if using OpenNARS backend.



### Phase 3: The Autopsy (Visualize Thoughts)

See what the agent is actually thinking. This script finds the logic rule associated with the goal and reconstructs the visual image of that concept using Nearest Neighbor Decoding.

```bash
python3 analyze_jepa.py

```

* **Output:** ASCII Art showing what `event_X` looks like to the agent.

---

## 3. Key Components

* **`jepa_components.py`**: Contains the `JEPALight` PyTorch model. Uses a Siamese network structure with an Exponential Moving Average (EMA) target to learn robust representations without negative sampling.
* **`quantizer.py`**: The **Symbol Grounding** engine. It maps the continuous 64-dim vectors from the JEPA to discrete Narsese terms. It supports persistence (`save/load`) and pruning (forgetting unused concepts).
* **`nars_interface.py` & `opennars_interface.py**`: The translation layers. They handle **Action Mapping** (converting `^pickup` to ID `3`) and **Surprise Detection** (monitoring prediction errors to trigger learning).

## 4. Credits

* **SeL-NAL**: Architecture inspired by *Self-Learning in Non-Axiomatic Logic* (Hammer et al.).
* **JEPA**: Vision system based on *Joint Embedding Predictive Architectures* (LeCun).
