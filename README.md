# EmergentCreativity

> *A place where we explore if creativity can emerge from randomness.*

A fully-furnished, physics-simulated 3-D apartment that houses a **neural-network "tenant"** trained with Reinforcement Learning.  The tenant perceives the world through all **five simulated senses**, acts on configurable biological drives (hunger, energy, bladder, happiness), and learns emergent behaviours through a rule-based reward system anyone can edit — no programming knowledge required.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Quick Start](#quick-start)
5. [Interactive Viewer](#interactive-viewer)
6. [Training the Agent](#training-the-agent)
7. [Configuring Rewards](#configuring-rewards)
8. [Five Senses](#five-senses)
9. [Tenant Vitals & Actions](#tenant-vitals--actions)
10. [Neural Network Design](#neural-network-design)
11. [Requirements](#requirements)
12. [Hardware Notes](#hardware-notes)

---

## Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Apartment (PyBullet)                         │
│  ┌─────────────┐  ┌─────────────┐  ┌───────────┐  ┌────────────┐  │
│  │ Living Room │  │   Bedroom   │  │  Kitchen  │  │  Bathroom  │  │
│  │  sofa, TV   │  │  bed, desk  │  │ fridge,   │  │  toilet,   │  │
│  │  books,     │  │  books,     │  │ stove,    │  │  shower    │  │
│  │  mess items │  │  controller │  │ table     │  │            │  │
│  └─────────────┘  └─────────────┘  └───────────┘  └────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                             ↕ senses / actions
                    ┌─────────────────────────┐
                    │       Tenant (RL Agent)  │
                    │  Vision CNN ──► LSTM ──► Actor  → action
                    │  Hearing MLP ──► Attention      → logits
                    │  Touch / Smell / Taste ──► Critic → value
                    └─────────────────────────┘
                             ↕ reward signal
                    ┌─────────────────────────┐
                    │   Reward Ruleset (YAML)  │
                    │  cleanliness, hunger,    │
                    │  energy, entertainment … │
                    └─────────────────────────┘
```

The simulation runs in real time on standard desktop hardware.  GPU training
is supported via PyTorch CUDA (with automatic CPU fallback).

---

## Architecture

| Layer | Technology | Role |
|---|---|---|
| **Physics** | [PyBullet](https://github.com/bulletphysics/bullet3) | Real-time rigid-body simulation |
| **RL env** | [Gymnasium](https://gymnasium.farama.org/) | Standard `reset/step/render` API |
| **Neural Net** | [PyTorch](https://pytorch.org/) | Multi-modal Actor-Critic with LSTM |
| **RL Algorithm** | Custom PPO & Online AC | PPO on GPU & Per-step Online Actor-Critic |
| **Config** | YAML | Human-editable reward rules |
| **Viewer** | [pygame](https://www.pygame.org/) | Interactive real-time UI |

---

## Project Structure

```
EmergentCreativity/
├── config/
│   └── rewards.yaml              ← edit THIS to change behaviour
├── src/
│   └── emergent_creativity/
│       ├── environment/
│       │   ├── physics_world.py  ← PyBullet wrapper
│       │   ├── objects.py        ← all furniture & interactive items
│       │   ├── apartment.py      ← 4-room furnished apartment builder
│       │   └── senses.py         ← 5 senses implementation
│       ├── tenant/
│       │   ├── agent.py          ← tenant with vitals & inventory
│       │   └── actions.py        ← 13-action discrete space
│       ├── nn/
│       │   ├── architecture.py   ← neuroplastic multi-modal network
│       │   ├── online_learner.py ← per-step online actor-critic
│       │   └── trainer.py        ← PPO training loop (GPU-ready)
│       ├── rewards/
│       │   └── ruleset.py        ← YAML-driven reward evaluator
│       ├── ui/
│       │   └── viewer.py         ← pygame interactive viewer
│       └── sim_env.py            ← Gymnasium environment
├── tests/                        ← pytest unit tests
├── scripts/
│   ├── train.py
│   └── view.py
├── main.py                       ← entry point
├── requirements.txt
└── pyproject.toml
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

For GPU-accelerated training (recommended) install the CUDA version of PyTorch:

```bash
# Visit https://pytorch.org/get-started/locally/ for your CUDA version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2. Run the interactive viewer (online learning enabled by default)

```bash
python main.py view
```
*Note: The viewer starts in online learning mode. Press `I` to toggle to manual control.*

### 3. Train the neural network

```bash
python main.py train --steps 1000000
```

### 4. Watch a trained agent

```bash
python main.py view --nn checkpoints/checkpoint_100000.pt
```

---

## Interactive Viewer

The viewer opens a **1280 × 720** window with three panels:

| Panel | Description |
|---|---|
| **Left** | First-person camera view from the tenant's eyes |
| **Top-right** | Top-down minimap (blue dot = tenant, heading arrow shown) |
| **Right** | Live vital bars + HUD (step, reward, action, episode) |
| **Bottom-left** | Rolling reward history graph |

### Keyboard Controls

| Key | Action |
|---|---|
| `SPACE` | Pause / resume |
| `R` | Reset episode |
| `I` | Toggle Manual ↔ NN agent control |
| `↑ ↓ ← →` | Move forward / backward / strafe |
| `A` / `D` | Turn left / right |
| `F` | Pick up nearest object |
| `G` | Put down held object |
| `E` | Interact (toggle TV, open fridge …) |
| `T` | Eat held food |
| `S` | Sleep (near bed/sofa) |
| `B` | Use bathroom (near toilet) |
| `Q` / `ESC` | Quit |

---

## Training the Agent

### Online Learning (Real-Time)

When running the interactive viewer, the agent uses a **per-step online Actor-Critic** algorithm by default. It learns on every single environment step using a 1-step TD(0) advantage.

```bash
# Start learning from scratch while watching
python main.py view

# Resume learning from an online checkpoint
python main.py view --nn checkpoints/online_5000.pt
```

### Headless PPO Training (Batch)

For faster, background training, you can use the headless PPO trainer:

```bash
# Basic training
python main.py train

# With custom settings
python main.py train \
  --steps 5000000 \
  --lr 0.0002 \
  --n-steps 4096 \
  --batch 128 \
  --save-dir checkpoints \
  --log-dir logs/tensorboard

# Resume from checkpoint
python main.py train --resume checkpoints/checkpoint_500000.pt

# Monitor in TensorBoard
tensorboard --logdir logs/tensorboard
```

Training metrics logged: episode reward, episode length, policy loss, value loss, entropy, clip fraction.

---

## Configuring Rewards

Open `config/rewards.yaml` and edit the `rules` section.  No programming knowledge required.

```yaml
rules:
  # Give +10 reward whenever the tenant eats food
  - name: eat_food
    event: ate_food
    reward: 10.0

  # Subtract 0.5 per step while the tenant is hungry
  - name: hungry_penalty
    per_step: true
    condition: "hunger > 0.6"
    reward: -0.5

  # Reward cleanliness
  - name: pick_up_mess
    event: picked_up_mess
    reward: 5.0
```

### Available condition variables

| Variable | Range | Meaning |
|---|---|---|
| `hunger` | 0–1 | 0 = full, 1 = starving |
| `energy` | 0–1 | 0 = exhausted, 1 = rested |
| `bladder` | 0–1 | 0 = empty, 1 = urgent |
| `happiness` | 0–1 | subjective mood |
| `mess_count` | int | objects on the floor |
| `idle_steps` | int | consecutive idle steps |
| `is_sleeping` | bool | tenant is asleep |
| `is_watching_tv` | bool | tenant near active TV |
| `is_reading` | bool | tenant reading a book |

### Available events

`ate_food`, `picked_up_mess`, `placed_item_on_surface`, `cleaned_surface`,
`watching_tv`, `reading_book`, `playing_game`, `sleeping`, `used_bathroom`

---

## Five Senses

| Sense | Implementation | NN Input Shape |
|---|---|---|
| **Vision** | PyBullet camera render | `(84, 84, 3)` RGB |
| **Hearing** | Inverse-square sound attenuation | `(8,)` per audio channel |
| **Touch** | PyBullet contact forces | `(4,)` front/back/left/right |
| **Smell** | Linear decay proximity | `(7,)` per smell type |
| **Taste** | Active during eating, fades | `(6,)` per taste type |
| **Vitals** | Biological needs | `(4,)` hunger/energy/bladder/happiness |

---

## Tenant Vitals & Actions

### Vitals (updated every simulation step)

| Vital | Default rate | Effect |
|---|---|---|
| Hunger | +0.0002/step | Penalised at > 0.6; episode ends at 1.0 |
| Energy | −0.0001/step (awake) / +0.0005 (sleeping) | Penalised at < 0.25; ends at 0.0 |
| Bladder | +0.00015/step | Penalised at > 0.8 |
| Happiness | −0.00005/step | Penalised at < 0.3; rewarded at > 0.75 |

All rates are configurable in `config/rewards.yaml → vitals`.

### Action Space (13 discrete actions)

| # | Action | Description |
|---|---|---|
| 0 | Idle | Do nothing |
| 1 | Move Forward | Step forward 1.5 m/s |
| 2 | Move Backward | Step backward |
| 3 | Strafe Left | Step left |
| 4 | Strafe Right | Step right |
| 5 | Turn Left | Rotate 10° left |
| 6 | Turn Right | Rotate 10° right |
| 7 | Pick Up | Grab nearest reachable object |
| 8 | Put Down | Place held object |
| 9 | Interact | Toggle TV, open fridge, etc. |
| 10 | Eat | Eat held food (reduces hunger) |
| 11 | Sleep | Sleep near bed/sofa (restores energy) |
| 12 | Use Bathroom | Use toilet (relieves bladder) |

---

## Neural Network Design

The network is designed for **neuroplasticity** and RL:

```
Vision (84×84×3)
    └─► CNN Encoder (conv3 → 256d)──┐
                                    ├─► Fusion (→ 256d)
Non-visual senses + vitals          │     └─► Attention Gate (neuromodulatory)
    └─► MLP Encoder (→ 64d) ────────┘           └─► LSTM Cell (256d, temporal memory)
                                                         ├─► Actor head → action logits
                                                         └─► Critic head → value estimate
```

**Neuroplasticity features:**
- **Attention gates** — learned sigmoid masks that dynamically weight features, analogous to dopaminergic modulation in biological neural circuits.
- **LSTM memory** — the agent can remember past states across many steps, enabling strategies that unfold over time.
- **Orthogonal weight initialisation** — promotes stable gradient flow during early training.
- **Dropout** — encourages sparse, redundant representations (analogous to synaptic pruning).

---

## Requirements

```
Python >= 3.10
pybullet >= 3.2.5
gymnasium >= 0.29.1
torch >= 2.1.0          (+ CUDA drivers for GPU training)
stable-baselines3 >= 2.2.1
pygame >= 2.5.0
pyyaml >= 6.0
numpy >= 1.24.0
tensorboard >= 2.14.0
```

---

## Hardware Notes

| Mode | Minimum hardware |
|---|---|
| Viewer (manual) | Any modern CPU, no GPU needed |
| Headless training | Any modern CPU (slow) |
| GPU training (recommended) | NVIDIA GPU with CUDA, ≥ 4 GB VRAM |
| Optimal training | NVIDIA RTX 3060+ or equivalent |

The simulation renders at 84 × 84 px internally for NN input (configurable in `rewards.yaml`).
The PyGame viewer renders a scaled-up 640 × 480 display-resolution view.
Physics runs at 60 Hz (also configurable).

---

## Running Tests

```bash
pytest tests/ -v
```

Tests for the reward system, object registry, senses, and action space run without any heavy dependencies.
Neural network tests auto-skip if PyTorch is not installed.

