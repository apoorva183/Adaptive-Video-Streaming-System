# Adaptive Video Streaming System

A simulation and deep reinforcement learning (DRL) framework for Adaptive Bitrate (ABR) streaming. It trains a PPO agent to outperform classic rule-based algorithms on Quality of Experience (QoE) metrics.

## Architecture

```
traces/
  train/          # bandwidth traces for training (trace1.txt, trace2.txt, ...)
  test/           # bandwidth traces for evaluation (trace4.txt, trace5.txt, ...)

simulator.py        # Core chunk-download/buffer simulator
abr_env.py          # Gymnasium environment wrapping the simulator (for standalone use)
policies.py         # Rule-based baselines: Fixed, Rate-Based, Buffer-Based
train_drl.py        # End-to-end PPO training + evaluation pipeline
utils_metrics.py    # QoE metric computation and scoring helpers
visualize_training.py  # Plots: reward curve, training loop diagram, metric bars
watch_live_training.py # Real-time training monitor (live reward curve)

outputs/            # Generated after training
  ppo_abr.zip                      # Saved PPO model
  monitor/ppo.monitor.csv          # SB3 training log
  metrics_drl_vs_baselines.csv     # Per-trace metrics for all players
  metrics_final.csv                # Averaged metrics per player
  bar_viewer_score.png             # Final comparison chart
  ppo_reward_curve.png             # Training reward curve
  ppo_loop_diagram.png             # PPO training loop diagram
```

## Setup

```bash
pip install -r requirements.txt
```

Tested with Python 3.10+.

## Usage

### Train the PPO agent and evaluate against baselines

```bash
python train_drl.py
```

This will:
1. Train PPO for 120,000 timesteps on `traces/train/`
2. Evaluate Fixed, Rate-Based (RB), Buffer-Based (BB), and DRL policies on `traces/test/`
3. Save metrics CSVs and charts to `outputs/`

### Monitor training live (in a separate terminal)

```bash
python watch_live_training.py
```

Opens an interactive plot that refreshes every second showing the PPO reward curve.

### Run baselines standalone

```python
from simulator import VideoSimulator, load_trace_txt
from policies import rate_based_policy

bitrate_ladder = [300, 600, 900, 1200, 1800, 3000]
trace_df = load_trace_txt("traces/test/trace4.txt")
sim = VideoSimulator(trace_df, bitrate_ladder)
log = sim.run(rate_based_policy(bitrate_ladder), num_chunks=40, player_name="RB")
print(log)
```

### Use the Gymnasium environment directly

```python
from abr_env import ABREnv

env = ABREnv(trace_dir="traces/train")
obs, info = env.reset()
for _ in range(48):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated:
        break
```

## Policies

| Policy | Description |
|--------|-------------|
| **Fixed** | Always streams at a fixed bitrate (default 900 kbps) |
| **Rate-Based (RB)** | Picks the highest bitrate ≤ mean recent bandwidth × safety factor |
| **Buffer-Based (BB)** | Picks bitrate based on buffer level thresholds |
| **DRL (PPO)** | Learned policy trained to maximise long-term QoE |

## Reward Function

```
reward = log(1 + bitrate) × quality_scale
       − rebuffer_penalty × rebuffer_seconds
       − switch_penalty  [if bitrate changed]
```

## QoE & Scoring

- **QoE** — Pensieve-style: `mean(log2(bitrate)) × 10 − 4 × total_rebuffer − switches`
- **ViewerScore (0–100)** — weighted composite: 45% bitrate + 45% no-rebuffer + 10% no-switches
- **Viewer Happiness (0–100)** — per-trace normalized variant (40/40/20 weights)
