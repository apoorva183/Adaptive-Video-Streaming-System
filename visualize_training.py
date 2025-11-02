# visualize_training.py
"""
All PPO visualizations for the DRL-ABR project.
Produces:
1. PPO reward curve  -> outputs/ppo_reward_curve.png
2. PPO training loop diagram -> outputs/ppo_loop_diagram.png
3. Final metric bars (if outputs/metrics_final.csv exists)
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------- 1) PPO reward curve -----------------------
def plot_reward_curve(monitor_dir="outputs/monitor",
                      out_path="outputs/ppo_reward_curve.png"):
    files = glob.glob(os.path.join(monitor_dir, "*.monitor.csv"))
    if not files:
        print("⚠️ No monitor files found, skipping reward curve.")
        return

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, comment="#")
            dfs.append(df)
        except Exception as e:
            print("Could not read", f, e)

    if not dfs:
        print("⚠️ Monitor files unreadable.")
        return

    df_all = pd.concat(dfs, ignore_index=True)

    if "r" not in df_all.columns:
        print("⚠️ No 'r' column in monitor file.")
        return

    # smooth
    df_all["r_smooth"] = df_all["r"].rolling(10, min_periods=1).mean()

    os.makedirs("outputs", exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot(df_all.index, df_all["r_smooth"], color="purple", label="reward (MA 10)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("PPO Training Progress")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"✅ Saved PPO reward curve → {out_path}")


# ----------------------- 2) PPO loop diagram -----------------------
def plot_training_loop_diagram(out_path="outputs/ppo_loop_diagram.png"):
    os.makedirs("outputs", exist_ok=True)
    plt.figure(figsize=(9, 5))

    # left: simulator
    plt.text(
        0.1, 0.8,
        "📹 Simulator\n(Mini Video World)",
        ha="center", fontsize=11,
        bbox=dict(boxstyle="round", fc="lavender")
    )

    # middle: observation
    plt.text(
        0.5, 0.8,
        "Observation\n[buffer, est_bw, last_bitrate]",
        ha="center", fontsize=10
    )

    # right: agent
    plt.text(
        0.9, 0.8,
        "🤖 PPO Agent\n(policy network)",
        ha="center", fontsize=11,
        bbox=dict(boxstyle="round", fc="mistyrose")
    )

    # arrows top
    plt.arrow(0.15, 0.8, 0.22, 0, head_width=0.03, length_includes_head=True, color="black")
    plt.arrow(0.55, 0.8, 0.25, 0, head_width=0.03, length_includes_head=True, color="black")

    # action arrow (agent → sim)
    plt.text(0.9, 0.55, "Action:\nchoose bitrate", ha="center", color="red")
    plt.arrow(0.9, 0.75, -0.4, -0.3, head_width=0.03, length_includes_head=True, color="black")

    # reward arrow (sim → agent)
    plt.text(0.5, 0.3, "Reward = quality − 6×rebuffer − switch", ha="center", color="green")
    plt.arrow(0.1, 0.75, 0.4, -0.3, head_width=0.03, length_includes_head=True, color="black")

    # bottom text
    plt.text(0.5, 0.05, "🧠 PPO updates weights to improve long-term QoE", ha="center", color="blue")

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"✅ Saved PPO loop diagram → {out_path}")


# ----------------------- 3) Final metrics -----------------------
def plot_final_metrics(metrics_path="outputs/metrics_final.csv"):
    if not os.path.exists(metrics_path):
        print("⚠️ metrics_final.csv not found, skipping metric plots.")
        return

    df = pd.read_csv(metrics_path)

    # viewer score
    plt.figure(figsize=(6, 4))
    plt.bar(df["player"], df["ViewerScore"], color="skyblue")
    plt.title("Viewer Score (0–100)")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig("outputs/bar_viewer_score_from_metrics.png")
    plt.close()

    # avg bitrate
    plt.figure(figsize=(6, 4))
    plt.bar(df["player"], df["avg_bitrate_kbps"])
    plt.title("Average bitrate")
    plt.ylabel("kbps")
    plt.tight_layout()
    plt.savefig("outputs/bar_avg_bitrate_from_metrics.png")
    plt.close()

    # rebuffer
    plt.figure(figsize=(6, 4))
    plt.bar(df["player"], df["total_rebuffer_s"], color="salmon")
    plt.title("Total rebuffer")
    plt.ylabel("seconds")
    plt.tight_layout()
    plt.savefig("outputs/bar_rebuffer_from_metrics.png")
    plt.close()

    print("✅ Saved metric plots to outputs/")
