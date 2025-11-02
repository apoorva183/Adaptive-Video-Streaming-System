# watch_live_training.py
import time
import os
import pandas as pd
import matplotlib.pyplot as plt

MONITOR_PATH = "outputs/monitor/ppo.monitor.csv"

def read_monitor(path):
    if not os.path.exists(path):
        return None
    # SB3 monitor file has comment lines starting with '#'
    df = pd.read_csv(path, comment="#", header=None, names=["r", "l", "t"])
    # keep only numeric rows
    df["r"] = pd.to_numeric(df["r"], errors="coerce")
    df["t"] = pd.to_numeric(df["t"], errors="coerce")
    df = df.dropna(subset=["r", "t"])
    if df.empty:
        return None
    # smooth a bit
    df["r_ma"] = df["r"].rolling(10, min_periods=1).mean()
    return df

def main():
    plt.ion()  # interactive mode
    fig, ax = plt.subplots(figsize=(7,4))

    while True:
        df = read_monitor(MONITOR_PATH)
        ax.clear()
        if df is not None:
            ax.plot(df["t"], df["r_ma"], label="Episode Reward (10-pt MA)")
            ax.set_xlabel("Timesteps")
            ax.set_ylabel("Reward")
            ax.set_title("PPO Training – live view")
            ax.grid(True)
            ax.legend()
        else:
            ax.text(0.5, 0.5, "Waiting for monitor file...", ha="center", va="center")
        plt.pause(1.0)   # refresh every 1s

if __name__ == "__main__":
    main()
