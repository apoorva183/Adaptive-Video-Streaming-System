# watch_live_training.py
"""
Live training monitor. Run in a separate terminal while train_drl.py is running.
Refreshes the reward plot every second. Press Ctrl+C to exit.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

MONITOR_PATH = "outputs/monitor/ppo.monitor.csv"
REFRESH_INTERVAL_S = 1.0


def read_monitor(path: str) -> "pd.DataFrame | None":
    """
    Parse a Stable-Baselines3 monitor CSV and return a smoothed reward DataFrame.

    Returns None if the file does not exist or contains no valid rows yet.
    The returned DataFrame has columns: r (raw reward), t (timestep), r_ma (10-step MA).
    """
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, comment="#", header=None, names=["r", "l", "t"])
    df["r"] = pd.to_numeric(df["r"], errors="coerce")
    df["t"] = pd.to_numeric(df["t"], errors="coerce")
    df = df.dropna(subset=["r", "t"])
    if df.empty:
        return None
    df["r_ma"] = df["r"].rolling(10, min_periods=1).mean()
    return df


def main():
    """Continuously plot the PPO reward curve until the user presses Ctrl+C."""
    plt.ion()
    fig, ax = plt.subplots(figsize=(7, 4))
    print(f"Watching {MONITOR_PATH} — press Ctrl+C to stop.")

    try:
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
            plt.pause(REFRESH_INTERVAL_S)
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        plt.close(fig)


if __name__ == "__main__":
    main()
