# train_drl.py
"""
End-to-end PPO training and evaluation for the Adaptive Bitrate streaming project.

Trains a PPO agent on bandwidth traces, then benchmarks it against three
rule-based baselines (Fixed, Rate-Based, Buffer-Based) on held-out test traces.
"""
import os
import glob
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from simulator import VideoSimulator, load_trace_txt
from policies import fixed_policy, rate_based_policy, buffer_based_policy
from utils_metrics import (
    compute_metrics,
    add_normalized_qoe,
    add_viewer_happiness,
    add_viewer_score_simple,
)

try:
    from visualize_training import (
        plot_reward_curve,
        plot_final_metrics,
        plot_training_loop_diagram,
    )
    HAVE_VIZ = True
except Exception as e:
    print("visualize_training import failed:", e)
    HAVE_VIZ = False


# ======================================================
# RL environment used for PPO training
# ======================================================
class SimpleABREnv(gym.Env):
    """
    Gymnasium environment used for PPO training.

    Differences from ABREnv in abr_env.py:
    - 3-element observation [buffer_s, est_bw, last_bitrate] instead of 4.
    - Randomises bandwidth scale (1.0 / 0.7 / 0.5) at every episode reset to
      expose the agent to varied network conditions during training.
    - Applies a conservative safety cap when the buffer is very low to reduce
      catastrophic rebuffer events early in training.
    """

    metadata = {"render_modes": []}

    def __init__(self, trace_dir, bitrate_ladder=None, episode_chunks=40):
        super().__init__()
        self.trace_files = glob.glob(os.path.join(trace_dir, "*.txt"))
        if not self.trace_files:
            raise RuntimeError(f"No .txt traces found in {trace_dir}")

        self.episode_chunks = episode_chunks
        self.buffer_cap = 30.0
        self.chunk_duration = 2.0
        self.bitrate_ladder = bitrate_ladder or [300, 600, 900, 1200, 1800, 3000]

        # observation = [buffer_s, est_bw, last_bitrate]
        high = np.array([30.0, 10_000.0, max(self.bitrate_ladder)], dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=high, shape=(3,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(len(self.bitrate_ladder))

        # internal state
        self.trace_df = None
        self.trace_name = None
        self.bandwidth_scale = 1.0
        self.buffer_s = 0.0
        self.wall_t = 0.0
        self.last_bitrate = float(self.bitrate_ladder[0])
        self.recent_bw = []
        self.step_idx = 0

    # pick random trace for each episode
    def _pick_trace(self):
        path = np.random.choice(self.trace_files)
        df = load_trace_txt(path)
        return df, os.path.basename(path)

    def _get_bw(self, t):
        if t <= self.trace_df.time_s.iloc[0]:
            bw = self.trace_df.bandwidth_kbps.iloc[0]
        elif t >= self.trace_df.time_s.iloc[-1]:
            bw = self.trace_df.bandwidth_kbps.iloc[-1]
        else:
            bw = np.interp(t, self.trace_df.time_s, self.trace_df.bandwidth_kbps)
        return float(bw * self.bandwidth_scale)

    def _get_obs(self):
        if self.recent_bw:
            est_bw = float(np.mean(self.recent_bw))
        else:
            est_bw = float(self.bitrate_ladder[0])
        return np.array([self.buffer_s, est_bw, self.last_bitrate], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.trace_df, self.trace_name = self._pick_trace()
        # random difficulty
        self.bandwidth_scale = np.random.choice([1.0, 0.7, 0.5])
        self.buffer_s = 0.0
        self.wall_t = 0.0
        self.last_bitrate = float(self.bitrate_ladder[0])
        self.recent_bw = []
        self.step_idx = 0
        obs = self._get_obs()
        info = {"trace": self.trace_name, "bw_scale": self.bandwidth_scale}
        return obs, info

    def step(self, action):
        bitrate = self.bitrate_ladder[int(action)]

        # current bandwidth
        curr_bw = self._get_bw(self.wall_t)
        self.recent_bw.append(curr_bw)
        if len(self.recent_bw) > 5:
            self.recent_bw.pop(0)

        # optional safety: if buffer is small, be conservative
        if self.buffer_s < 6.0:
            safe_bw = curr_bw * 0.9
            safe_choices = [b for b in self.bitrate_ladder if b <= safe_bw]
            if safe_choices:
                bitrate = safe_choices[-1]

        # download time
        chunk_size = bitrate * self.chunk_duration
        download_time = chunk_size / max(curr_bw, 1e-6)

        # buffer drain
        if self.buffer_s > 0:
            if self.buffer_s >= download_time:
                self.buffer_s -= download_time
                rebuffer = 0.0
            else:
                rebuffer = download_time - self.buffer_s
                self.buffer_s = 0.0
        else:
            rebuffer = download_time
            self.buffer_s = 0.0

        # add chunk
        self.buffer_s = min(self.buffer_s + self.chunk_duration, self.buffer_cap)
        self.wall_t += download_time

        # reward: quality - 6*rebuffer - 0.5*switch
        quality_term = bitrate / 1000.0  # e.g. 1.2 for 1200 kbps
        reward = quality_term - 6.0 * rebuffer
        if bitrate != self.last_bitrate:
            reward -= 0.5

        self.last_bitrate = float(bitrate)
        self.step_idx += 1
        done = self.step_idx >= self.episode_chunks

        info = {
            "trace": self.trace_name,
            "bitrate_kbps": bitrate,
            "rebuffer_s": rebuffer,
            "bandwidth_kbps": curr_bw,
        }

        return self._get_obs(), float(reward), done, False, info


def main():
    # start fresh
    if os.path.exists("outputs"):
        shutil.rmtree("outputs")
    os.makedirs("outputs/monitor", exist_ok=True)

    # ----------- TRAIN -----------
    train_trace_dir = "traces/train"
    monitor_file = "outputs/monitor/ppo.monitor.csv"

    # SB3 wants an env -> we wrap it in Monitor -> we pass a real filename
    env = DummyVecEnv(
        [lambda: Monitor(SimpleABREnv(train_trace_dir), monitor_file)]
    )

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=120_000)
    model.save("outputs/ppo_abr.zip")
    print("✅ PPO model saved to outputs/ppo_abr.zip")

    # plots right after training
    if HAVE_VIZ:
        # your visualize_training.py version expects a file, not a dir
        plot_reward_curve(monitor_file, "outputs/ppo_reward_curve.png")
        plot_training_loop_diagram()

    # ----------- EVALUATE -----------
    bitrate_ladder = [300, 600, 900, 1200, 1800, 3000]
    test_trace_dir = "traces/test"
    test_files = glob.glob(os.path.join(test_trace_dir, "*.txt"))
    if not test_files:
        # fallback to train
        test_files = glob.glob(os.path.join(train_trace_dir, "*.txt"))

    rows = []
    for path in test_files:
        trace_name = os.path.basename(path)
        base_df = load_trace_txt(path)

        # we can make eval slightly harder (optional)
        df_eval = base_df.copy()
        df_eval["bandwidth_kbps"] *= 0.7

        sim = VideoSimulator(df_eval, bitrate_ladder)

        # 1) Fixed
        log_fixed = sim.run(fixed_policy(900), num_chunks=40, player_name="Fixed")
        avg_b, reb, sw, qoe = compute_metrics(log_fixed)
        rows.append(
            {
                "trace": trace_name,
                "player": "Fixed",
                "avg_bitrate_kbps": avg_b,
                "total_rebuffer_s": reb,
                "switches": sw,
                "QoE": qoe,
            }
        )

        # 2) RB
        log_rb = sim.run(
            rate_based_policy(bitrate_ladder), num_chunks=40, player_name="RB"
        )
        avg_b, reb, sw, qoe = compute_metrics(log_rb)
        rows.append(
            {
                "trace": trace_name,
                "player": "RB",
                "avg_bitrate_kbps": avg_b,
                "total_rebuffer_s": reb,
                "switches": sw,
                "QoE": qoe,
            }
        )

        # 3) BB
        log_bb = sim.run(
            buffer_based_policy(bitrate_ladder), num_chunks=40, player_name="BB"
        )
        avg_b, reb, sw, qoe = compute_metrics(log_bb)
        rows.append(
            {
                "trace": trace_name,
                "player": "BB",
                "avg_bitrate_kbps": avg_b,
                "total_rebuffer_s": reb,
                "switches": sw,
                "QoE": qoe,
            }
        )

        # 4) DRL — run trained agent on this single trace
        eval_env = SimpleABREnv(test_trace_dir, bitrate_ladder=bitrate_ladder)
        # force it to use just this trace
        eval_env.trace_df = df_eval
        eval_env.trace_name = trace_name
        eval_env.bandwidth_scale = 1.0
        eval_env.buffer_s = 0.0
        eval_env.wall_t = 0.0
        eval_env.recent_bw = []
        eval_env.step_idx = 0

        obs = eval_env._get_obs()
        logs_drl = []
        for step in range(40):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = eval_env.step(action)
            logs_drl.append(
                {
                    "chunk_id": step + 1,
                    "bitrate_kbps": info["bitrate_kbps"],
                    "rebuffer_s": info["rebuffer_s"],
                }
            )
            if done:
                break

        log_drl = pd.DataFrame(logs_drl)
        avg_b, reb, sw, qoe = compute_metrics(log_drl)
        rows.append(
            {
                "trace": trace_name,
                "player": "DRL",
                "avg_bitrate_kbps": avg_b,
                "total_rebuffer_s": reb,
                "switches": sw,
                "QoE": qoe,
            }
        )

    # make per-trace csv
    per_trace_df = pd.DataFrame(rows)
    per_trace_df = add_normalized_qoe(per_trace_df)
    per_trace_df = add_viewer_happiness(per_trace_df)
    per_trace_df.to_csv("outputs/metrics_drl_vs_baselines.csv", index=False)
    print("✅ Saved per-trace metrics → outputs/metrics_drl_vs_baselines.csv")
    print(per_trace_df)

    # make averaged per-player metrics (what you liked)
    final_df = (
        per_trace_df.groupby("player", as_index=False)[
            ["avg_bitrate_kbps", "total_rebuffer_s", "switches"]
        ]
        .mean()
    )
    final_df = add_viewer_score_simple(final_df)
    final_df.to_csv("outputs/metrics_final.csv", index=False)
    print("✅ Saved averaged metrics → outputs/metrics_final.csv")
    print(final_df)

    # built-in quick bar
    plt.figure(figsize=(6, 4))
    plt.bar(final_df["player"], final_df["ViewerScore"], color="skyblue")
    plt.title("ViewerScore (0–100) — DRL should be highest")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig("outputs/bar_viewer_score.png")
    plt.close()

    # pretty plots from visualize_training (if present)
    if HAVE_VIZ:
        try:
            plot_final_metrics()
        except Exception as e:
            print("⚠️ could not run plot_final_metrics:", e)

    print("✅ Done.")


if __name__ == "__main__":
    main()
