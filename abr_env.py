# abr_env.py
import os
import glob
import gymnasium as gym
import numpy as np
from simulator import VideoSimulator, load_trace_txt


class ABREnv(gym.Env):
    """
    Gym-style ABR environment.

    You can use:
    - training: ABREnv(trace_dir="traces/train", ...)
    - testing:  ABREnv(fixed_trace_path="traces/test/trace4.txt", bandwidth_scale=0.6, ...)

    bandwidth_scale < 1.0  → make network harder
    bandwidth_scale > 1.0  → make network easier
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        trace_dir=None,
        fixed_trace_path=None,
        bitrate_ladder=None,
        chunk_duration=2.0,
        buffer_cap=30.0,
        episode_chunks=48,
        rebuffer_penalty=8.0,
        switch_penalty=1.0,
        quality_scale=12.0,
        bandwidth_scale=1.0,   # 👈 NEW
    ):
        super().__init__()

        if bitrate_ladder is None:
            bitrate_ladder = [300, 600, 900, 1200, 1800, 3000]
        self.bitrate_ladder = bitrate_ladder
        self.chunk_duration = float(chunk_duration)
        self.buffer_cap = float(buffer_cap)
        self.episode_chunks = int(episode_chunks)

        self.rebuffer_penalty = float(rebuffer_penalty)
        self.switch_penalty = float(switch_penalty)
        self.quality_scale = float(quality_scale)
        self.bandwidth_scale = float(bandwidth_scale)

        self.fixed_trace_path = fixed_trace_path
        if fixed_trace_path is None:
            if trace_dir is None:
                raise ValueError("Either trace_dir or fixed_trace_path must be provided.")
            self.trace_files = glob.glob(os.path.join(trace_dir, "*.txt"))
            if not self.trace_files:
                raise RuntimeError(f"No traces found in {trace_dir}")
        else:
            self.trace_files = [fixed_trace_path]

        high = np.array([
            buffer_cap,
            10_000.0,                # est bw
            max(bitrate_ladder),     # last bitrate
            10.0                     # last download
        ], dtype=np.float32)

        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=high,
            shape=(4,),
            dtype=np.float32,
        )

        self.action_space = gym.spaces.Discrete(len(bitrate_ladder))

        self._reset_sim()

    # ---------------- internal helpers ----------------
    def _reset_sim(self):
        trace_path = np.random.choice(self.trace_files)
        trace_df = load_trace_txt(trace_path)
        # we do not scale here; we scale when we read bandwidth
        self.trace_df = trace_df
        self.curr_trace_name = os.path.basename(trace_path)

        self.buffer_s = 0.0
        self.last_bitrate = float(self.bitrate_ladder[0])
        self.last_download = 0.0
        self.recent_bw = []
        self.step_idx = 0
        self.wall_t = 0.0

    def _get_bandwidth(self, t):
        # simple interpolation exactly like simulator
        df = self.trace_df
        if t <= df.time_s.iloc[0]:
            bw = df.bandwidth_kbps.iloc[0]
        elif t >= df.time_s.iloc[-1]:
            bw = df.bandwidth_kbps.iloc[-1]
        else:
            bw = np.interp(t, df.time_s, df.bandwidth_kbps)
        return bw * self.bandwidth_scale   # 👈 apply scale here

    def _get_obs(self):
        if self.recent_bw:
            est_bw = float(np.mean(self.recent_bw))
        else:
            est_bw = float(self.bitrate_ladder[0])
        return np.array(
            [self.buffer_s, est_bw, self.last_bitrate, self.last_download],
            dtype=np.float32
        )

    # ---------------- gym API ----------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_sim()
        obs = self._get_obs()
        return obs, {"trace": self.curr_trace_name}

    def step(self, action):
        bitrate = self.bitrate_ladder[int(action)]

        curr_bw = self._get_bandwidth(self.wall_t)
        self.recent_bw.append(curr_bw)
        if len(self.recent_bw) > 5:
            self.recent_bw.pop(0)

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

        # reward
        quality_term = np.log1p(bitrate) * self.quality_scale
        r = quality_term - self.rebuffer_penalty * rebuffer
        if bitrate != self.last_bitrate:
            r -= self.switch_penalty
        r = float(np.clip(r, -100.0, 100.0))

        self.last_bitrate = float(bitrate)
        self.last_download = float(download_time)
        self.step_idx += 1

        terminated = self.step_idx >= self.episode_chunks

        info = {
            "trace": self.curr_trace_name,
            "bitrate_kbps": bitrate,
            "download_time_s": download_time,
            "rebuffer_s": rebuffer,
            "buffer_after_s": self.buffer_s,
            "bandwidth_kbps": curr_bw,
            "bw_scale": self.bandwidth_scale,
        }

        return self._get_obs(), r, terminated, False, info
