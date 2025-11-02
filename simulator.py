# simulator.py
import numpy as np
import pandas as pd


class VideoSimulator:
    """
    Core simulator: given a bandwidth trace and a bitrate choice,
    it simulates downloading a video chunk, buffer drain/fill, and rebuffer.
    """
    def __init__(self, trace_df, bitrate_ladder, chunk_duration=2.0, buffer_cap=30.0):
        """
        trace_df must have columns: ['time_s', 'bandwidth_kbps']
        """
        self.trace = trace_df.reset_index(drop=True)
        self.bitrate_ladder = bitrate_ladder
        self.chunk_duration = float(chunk_duration)
        self.buffer_cap = float(buffer_cap)

    def get_bandwidth(self, t):
        """ Linear interpolate bandwidth at time t (seconds). """
        if t <= self.trace.time_s.iloc[0]:
            return float(self.trace.bandwidth_kbps.iloc[0])
        if t >= self.trace.time_s.iloc[-1]:
            return float(self.trace.bandwidth_kbps.iloc[-1])
        return float(np.interp(t, self.trace.time_s, self.trace.bandwidth_kbps))

    def run(self, policy_fn, num_chunks=30, player_name="unknown"):
        """
        policy_fn(buffer_s, recent_bw_list) -> bitrate_kbps
        returns per-chunk log (DataFrame)
        """
        logs = []
        buffer_s = 0.0
        wall_t = 0.0
        total_rebuffer = 0.0
        recent_bw = []

        for chunk_id in range(1, num_chunks + 1):
            # 1) read current bw from trace
            curr_bw = self.get_bandwidth(wall_t)
            recent_bw.append(curr_bw)
            if len(recent_bw) > 5:
                recent_bw.pop(0)

            # 2) choose bitrate
            bitrate = float(policy_fn(buffer_s, recent_bw))

            # 3) download time (kbits / kbps = s)
            chunk_size = bitrate * self.chunk_duration  # kbits
            download_time = chunk_size / max(curr_bw, 1e-6)

            # 4) drain buffer while downloading
            if buffer_s > 0:
                if buffer_s >= download_time:
                    buffer_s -= download_time
                    rebuffer = 0.0
                else:
                    rebuffer = download_time - buffer_s
                    buffer_s = 0.0
            else:
                rebuffer = download_time
                buffer_s = 0.0

            total_rebuffer += rebuffer

            # 5) after download, add the chunk to buffer
            buffer_s = min(buffer_s + self.chunk_duration, self.buffer_cap)

            # 6) move time forward
            wall_t += download_time

            logs.append({
                "chunk_id": chunk_id,
                "player": player_name,
                "bitrate_kbps": bitrate,
                "bandwidth_kbps": curr_bw,
                "download_time_s": download_time,
                "rebuffer_s": rebuffer,
                "buffer_after_s": buffer_s,
                "wall_time_s": wall_t,
            })

        return pd.DataFrame(logs)


# handy loader for your txt traces
def load_trace_txt(path):
    """
    Your txt format looks like:
    col0: time (or something increasing)
    col5: bandwidth
    We'll map it to (time_s, bandwidth_kbps)
    """
    df = pd.read_csv(path, sep=r"\s+", header=None)
    # col 0 => time, col 5 => bandwidth
    time_s = df.iloc[:, 0].astype(float)
    bw_kbps = df.iloc[:, 5].astype(float)
    return pd.DataFrame({"time_s": time_s, "bandwidth_kbps": bw_kbps})
