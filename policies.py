# policies.py
import numpy as np


def fixed_policy(fixed_bitrate=900):
    """Always pick the same bitrate."""
    def fn(buffer_s, recent_bw, ladder):
        return fixed_bitrate
    return fn


def rate_based_policy(bitrate_ladder, safety_factor=0.85):
    """
    Look at recent bandwidth, pick the highest bitrate below it.
    """
    def fn(buffer_s, recent_bw, ladder=None):
        ladder = bitrate_ladder if ladder is None else ladder
        if recent_bw:
            est_bw = np.mean(recent_bw) * safety_factor
        else:
            est_bw = ladder[0]
        choices = [b for b in ladder if b <= est_bw]
        return choices[-1] if choices else ladder[0]
    return fn


def buffer_based_policy(bitrate_ladder):
    """
    Look at buffer fullness.
    low buffer -> lowest bitrate
    mid buffer -> middle bitrate
    high buffer -> highest bitrate
    """
    low, high = 5.0, 15.0

    def fn(buffer_s, recent_bw, ladder=None):
        ladder = bitrate_ladder if ladder is None else ladder
        if buffer_s < low:
            return ladder[0]
        elif buffer_s < high:
            return ladder[len(ladder)//2]
        else:
            return ladder[-1]
    return fn
