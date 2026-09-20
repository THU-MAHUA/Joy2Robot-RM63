"""Read old/restored teleop episodes without bridging paused segments."""
import json
from pathlib import Path
import cv2
import numpy as np

from .demonstration import STATE_KEYS, STATE_COMPONENTS


def observation(path, frame):
    rgb = cv2.imread(str(path / frame["images"]["wrist_1"]), cv2.IMREAD_COLOR)
    if rgb is None:
        raise ValueError("Missing episode image")
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    if rgb.shape != (128, 128, 3):
        raise ValueError("Unexpected wrist image dimensions")
    state = np.concatenate([np.asarray(frame["state"][key]) for key in STATE_KEYS]).astype(np.float32)
    if state.shape != (len(STATE_COMPONENTS),) or not np.isfinite(state).all():
        raise ValueError("Invalid 19-value state")
    return {"images": {"wrist_1": rgb}, "state": state}


def load_transitions(episode):
    path = Path(episode)
    with open(path / "outcome.json") as stream:
        outcome = json.load(stream)
    if outcome["status"] not in ("success", "failure") or not outcome.get("stop_confirmed"):
        return
    with open(path / "frames.jsonl") as stream:
        frames = [json.loads(line) for line in stream if line.strip()]
    for index in range(1, len(frames)):
        before, after = frames[index - 1:index + 1]
        if before["segment_index"] != after["segment_index"] or after["initial"]:
            continue
        if not after["xy_compatible"] or after["action_xy_m"] is None:
            continue
        terminal = bool(after["terminal"]) and index == len(frames) - 1
        next_gap = (index == len(frames) - 1 or
                    frames[index + 1]["segment_index"] != after["segment_index"] or
                    not frames[index + 1]["xy_compatible"])
        action = np.asarray(after["action_xy_m"], dtype=np.float32)
        if action.shape != (2,) or not np.isfinite(action).all() or not after["dt"] > 0:
            raise ValueError("Invalid interval action/timing")
        yield dict(observations=observation(path, before),
                   actions=action, next_observations=observation(path, after),
                   rewards=float(terminal and outcome["status"] == "success"),
                   masks=0. if terminal else 1., dones=terminal or next_gap,
                   terminated=terminal, truncated=bool(next_gap and not terminal),
                   outcome=outcome["status"], dt=after["dt"],
                   segment_index=after["segment_index"])
