import gymnasium as gym
import numpy as np
from stable_baselines3.common.monitor import Monitor

from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
from gym_pybullet_drones.utils.enums import ActionType, DroneModel, ObservationType, Physics


EVAL_SEEDS = tuple(range(10))
EVAL_DEVICE = "cpu"
POWERLOOP_TRACK_SCALE = 1.0
POWERLOOP_POSITION_VARIATION = 0.02
POWERLOOP_ORIENTATION_VARIATION = 0.02


class SeededEpisodeWrapper(gym.Wrapper):
    def __init__(self, env, seed=0):
        super().__init__(env)
        self.next_seed = seed

    def reset(self, **kwargs):
        current_seed = kwargs.get("seed")
        if current_seed is None:
            current_seed = self.next_seed
        kwargs["seed"] = current_seed
        result = self.env.reset(**kwargs)
        self.next_seed = current_seed + 1
        return result


class FixedSeedEvalWrapper(gym.Wrapper):
    def __init__(self, env, seeds=EVAL_SEEDS):
        super().__init__(env)
        self.seeds = seeds
        self.idx = 0

    def reset(self, **kwargs):
        current_seed = self.seeds[self.idx % len(self.seeds)]
        kwargs["seed"] = current_seed
        print("current_seed ", current_seed)
        self.idx += 1
        return self.env.reset(**kwargs)


def make_eval_env(
    multiagent=False,
    gui=False,
    record=False,
    drone_model=DroneModel.CF2X250,
    obs=ObservationType.KIN,
    act=ActionType.BRUSHLESS_THRUST,
    physics=Physics.PYB_WIND,
    random_targets=False,
    num_drones=1,
    track_scale=POWERLOOP_TRACK_SCALE,
    track_position_variation=POWERLOOP_POSITION_VARIATION,
    track_orientation_variation=POWERLOOP_ORIENTATION_VARIATION,
):
    if multiagent:
        env = MultiHoverAviary(
            drone_model=drone_model,
            num_drones=num_drones,
            obs=obs,
            act=act,
            physics=physics,
            gui=gui,
            record=record,
        )
    else:
        env = HoverAviary(
            drone_model=drone_model,
            initial_xyzs=np.array([[0.0, 0.0, 1.0]]),
            initial_rpys=np.zeros((1, 3)),
            obs=obs,
            act=act,
            physics=physics,
            gui=gui,
            record=record,
            random_targets=random_targets,
            pyb_freq=240,
            ctrl_freq=60,
            randomized=True,
            track_scale=track_scale,
            track_position_variation=track_position_variation,
            track_orientation_variation=track_orientation_variation,
        )

    return FixedSeedEvalWrapper(Monitor(env))
