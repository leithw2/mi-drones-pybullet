"""Train PPO to hold [0, 0, 1] while facing sequential random targets."""

import argparse
import os
import time
from datetime import datetime

import numpy as np
import pybullet as p
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import ActionType, ObservationType, Physics
from gym_pybullet_drones.utils.utils import sync


class RandomTargetYawHoverAviary(HoverAviary):
    """Hover at a fixed position and turn toward one random target at a time."""

    def __init__(self, *args, **kwargs):
        kwargs.update(
            initial_xyzs=np.array([[0.0, 0.0, 1.0]]),
            initial_rpys=np.array([[0.0, 0.0, 0.0]]),
            random_targets=False,
        )
        super().__init__(*args, **kwargs)
        self.one_only_target = True
        self.HOVER_POS = np.array([0.0, 0.0, 1.0])
        self.TARGET_POS = self.HOVER_POS.copy()
        self._previous_heading_error = 0.0
        self.targets_reached = 0
        self.target_radius = 1.5
        self.heading_threshold = np.deg2rad(12.0)
        self.hover_phase_steps = 3000000
        self.curriculum_steps = 0

    def _new_random_target(self):
        angle = np.random.uniform(-np.pi, np.pi)
        radius = np.random.uniform(0.75, self.target_radius)
        self.TARGET_POS = self.HOVER_POS + np.array([
            radius * np.cos(angle),
            radius * np.sin(angle),
            0.0,
        ])

    @property
    def orientation_phase(self):
        return self.curriculum_steps >= self.hover_phase_steps

    def _heading_error(self, state):
        target_xy = self.TARGET_POS[:2] - state[:2]
        target_norm = np.linalg.norm(target_xy)
        if target_norm < 1e-8:
            return 0.0

        rotation_matrix = np.asarray(
            p.getMatrixFromQuaternion(state[3:7])
        ).reshape(3, 3)
        forward_xy = rotation_matrix[:2, 0]
        forward_norm = np.linalg.norm(forward_xy)
        if forward_norm < 1e-8:
            return np.pi

        forward_xy /= forward_norm
        target_xy /= target_norm
        cross_z = forward_xy[0] * target_xy[1] - forward_xy[1] * target_xy[0]
        dot = np.clip(np.dot(forward_xy, target_xy), -1.0, 1.0)
        return float(np.arctan2(cross_z, dot))

    def _computeObs(self):
        """Expose the random target direction in the drone body frame."""
        observation = super()._computeObs()
        if observation is None:
            return observation
        state = self._getDroneStateVector(0)
        target_direction = self.TARGET_POS - state[:3]
        target_norm = np.linalg.norm(target_direction)
        if target_norm > 1e-8:
            rotation_matrix = np.asarray(
                p.getMatrixFromQuaternion(state[3:7])
            ).reshape(3, 3)
            target_direction_body = rotation_matrix.T @ (target_direction / target_norm)
            observation[0, :3] = target_direction_body.astype(np.float32)
        return observation

    def reset(self, *args, **kwargs):
        observation, info = super().reset(*args, **kwargs)
        self.HOVER_POS = np.array([0.0, 0.0, 1.0])
        if self.orientation_phase:
            self._new_random_target()
        else:
            self.TARGET_POS = self.HOVER_POS.copy()
        state = self._getDroneStateVector(0)
        self._previous_heading_error = self._heading_error(state)
        self.targets_reached = 0
        return self._computeObs(), info

    def _computeReward(self):
        state = self._getDroneStateVector(0)
        position_error = np.linalg.norm(state[0:3] - self.HOVER_POS)
        linear_speed = np.linalg.norm(state[10:13])
        roll_pitch_error = np.linalg.norm(state[7:9])

        if not self.orientation_phase:
            self.curriculum_steps += 1
            hover_reward = 4.0 * np.exp(-25.0 * position_error**2)
            position_penalty = 8.0 * position_error**2
            stability_penalty = 0.8 * roll_pitch_error + 0.25 * linear_speed
            return float(hover_reward - position_penalty - stability_penalty)

        heading_error = self._heading_error(state)
        heading_progress = abs(self._previous_heading_error) - abs(heading_error)
        self._previous_heading_error = heading_error

        position_reward = 3.0 * np.exp(-18.0 * position_error**2)
        position_penalty = 4.0 * position_error**2
        heading_reward = 1.5 * (1.0 - abs(heading_error) / np.pi)
        progress_reward = 5.0 * np.clip(heading_progress, -0.2, 0.2)
        stability_penalty = 0.8 * roll_pitch_error + 0.2 * linear_speed
        reward = position_reward + heading_reward + progress_reward - stability_penalty
        reward -= position_penalty

        if abs(heading_error) <= self.heading_threshold:
            self.targets_reached += 1            
            reward += 8.0
            self._new_random_target()
            self._previous_heading_error = self._heading_error(state)

        self.curriculum_steps += 1
        return float(reward)

    def _computeTerminated(self):
        """Do not terminate when the fixed hover point is reached."""
        state = self._getDroneStateVector(0)
        penalty = 200.0 / self.score

        position_error = np.linalg.norm(state[:3] - self.HOVER_POS)
        if position_error > 0.8 or state[2] > 2.0:
            return True, penalty
        if abs(state[7]) > 1.1 or abs(state[8]) > 1.1:
            return True, penalty
        if state[2] < 0.02 or self.obstacle_collision:
            return True, penalty
        return False, 0


class SpinProgressCallback(BaseCallback):
    """Report random targets reached by the first training environment."""

    def _on_step(self):
        if self.num_timesteps % 10000 == 0:
            environment = self.training_env.envs[0].unwrapped
            print(
                f"timesteps={self.num_timesteps} "
                f"phase={'orientation' if environment.orientation_phase else 'hover'} "
                f"targets_reached={environment.targets_reached} "
                f"heading_error_deg={np.degrees(environment._previous_heading_error):.1f}"
            )
        return True


def make_spin_environment(hover_phase_steps=3000000):
    environment = RandomTargetYawHoverAviary(
        obs=ObservationType.KIN,
        act=ActionType.RPM,
        physics=Physics.PYB,
        ctrl_freq=60,
    )
    environment.hover_phase_steps = hover_phase_steps
    return environment


def evaluate_gui(model_path, episodes=5, max_steps=3600, episode_pause=2.0):
    """Show a trained model in PyBullet at wall-clock speed."""
    environment = RandomTargetYawHoverAviary(
        gui=True,
        obs=ObservationType.KIN,
        act=ActionType.RPM,
        physics=Physics.PYB,
        ctrl_freq=60,
    )
    environment.hover_phase_steps = 0
    model = PPO.load(model_path, env=environment, device="cpu")

    try:
        for episode in range(episodes):
            observation, _ = environment.reset(seed=episode)
            start_time = time.time()
            for step in range(max_steps):
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, _ = environment.step(action)

                if step % environment.CTRL_FREQ == 0:
                    state = environment._getDroneStateVector(0)
                    print(
                        f"episodio={episode + 1} "
                        f"tiempo={step / environment.CTRL_FREQ:5.1f}s "
                        f"objetivos={environment.targets_reached} "
                        f"error_z={np.degrees(environment._heading_error(state)):6.1f} deg "
                        f"pos=({state[0]:.2f}, {state[1]:.2f}, {state[2]:.2f})"
                    )

                sync(step, start_time, environment.CTRL_TIMESTEP)
                if terminated or truncated:
                    break

            print(
                f"Episodio {episode + 1}: "
                f"{environment.targets_reached} objetivos alcanzados"
            )
            if episode < episodes - 1:
                time.sleep(episode_pause)
    finally:
        environment.close()


def run(total_timesteps=3000000, n_envs=12, output_folder="results", gui=False,
    model_path=None, episodes=5, max_steps=3600, episode_pause=2.0,
    hover_phase_steps=3000000, continue_from=None):
    if model_path:
        evaluate_gui(
            model_path,
            episodes=episodes,
            max_steps=max_steps,
            episode_pause=episode_pause,
        )
        return

    model_file = None
    if continue_from:
        filename = continue_from
        model_file = os.path.join(filename, "final_model.zip")
        if not os.path.isfile(model_file):
            raise FileNotFoundError(
                f"No se encontró el modelo para continuar: {model_file}"
            )
    else:
        filename = os.path.join(
            output_folder,
            "hovering_yaw_spin_" + datetime.now().strftime("%m.%d.%Y_%H.%M.%S"),
        )
    os.makedirs(filename, exist_ok=True)

    train_env = make_vec_env(
        make_spin_environment,
        env_kwargs=dict(hover_phase_steps=hover_phase_steps),
        n_envs=n_envs,
        seed=0,
    )
    if continue_from:
        print(f"Continuando entrenamiento desde: {model_file}")
        model = PPO.load(
            str(model_file),
            env=train_env,
            device="cpu",
        )
        model.tensorboard_log = os.path.join(filename, "tb")
    else:
        model = PPO(
            "MlpPolicy",
            train_env,
            device="cpu",
            tensorboard_log=os.path.join(filename, "tb"),
            n_steps=int(512 * 4),
            batch_size=int(256 * 4),
            n_epochs=10,
            gae_lambda=0.95,
            learning_rate=lambda progress: (
                0.00005 + (0.0007 - 0.00005) * ((progress - 0.25) / 0.75)
                if progress > 0.25 else 0.00005
            ),
            policy_kwargs=dict(
                net_arch=[dict(pi=[256, 256, 182], vf=[256, 256, 128])],
                activation_fn=torch.nn.Tanh,
                log_std_init=-2.0,
                ortho_init=True,
            ),
            ent_coef=0.01,
            clip_range=0.2,
            verbose=1,
        )
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=SpinProgressCallback(),
            reset_num_timesteps=not bool(continue_from),
        )
        model.save(os.path.join(filename, "final_model"))
        mean_reward, std_reward = evaluate_policy(
            model,
            train_env,
            n_eval_episodes=5,
            deterministic=True,
        )
        print(f"Evaluacion: reward={mean_reward:.3f} +/- {std_reward:.3f}")
        print(f"Modelo guardado en: {filename}")

        if gui:
            evaluate_gui(
                os.path.join(filename, "final_model.zip"),
                episodes=episodes,
                max_steps=max_steps,
                episode_pause=episode_pause,
            )
    finally:
        train_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_timesteps", type=int, default=10)
    parser.add_argument("--n_envs", type=int, default=12)
    parser.add_argument("--output_folder", type=str, default="results")
    parser.add_argument("--gui", action="store_true", help="Mostrar el modelo al terminar el entrenamiento")
    parser.add_argument("--model_path", type=str, default=None, help="Evaluar este modelo directamente en GUI")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=3600)
    parser.add_argument("--episode_pause", type=float, default=2.0)
    parser.add_argument("--hover_phase_steps", type=int, default=300000)
    parser.add_argument("--continue_from", type=str, default=None,
                        help="Carpeta que contiene final_model.zip para continuar")
    args = parser.parse_args()
    run(**vars(args))