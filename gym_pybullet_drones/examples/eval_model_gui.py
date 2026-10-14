"""Evaluate the configured PPO model using PyBullet's native GUI controls."""

import os
import time

import numpy as np
import pybullet as p
import torch
from stable_baselines3 import PPO

from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import ActionType, ObservationType, Physics
from gym_pybullet_drones.utils.utils import sync


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "results",
    "Race09.28.2026_15.41.08",
    "best_model.zip",
)
NUM_EPISODES = 3
MAX_STEPS = None
RANDOM_TARGETS = True
RECORD_VIDEO = False


def create_controls(client_id):
    """Create PyBullet debug sliders used during evaluation."""
    return {
        "episodes": p.addUserDebugParameter(
            "Episodes", 1, 20, NUM_EPISODES, physicsClientId=client_id
        ),
        "speed": p.addUserDebugParameter(
            "Speed (x)", 0.1, 3.0, 1.0, physicsClientId=client_id
        ),
        "pause": p.addUserDebugParameter(
            "Pause toggle (move between 0 and 1)",
            0,
            1,
            0,
            physicsClientId=client_id,
        ),
        "stop": p.addUserDebugParameter(
            "Stop (0=run, 1=stop)", 0, 1, 0, physicsClientId=client_id
        ),
        "camera": p.addUserDebugParameter(
            "Camera (0=FPV, 1=chase, 2=orbit)",
            0,
            2,
            0,
            physicsClientId=client_id,
        ),
        "lidar": p.addUserDebugParameter(
            "Show LiDAR rays (0/1)", 0, 1, 0, physicsClientId=client_id
        ),
        "shadows": p.addUserDebugParameter(
            "Shadows (0/1)", 0, 1, 1, physicsClientId=client_id
        ),
        "reflections": p.addUserDebugParameter(
            "Planar reflections (0/1)", 0, 1, 0, physicsClientId=client_id
        ),
    }


def keyboard_pressed(client_id, key):
    events = p.getKeyboardEvents(physicsClientId=client_id)
    return bool(events.get(key, 0) & p.KEY_WAS_TRIGGERED)


def update_hud(env, episode, num_episodes, total_reward, paused, hud_id):
    position = np.asarray(env.pos[0])
    speed = float(np.linalg.norm(env.vel[0]))
    yaw = float(np.degrees(env.rpy[0, 2]) % 360)
    sim_time = env.step_counter / env.PYB_FREQ
    gate_text = "GATE --"
    if getattr(env, "is_powerloop", False) and getattr(env, "gate_data", None):
        gate_index = int(env.current_gate_idx) % len(env.gate_data)
        gate_text = (
            f"GATE {gate_index + 1}/{len(env.gate_data)} "
            f"{env.gate_data[gate_index]['name']}"
        )

    status = "PAUSED" if paused else "FLYING"
    text = (
        f"{status} | EPISODE {episode}/{num_episodes} | "
        f"SCORE {getattr(env, 'score', 0)} | {gate_text}\n"
        f"ALT {position[2]:.2f} m | SPEED {speed:.2f} m/s | "
        f"HDG {yaw:.1f} deg | REWARD {total_reward:.1f} | "
        f"TIME {sim_time:.1f} s\n"
        "SPACE: pause/resume | Q: stop evaluation"
    )
    return p.addUserDebugText(
        text,
        position + np.array([0.0, 0.0, 0.45]),
        textColorRGB=[1.0, 1.0, 1.0],
        textSize=1.25,
        lifeTime=0,
        replaceItemUniqueId=hud_id,
        physicsClientId=env.CLIENT,
    )


def set_lidar_visibility(env, visible):
    visible = bool(visible)
    if visible == env._show_lidar_debug:
        return
    env._show_lidar_debug = visible
    if not visible:
        for item_id in env.lidar_ids:
            if item_id >= 0:
                p.removeUserDebugItem(int(item_id), physicsClientId=env.CLIENT)
        env.lidar_ids = [-1] * len(env.lidar_ids)


def evaluate():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Configured PPO model does not exist: {os.path.abspath(MODEL_PATH)}"
        )

    env = HoverAviary(
        gui=True,
        obs=ObservationType.KIN,
        act=ActionType.RPM,
        record=RECORD_VIDEO,
        random_targets=RANDOM_TARGETS,
        physics=Physics.PYB,
    )
    try:
        client_id = env.CLIENT
        model = PPO.load(MODEL_PATH, env=env, device=DEVICE, verbose=0)
        obs, _ = env.reset(seed=0)
        controls = create_controls(client_id)
        print(f"Loaded model: {os.path.abspath(MODEL_PATH)}")
        print("Use the PyBullet sliders to pause, change speed, and select camera.")
        print("Keyboard: SPACE pause/resume, Q stop.")

        paused = False
        stopped = False
        last_pause_slider = 0
        last_stop_slider = 0
        last_shadows = 1
        last_reflections = 0
        last_lidar = False
        hud_id = -1
        episode_count = NUM_EPISODES
        episode = 0

        while episode < episode_count and not stopped:
            requested_episodes = int(round(p.readUserDebugParameter(
                controls["episodes"], physicsClientId=client_id
            )))
            episode_count = max(1, requested_episodes)
            if episode >= episode_count:
                break
            episode += 1

            if episode > 1:
                obs, _ = env.reset(seed=episode - 1)
            total_reward = 0.0
            start_time = time.time()
            max_episode_steps = MAX_STEPS or int(
                (env.EPISODE_LEN_SEC + 20) * env.CTRL_FREQ
            )

            for step in range(max_episode_steps):
                if keyboard_pressed(client_id, ord("q")):
                    stopped = True
                    break
                pause_slider = int(round(p.readUserDebugParameter(
                    controls["pause"], physicsClientId=client_id
                )))
                if pause_slider != last_pause_slider:
                    paused = not paused
                    last_pause_slider = pause_slider
                if keyboard_pressed(client_id, p.B3G_SPACE):
                    paused = not paused

                stop_slider = int(round(p.readUserDebugParameter(
                    controls["stop"], physicsClientId=client_id
                )))
                if stop_slider:
                    stopped = True
                    break
                if paused:
                    update_hud(
                        env, episode, episode_count, total_reward, paused, hud_id
                    )
                    while paused and not stopped:
                        if keyboard_pressed(client_id, ord("q")):
                            stopped = True
                            break
                        slider_pause = int(round(p.readUserDebugParameter(
                            controls["pause"], physicsClientId=client_id
                        )))
                        if slider_pause != last_pause_slider:
                            paused = not paused
                            last_pause_slider = slider_pause
                        if keyboard_pressed(client_id, p.B3G_SPACE):
                            paused = not paused
                        if p.readUserDebugParameter(
                            controls["stop"], physicsClientId=client_id
                        ) >= 0.5:
                            stopped = True
                            break
                        time.sleep(0.05)
                    start_time = time.time() - step * env.CTRL_TIMESTEP
                    if stopped:
                        break

                speed = float(p.readUserDebugParameter(
                    controls["speed"], physicsClientId=client_id
                ))
                speed = max(0.1, speed)
                env._presentation_camera_mode = int(round(
                    p.readUserDebugParameter(
                        controls["camera"], physicsClientId=client_id
                    )
                ))

                lidar_value = bool(round(p.readUserDebugParameter(
                    controls["lidar"], physicsClientId=client_id
                )))
                if lidar_value != last_lidar:
                    set_lidar_visibility(env, lidar_value)
                    last_lidar = lidar_value

                shadows = int(round(p.readUserDebugParameter(
                    controls["shadows"], physicsClientId=client_id
                )))
                if shadows != last_shadows:
                    p.configureDebugVisualizer(
                        p.COV_ENABLE_SHADOWS, shadows, physicsClientId=client_id
                    )
                    last_shadows = shadows

                reflections = int(round(p.readUserDebugParameter(
                    controls["reflections"], physicsClientId=client_id
                )))
                if reflections != last_reflections:
                    p.configureDebugVisualizer(
                        p.COV_ENABLE_PLANAR_REFLECTION,
                        reflections,
                        physicsClientId=client_id,
                    )
                    last_reflections = reflections

                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += float(reward)
                if step % max(1, env.CTRL_FREQ // 5) == 0:
                    hud_id = update_hud(
                        env, episode, episode_count, total_reward, paused, hud_id
                    )

                sync(step, start_time, env.CTRL_TIMESTEP / speed)
                if terminated or truncated:
                    break

            print(
                f"Episode {episode}/{episode_count}: "
                f"score={getattr(env, 'score', 0)}, reward={total_reward:.2f}"
            )

        print("Evaluation stopped." if stopped else "Evaluation complete.")
    finally:
        env.close()


if __name__ == "__main__":
    evaluate()
