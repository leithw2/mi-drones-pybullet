import os
import argparse
import numpy as np
import torch
from stable_baselines3 import PPO
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gym_pybullet_drones.utils.utils import sync
import time

def evaluate_model(model_path, multiagent=False, gui=True, record_video=False, output_folder='results', colab=False, episodes=3, max_steps=None, speed_factor=1.0):
    DEFAULT_OBS = ObservationType('kin')
    DEFAULT_ACT = ActionType('rpm')
    DEFAULT_AGENTS = 1
    if not multiagent:
        test_env = HoverAviary(gui=gui,
                               obs=DEFAULT_OBS,
                               act=DEFAULT_ACT,
                               record=record_video, 
                               random_targets=True)
    else:
        test_env = MultiHoverAviary(gui=gui,
                                    num_drones=DEFAULT_AGENTS,
                                    obs=DEFAULT_OBS,
                                    act=DEFAULT_ACT,
                                    record=record_video)
    logger = Logger(logging_freq_hz=int(test_env.CTRL_FREQ),
                num_drones=DEFAULT_AGENTS if multiagent else 1,
                output_folder=output_folder,
                colab=colab)
    model = PPO.load(model_path, env=test_env, device="cpu")
    for ep in range(episodes):
        obs, info = test_env.reset(seed=ep, options={})
        start = time.time()
        total_reward = 0
        for i in range(max_steps or (test_env.EPISODE_LEN_SEC+20)*test_env.CTRL_FREQ):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            obs2 = obs.squeeze()
            act2 = action.squeeze()
            total_reward += reward
            if DEFAULT_OBS == ObservationType.KIN:
                if not multiagent:
                    logger.log(drone=0,
                        timestamp=i/test_env.CTRL_FREQ,
                        state=np.hstack([obs2[0:3],
                                            np.zeros(4),
                                            obs2[3:15],
                                            act2]),
                        control=np.zeros(12))
                else:
                    for d in range(DEFAULT_AGENTS):
                        logger.log(drone=d,
                            timestamp=i/test_env.CTRL_FREQ,
                            state=np.hstack([obs2[d][0:3],
                                                np.zeros(4),
                                                obs2[d][3:15],
                                                act2[d]]),
                            control=np.zeros(12))
            if hasattr(test_env, 'render'):
                test_env.render()
            sync(i, start, test_env.CTRL_TIMESTEP * speed_factor)
            if terminated or truncated:
                break
        print(f"Episodio {ep+1}: reward total = {total_reward}")
    test_env.close()
    if DEFAULT_OBS == ObservationType.KIN:
        logger.plot()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluar un modelo PPO de gym-pybullet-drones')
    parser.add_argument('--model_path', type=str, default= os.path.join('results','save-08.27.2025_23.18.20', 'best_model.zip'), help='Ruta al archivo .zip del modelo PPO')
    parser.add_argument('--multiagent', default=False, type=bool, help='Usar MultiHoverAviary (default: False)')
    parser.add_argument('--gui', default=True, type=bool, help='Mostrar GUI (default: True)')
    parser.add_argument('--record_video', default=False, type=bool, help='Grabar video (default: False)')
    parser.add_argument('--output_folder', default='results', type=str, help='Carpeta de logs')
    parser.add_argument('--colab', default=False, type=bool, help='Modo Colab')
    parser.add_argument('--episodes', default=3, type=int, help='Cantidad de episodios a evaluar')
    parser.add_argument('--max_steps', default=None, type=int, help='Máximo de pasos por episodio')
    parser.add_argument('--speed_factor', default=1.0, type=float, help='Multiplicador de velocidad de la visualización (1.0=normal, <1.0=rápido, >1.0=lento)')
    args = parser.parse_args()
    evaluate_model(**vars(args))
