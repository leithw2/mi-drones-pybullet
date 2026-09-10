import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import numpy as np
import pybullet as p
import torch
import time
from stable_baselines3 import PPO
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType, Physics

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')

def evaluate_model(model_path, multiagent=False, gui=False, record_video=False, output_folder='results', colab=False, episodes=30, max_steps=3000000):
    DEFAULT_OBS = ObservationType('kin')
    DEFAULT_ACT = ActionType('rpm')
    DEFAULT_AGENTS = 1
    
    # 1. Desactivar la GUI para evitar cuellos de botella de renderizado
    if not multiagent:
        test_env = HoverAviary(
            gui=gui,
            obs=DEFAULT_OBS,
            act=DEFAULT_ACT,
            record=record_video,
            initial_xyzs=np.array([[0, 0, 1]]),
            initial_rpys=np.array([[0, 0, 0]]),
            random_targets=False,
            physics=Physics.PYB,
            pyb_freq=240,
            ctrl_freq=60
        )
    else:
        test_env = MultiHoverAviary(
            gui=gui,
            num_drones=DEFAULT_AGENTS,
            obs=DEFAULT_OBS,
            act=DEFAULT_ACT,
            record=record_video
        )
    if gui:
        p.setRealTimeSimulation(0, physicsClientId=test_env.CLIENT)
    # 2. Cargar en GPU si está disponible (CUDA)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PPO.load(model_path, env=test_env, device=device, verbose=0)

    start_total_time = time.time()

    for ep in range(episodes):
        obs, info = test_env.reset(seed=ep, options={})
        total_reward = 0

        for i in range(max_steps or (test_env.EPISODE_LEN_SEC * 300 * test_env.CTRL_FREQ)):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            total_reward += reward

            # ELIMINADO: test_env.render(mode='rgb_array') -> Consume procesador innecesariamente
            # ELIMINADO: sync(...) -> Elimina el limitador de velocidad en tiempo real

            if terminated or truncated:
                break
        
        print(f"Episodio {ep+1}: reward total = {total_reward}")

    test_env.close()
    print(f"Tiempo total transcurrido: {time.time() - start_total_time:.2f} segundos")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluar un modelo PPO de gym-pybullet-drones a máxima velocidad')
    default_model_path = os.path.join(RESULTS_DIR, 'ToF_yawLocal09.02.2026_18.43.31', 'best_model')
    
    # 3. Argumentos ajustados para rendimiento máximo por defecto
    parser.add_argument('--model_path', type=str, default=default_model_path, help='Ruta al modelo PPO')
    parser.add_argument('--multiagent', action='store_true', help='Usar MultiHoverAviary')
    parser.add_argument('--gui', action='store_true', default=False, help='Mostrar GUI (Desactivado por defecto para velocidad)')
    parser.add_argument('--record_video', action='store_true', default=False, help='Grabar video (Desactivado por defecto)')
    parser.add_argument('--episodes', default=90, type=int, help='Cantidad de episodios')
    parser.add_argument('--max_steps', default=None, type=int, help='Máximo de pasos por episodio')
    
    args = parser.parse_args()
    evaluate_model(**vars(args))