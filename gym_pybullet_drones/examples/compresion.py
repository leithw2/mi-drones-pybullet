import torch
import torch.nn.utils.prune as prune
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env


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
from gym_pybullet_drones.utils.enums import ObservationType, ActionType, Physics
from RewardPlotter import RewardPlotter

import time
import os
import matplotlib.pyplot as plt
import numpy as np
import torch

def evaluate_model(model_path, multiagent=False, gui=True, record_video=False, output_folder='results', colab=False, episodes=30, max_steps=3000000, speed_factor=1.0):
    DEFAULT_OBS = ObservationType('kin')
    DEFAULT_ACT = ActionType('rpm')
    DEFAULT_AGENTS = 1
    if not multiagent:
        test_env = HoverAviary(gui=gui,
                               obs=DEFAULT_OBS,
                               act=DEFAULT_ACT,
                               record=record_video,
                               initial_xyzs=np.array([[0,0,.5]]),
                               initial_rpys=np.array([[0,0,0]]),
                               random_targets=False, physics=Physics.PYB, pyb_freq = 240, ctrl_freq = 60)
    else:
        test_env = MultiHoverAviary(gui=gui,
                                    num_drones=DEFAULT_AGENTS,
                                    obs=DEFAULT_OBS,
                                    act=DEFAULT_ACT,
                                    record=record_video)
    # logger = Logger(logging_freq_hz=int(test_env.CTRL_FREQ),
    #             num_drones=DEFAULT_AGENTS if multiagent else 1,
    #             output_folder=output_folder,
    #             colab=colab)
    import os
    print("Directorio actual:", os.getcwd())
    #plotter = RewardPlotter(title="Reward en Tiempo Real")
    # Cargar modelo maestro (entrenado)
    DEFAULT_OUTPUT_FOLDER = 'results'
    model = PPO.load(model_path, env=test_env, device="cpu", verbose=0)

    policy_original = model.policy

    # Crear una copia profunda para podar (no modificar el original)
    import copy
    policy_podada = copy.deepcopy(policy_original)

    # Aplicar poda agresiva (80%) a las capas lineales ocultas
    EXCLUDE_NAMES = ['action_net', 'value_net']
    for name, module in policy_podada.named_modules():
        if isinstance(module, torch.nn.Linear):
            if any(excluded in name for excluded in EXCLUDE_NAMES):
                prune.l1_unstructured(module, name="weight", amount=0.3)  # suave
            else:
                prune.l1_unstructured(module, name="weight", amount=0.8)  # agresiva

    # Hacer la poda permanente (opcional pero recomendado para liberar memoria)
    for name, module in policy_podada.named_modules():
        if isinstance(module, torch.nn.Linear):
            try:
                prune.remove(module, 'weight')
            except:
                pass
    model.policy = policy_podada
    
    visualizar_esparsidad(policy_original, policy_podada, 'mlp_extractor.policy_net.0')

    
    for ep in range(episodes):
        obs, info = test_env.reset(seed=ep, options={})
        start = time.time()
        total_reward = 0

        for i in range(max_steps or (test_env.EPISODE_LEN_SEC)*300*test_env.CTRL_FREQ):
            action, _states = model.predict(obs, deterministic=True, )
            obs, reward, terminated, truncated, info = test_env.step(action)
            obs2 = obs.squeeze()
            act2 = action.squeeze()
            total_reward += reward
            # if DEFAULT_OBS == ObservationType.KIN:
            #     if not multiagent:
            #         logger.log(drone=0,
            #             timestamp=i/test_env.CTRL_FREQ,
            #             state=np.hstack([obs2[0:3],
            #                                 np.zeros(4),
            #                                 obs2[3:12],
            #                                 act2]),
            #             control=np.zeros(12))
            #     else:
            #         for d in range(DEFAULT_AGENTS):
            #             logger.log(drone=d,
            #                 timestamp=i/test_env.CTRL_FREQ,
            #                 state=np.hstack([obs2[d][0:3],
            #                                     np.zeros(4),
            #                                     obs2[d][3:12],
            #                                     act2[d]]),
            #                 control=np.zeros(12))
            
            # if i % 5 == 0: # Solo grafica cada 10 pasos
            #     plotter.update(total_reward)
            if hasattr(test_env, 'render'):
                test_env.render()
            sync(i, start, test_env.CTRL_TIMESTEP * speed_factor)
            if terminated or truncated:
                break
        print(f"Episodio {ep+1}: reward total = {total_reward}")
    test_env.close()
    # if DEFAULT_OBS == ObservationType.KIN:
    #     logger.plot()
def visualizar_esparsidad(policy_original, policy_podada, capa_nombre='mlp_extractor.policy_net.0'):
    """
    Compara los pesos de una capa específica antes y después de la poda.
    Muestra una imagen binaria (negro = peso cero, blanco = peso no cero).
    """
    # Obtener los pesos de la capa original (extraer del state_dict si no está podada)
    # Nota: si ya aplicaste prune.remove(), los pesos están en 'weight' normal.
    pesos_orig = policy_original.state_dict()[capa_nombre + '.weight'].detach().cpu().numpy()
    pesos_pod  = policy_podada.state_dict()[capa_nombre + '.weight'].detach().cpu().numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Mapa binario: 1 donde |peso| > 1e-6, 0 en otro caso
    sparsity_orig = np.abs(pesos_orig) > 1e-6
    sparsity_pod  = np.abs(pesos_pod) > 1e-6
    
    axes[0].imshow(sparsity_orig, aspect='auto', cmap='gray_r')
    axes[0].set_title(f'Original - No ceros: {np.sum(sparsity_orig)} / {sparsity_orig.size}')
    axes[1].imshow(sparsity_pod, aspect='auto', cmap='gray_r')
    axes[1].set_title(f'Podada - No ceros: {np.sum(sparsity_pod)} / {sparsity_pod.size}')
    
    for ax in axes:
        ax.set_xlabel('Neurona entrada')
        ax.set_ylabel('Neurona salida')
    plt.tight_layout()
    plt.savefig('sparsity_comparison.png', dpi=150)
    plt.show()
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluar un modelo PPO de gym-pybullet-drones')
    parser.add_argument('--model_path', type=str, default= os.path.join('results', 'compresion', 'final_model.zip'), help='Ruta al archivo .zip del modelo PPO')
    parser.add_argument('--multiagent', default=False, type=bool, help='Usar MultiHoverAviary (default: False)')
    parser.add_argument('--gui', default=True, type=bool, help='Mostrar GUI (default: True)')
    parser.add_argument('--record_video', default=True, type=bool, help='Grabar video (default: False)')
    parser.add_argument('--output_folder', default='results', type=str, help='Carpeta de logs')
    parser.add_argument('--colab', default=False, type=bool, help='Modo Colab')
    parser.add_argument('--episodes', default=90, type=int, help='Cantidad de episodios a evaluar')
    parser.add_argument('--max_steps', default=None, type=int, help='Máximo de pasos por episodio')
    parser.add_argument('--speed_factor', default=1, type=float, help='Multiplicador de velocidad de la visualización (1.0=normal, <1.0=rápido, >1.0=lento)')
    args = parser.parse_args()
    evaluate_model(**vars(args))
