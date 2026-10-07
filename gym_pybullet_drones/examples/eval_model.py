import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import numpy as np
import pybullet as p
import time
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from gym_pybullet_drones.utils.enums import DroneModel, ObservationType, ActionType, Physics
from gym_pybullet_drones.utils.evaluation import (
    EVAL_DEVICE,
    EVAL_SEEDS,
    POWERLOOP_ORIENTATION_VARIATION,
    POWERLOOP_POSITION_VARIATION,
    POWERLOOP_TRACK_SCALE,
    make_eval_env,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results')

def control_simulation_speed(sim_speed, ctrl_freq):
    """
    Controla la velocidad de simulación.

    sim_speed:
        0    = máxima velocidad
        1.0  = tiempo real
        2.0  = 2x
        0.5  = 0.5x
    """

    if sim_speed <= 0:
        return 0.0

    return 1.0 / (ctrl_freq * sim_speed)

def evaluate_model(model_path, multiagent=False, gui=False, record_video=False, output_folder='results', colab=False, episodes=10, sim_speed=1.0, track_scale=POWERLOOP_TRACK_SCALE, track_position_variation=POWERLOOP_POSITION_VARIATION, track_orientation_variation=POWERLOOP_ORIENTATION_VARIATION):
    DEFAULT_OBS = ObservationType('kin')
    DEFAULT_DRONE = DroneModel.CF2X250
    DEFAULT_ACT = ActionType.BRUSHLESS_THRUST
    DEFAULT_AGENTS = 1
    
    test_env = make_eval_env(
        multiagent=multiagent,
        gui=gui,
        record=record_video,
        drone_model=DEFAULT_DRONE,
        obs=DEFAULT_OBS,
        act=DEFAULT_ACT,
        physics=Physics.PYB_WIND,
        random_targets=False,
        num_drones=DEFAULT_AGENTS,
        track_scale=track_scale,
        track_position_variation=track_position_variation,
        track_orientation_variation=track_orientation_variation,
    )

    if gui:
        p.setRealTimeSimulation(
            0,
            physicsClientId=test_env.unwrapped.CLIENT
        )

    ctrl_freq = test_env.unwrapped.CTRL_FREQ

    sim_timestep = control_simulation_speed(
        sim_speed,
        ctrl_freq
    )

    print(f"[INFO] Velocidad de simulación: {sim_speed}x")
    print(f"[INFO] Control frequency: {ctrl_freq} Hz")
    print(f"[INFO] Evaluating {episodes} episodes with seeds {EVAL_SEEDS}")
    print(f"[INFO] Device: {EVAL_DEVICE}")
    
    
    model = PPO.load(model_path, env=test_env, device=EVAL_DEVICE, verbose=0)

    start_total_time = time.time()

    def pace_evaluation(_locals, _globals):
        if sim_timestep > 0:
            time.sleep(sim_timestep)

    episode_rewards, _episode_lengths = evaluate_policy(
        model,
        test_env,
        n_eval_episodes=episodes,
        deterministic=True,
        render=False,
        return_episode_rewards=True,
        callback=pace_evaluation,
    )

    for ep, reward in enumerate(episode_rewards, start=1):
        print(f"Episodio {ep}: reward total = {reward}")
    print(
        f"Promedio ({episodes} episodios): "
        f"{np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}"
    )

    test_env.close()
    print(f"Tiempo total transcurrido: {time.time() - start_total_time:.2f} segundos")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluar un modelo PPO de gym-pybullet-drones a máxima velocidad')
    default_model_path = os.path.join(RESULTS_DIR, 'Race09.28.2026_15.41.08', 'best_model')
    
    parser.add_argument('--model_path', type=str, default=default_model_path, help='Ruta al modelo PPO')
    parser.add_argument('--multiagent', action='store_true', help='Usar MultiHoverAviary')
    parser.add_argument('--gui', action='store_true', default=False, help='Mostrar GUI')
    parser.add_argument('--record_video', action='store_true', default=False, help='Grabar video')
    parser.add_argument('--episodes', default=10, type=int, help='Cantidad de episodios (por defecto, 10 como EvalCallback)')
    parser.add_argument('--sim_speed', default=1.0, type=float, help='Velocidad de simulación: 0=maxima, 1=tiempo real, 2=2x')
    parser.add_argument('--track_scale', default=POWERLOOP_TRACK_SCALE, type=float, help='Escala espacial de la pista; no cambia el tamaño de las puertas')
    parser.add_argument('--track_position_variation', default=POWERLOOP_POSITION_VARIATION, type=float, help='Variación de posición como fracción del tamaño de pista; 0 desactiva')
    parser.add_argument('--track_orientation_variation', default=POWERLOOP_ORIENTATION_VARIATION, type=float, help='Variación de yaw como fracción de pi; 0 desactiva')
    args = parser.parse_args()
    evaluate_model(**vars(args))