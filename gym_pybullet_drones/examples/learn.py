from stable_baselines3.common.callbacks import BaseCallback
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType


# Callback para renderizar el entorno de entrenamiento en cada paso, con soporte para cámara lenta
class TrainRenderCallback(BaseCallback):
    def __init__(self, env, sync_human_speed=False, slow_factor=0.1, verbose=1):
        """
        slow_factor > 1.0 hará la simulación más lenta (cámara lenta).
        slow_factor = 1.0 es velocidad real.
        slow_factor < 1.0 es más rápido.
        """
        super().__init__(verbose)
        self.env = env
        self.sync_human_speed = sync_human_speed
        self.slow_factor = slow_factor
        self._start_time = None

    def _on_training_start(self) -> None:
        if self.sync_human_speed or self.slow_factor != 1.0:
            self._start_time = time.time()

    def _on_step(self) -> bool:
        if hasattr(self.env, 'render'):
            try:
                self.env.render()
            except Exception as e:
                print(f"[WARN] Render error during training: {e}")
        # Sincronizar a velocidad humana o cámara lenta si está activado
        if (self.sync_human_speed or self.slow_factor != 1.0) and hasattr(self.env, 'CTRL_TIMESTEP') and self._start_time is not None:
            i = self.num_timesteps
            elapsed = time.time() - self._start_time
            expected = i * self.env.CTRL_TIMESTEP * self.slow_factor
            to_wait = expected - elapsed
            if to_wait > 0:
                time.sleep(to_wait)
        return True
"""Script demonstrating the use of `gym_pybullet_drones`'s Gymnasium interface.

Classes HoverAviary and MultiHoverAviary are used as learning envs for the PPO algorithm.

Example
-------
In a terminal, run as:

    $ python learn.py --multiagent false
    $ python learn.py --multiagent true

Notes
-----
This is a minimal working example integrating `gym-pybullet-drones` with 
reinforcement learning library `stable-baselines3`.

"""
import os
import time
from datetime import datetime
import argparse
import gymnasium as gym
import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

print(torch.cuda.is_available())
DEFAULT_GUI = False
DEFAULT_RECORD_VIDEO = True
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

DEFAULT_OBS = ObservationType('kin') # 'kin' or 'rgb'
DEFAULT_ACT = ActionType('rpm') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'
DEFAULT_AGENTS = 1
DEFAULT_MA = False
physics=Physics.PYB # Physics.PYB or Physics.PYB_CUSTOM or Physics.PYB_WIND
CONTINUE_FROM = os.path.join(DEFAULT_OUTPUT_FOLDER,'save-08.27.2025_14.31.12')
CONTINUE_FROM = None # None or path to saved model folder


def run(multiagent=DEFAULT_MA, output_folder=DEFAULT_OUTPUT_FOLDER, gui=DEFAULT_GUI, plot=True, colab=DEFAULT_COLAB, record_video=DEFAULT_RECORD_VIDEO, local=True, continue_from=None):
    # Si se especifica un modelo para continuar, usar ese path, si no, crear uno nuevo
    print(f"[AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA] Continuando entrenamiento desde: {continue_from}")
    if continue_from:
        filename = continue_from
        print(f"[INFO] Continuando entrenamiento desde: {filename}")
    else:
        filename = os.path.join(output_folder,'save-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    if not os.path.exists(filename):
        os.makedirs(filename+'/')
        print(f"[INFO] Creando carpeta {filename}/")
    # Alternar entre entrenamiento con render (GUI) y entrenamiento rápido (vectorizado)
    if gui:
        if not multiagent:
            train_env = HoverAviary(gui=True, obs=DEFAULT_OBS, act=DEFAULT_ACT, physics=physics)
            eval_env = HoverAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT, physics=physics)
        else:
            train_env = MultiHoverAviary(gui=True, num_drones=DEFAULT_AGENTS, obs=DEFAULT_OBS, act=DEFAULT_ACT, physics=physics)
            eval_env = MultiHoverAviary(num_drones=DEFAULT_AGENTS, obs=DEFAULT_OBS, act=DEFAULT_ACT, physics=physics)
        use_render_callback = True
    else:
        if not multiagent:
            train_env = make_vec_env(HoverAviary,
                                    env_kwargs=dict(obs=DEFAULT_OBS, act=DEFAULT_ACT,physics=physics),
                                    n_envs=12,
                                    seed=0)
            eval_env = HoverAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT,physics=physics)
        else:
            train_env = make_vec_env(MultiHoverAviary,
                                    env_kwargs=dict(num_drones=DEFAULT_AGENTS, obs=DEFAULT_OBS, act=DEFAULT_ACT,physics=physics),
                                    n_envs=12,
                                    seed=0)
            eval_env = MultiHoverAviary(num_drones=DEFAULT_AGENTS, obs=DEFAULT_OBS, act=DEFAULT_ACT,physics=physics)
        use_render_callback = False

    #### Check the environment's spaces ########################
    print('[INFO] Action space:', train_env.action_space)
    print('[INFO] Observation space:', train_env.observation_space)


    #### Train the model con manejo de interrupción ###########
    if continue_from and os.path.isfile(os.path.join(filename, 'final_model.zip')):
        print(f"[INFO] Cargando modelo guardado de {os.path.join(filename, 'final_model.zip')}")
        model = PPO.load(os.path.join(filename, 'final_model.zip'), env=train_env, device="cpu")
        # El modelo ya contiene num_timesteps internamente
        model.tensorboard_log = filename+'/tb/'
    else:
        model = PPO('MlpPolicy',
                    train_env,
                    device="cpu",
                    tensorboard_log=filename+'/tb/',
                    n_steps=4096,          # Mayor para más estabilidad
                    batch_size=128,
                    n_epochs=20,        # Tamaño de mini-lote
                    learning_rate = lambda p: 0.00005 + (0.0007 - 0.00005) * ((p - 0.25) / 0.75) if p > 0.25 else 0.00005,
                    policy_kwargs=dict(
                    net_arch=[dict(pi=[256, 256], vf=[256, 256])],
                    activation_fn=torch.nn.Tanh,  # Suaviza salidas
                    ),
                    ent_coef=0.015,
                    clip_range=0.3,
                    verbose=1)
        print(f"[INFO] Creando modelo en {filename}") #
        
    #### Target cumulative rewards (problem-dependent) ##########
    if DEFAULT_ACT == ActionType.ONE_D_RPM:
        target_reward = 474.15 if not multiagent else 949.5
    else:
        target_reward = 4500 if not multiagent else 920.
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=target_reward, verbose=1)
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=callback_on_best,
        verbose=1,
        best_model_save_path=filename+'/',
        log_path=filename+'/',
        eval_freq=int(2000),
        deterministic=True,
        render=False
    )
    try:
        if use_render_callback:
            train_render_callback = TrainRenderCallback(train_env, sync_human_speed=False)
            model.learn(total_timesteps=int(1e7) if local else int(1e2),
                        callback=[eval_callback, train_render_callback],
                        log_interval=100,
                        reset_num_timesteps=False if continue_from else True)
        else:
            model.learn(total_timesteps=int(1e7) if local else int(1e2),
                        callback=eval_callback,
                        log_interval=100,
                        reset_num_timesteps=False if continue_from else True)
    except KeyboardInterrupt:
        print("\n[INFO] Entrenamiento interrumpido por el usuario. Guardando el modelo actual...")
    finally:
        #### Save the model ########################################
        model.save(filename+'/final_model.zip')
        print(f"[INFO] Modelo guardado en {filename+'/final_model.zip'}")

    #### Print training progression ############################
    if os.path.exists(filename+'/evaluations.npz'):
        with np.load(filename+'/evaluations.npz') as data:
            for j in range(data['timesteps'].shape[0]):
                print(str(data['timesteps'][j])+","+str(data['results'][j][0])) 

    if local:
        input("Press Enter to continue...")

    #if os.path.isfile(filename+'/final_model.zip'):
    #    path = filename+'/final_model.zip' 
    path = None
    if os.path.isfile(filename+'/best_model.zip'):
        path = filename+'/best_model.zip'
    elif os.path.isfile(filename+'/final_model.zip'):
        path = filename+'/final_model.zip'
    else:
        print("[ERROR]: no model under the specified path", filename)
    if path:
        model = PPO.load(path)

    #### Show (and record a video of) the model's performance ##
    # Siempre mostrar el entorno final con GUI para visualizar el resultado, aunque el entrenamiento haya sido sin GUI
    if not multiagent:
        test_env = HoverAviary(gui=True,
                               obs=DEFAULT_OBS,
                               act=DEFAULT_ACT,
                               record=record_video,
                               physics=physics,
                               )
        test_env_nogui = HoverAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT)
    else:
        test_env = MultiHoverAviary(gui=True,
                                        num_drones=DEFAULT_AGENTS,
                                        obs=DEFAULT_OBS,
                                        act=DEFAULT_ACT,
                                        record=record_video,)
        test_env_nogui = MultiHoverAviary(num_drones=DEFAULT_AGENTS, obs=DEFAULT_OBS, act=DEFAULT_ACT)
    logger = Logger(logging_freq_hz=int(test_env.CTRL_FREQ),
                num_drones=DEFAULT_AGENTS if multiagent else 1,
                output_folder=output_folder,
                colab=colab
                )

    """    mean_reward, std_reward = evaluate_policy(model,
                                              test_env_nogui,
                                              n_eval_episodes=10,
                                              )
    print("\n\n\nMean reward ", mean_reward, " +- ", std_reward, "\n\n") """


    if path:
        obs, info = test_env.reset(seed=0, options={})
        start = time.time()
        for i in range((test_env.EPISODE_LEN_SEC+20)*test_env.CTRL_FREQ):
            action, _states = model.predict(obs,
                                            deterministic=True,
                                            )
            obs, reward, terminated, truncated, info = test_env.step(action)
            obs2 = obs.squeeze()
            act2 = action.squeeze()
            print("Obs:", obs, "\tAction", action, "\tReward:", reward, "\tTerminated:", terminated, "\tTruncated:", truncated)
            if DEFAULT_OBS == ObservationType.KIN:
                if not multiagent:
                    logger.log(drone=0,
                        timestamp=i/test_env.CTRL_FREQ,
                        state=np.hstack([obs2[0:3],
                                            np.zeros(4),
                                            obs2[3:15],
                                            act2
                                            ]),
                        control=np.zeros(12)
                        )
                else:
                    for d in range(DEFAULT_AGENTS):
                        logger.log(drone=d,
                            timestamp=i/test_env.CTRL_FREQ,
                            state=np.hstack([obs2[d][0:3],
                                                np.zeros(4),
                                                obs2[d][3:15],
                                                act2[d]
                                                ]),
                            control=np.zeros(12)
                            )
            if hasattr(test_env, 'render'):
                test_env.render()
            print(terminated)
            sync(i, start, test_env.CTRL_TIMESTEP)
            if terminated:
                obs, info = test_env.reset(seed=0, options={})
            if truncated:
                obs, info = test_env.reset(seed=0, options={})    
        test_env.close()
    else:
        print("[ERROR]: No se pudo cargar el modelo para la evaluación final.")

    if plot and DEFAULT_OBS == ObservationType.KIN:
        logger.plot()

if __name__ == '__main__':
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script')
    parser.add_argument('--multiagent',         default=DEFAULT_MA,            type=str2bool,      help='Whether to use example LeaderFollower instead of Hover (default: False)', metavar='')
    parser.add_argument('--gui',                default=DEFAULT_GUI,           type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VIDEO,  type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--output_folder',      default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB,         type=bool,          help='Whether example is being run by a notebook (default: "False")', metavar='')
    parser.add_argument('--continue_from',      default=CONTINUE_FROM,                  type=str,           help='Ruta a la carpeta del modelo guardado para continuar entrenamiento', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))
