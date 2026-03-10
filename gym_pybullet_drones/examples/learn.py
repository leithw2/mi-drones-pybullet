from stable_baselines3.common.callbacks import BaseCallback
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType


# Callback para renderizar el entorno de entrenamiento en cada paso, con soporte para cámara lenta
class TrainRenderCallback(BaseCallback):
    def __init__(self, env, sync_human_speed=False, slow_factor=1, verbose=1):
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
from torch.utils.tensorboard import SummaryWriter

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import constant_fn

from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.envs.MultiHoverAviary import MultiHoverAviary
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType


# Device autodetection and PyTorch perf tweaks
print("CUDA available:", torch.cuda.is_available())
DEVICE ="cpu"
# Use half of logical cores to avoid oversubscription with VecEnv workers
try:
    torch.set_num_threads(max(1, (os.cpu_count() or 1)//2))
except Exception:
    pass
try:
    torch.backends.cudnn.benchmark = True
except Exception:
    pass
#N_ENVS = min(16, max(1, (os.cpu_count() or 1)))
N_ENVS = 12 # For debugging, set to 1 to avoid multiprocessing issues
# Callback para loggear en TensorBoard: gráfico del modelo, histogramas de parámetros y LR
class TensorboardCallback(BaseCallback):
    def __init__(self, tb_log_dir, log_freq=2000, verbose=0):
        super().__init__(verbose)
        self.tb_log_dir = tb_log_dir
        self.log_freq = log_freq
        self.writer = None

    def _on_training_start(self) -> None:
        try:
            self.writer = SummaryWriter(self.tb_log_dir)
            # Intentar agregar el grafo del modelo (puede fallar con algunas políticas)
            obs_space = None
            try:
                obs_space = self.training_env.observation_space.shape
            except Exception:
                # VecEnv/Wrapper pueden requerir acceder al env interno
                try:
                    obs_space = self.training_env.envs[0].observation_space.shape
                except Exception:
                    obs_space = None
            if obs_space is not None:
                # Obtener shape de observación y tensor dummy
                obs_shape = obs_space
                dummy = torch.zeros(1, *obs_shape)
                dummy_flat = dummy.view(1, -1)   # Flatten si el extractor es FlattenExtractor

                try:
                    # Trazar solo las sub-redes deterministas bajo torch.no_grad()
                    with torch.no_grad():
                        policy_net = self.model.policy.mlp_extractor.policy_net
                        action_net = self.model.policy.action_net
                        policy_net.eval()
                        action_net.eval()
                        # pasar dummy_flat.detach() para evitar grad en non-leaf tensors
                        try:
                            self.writer.add_graph(policy_net, (dummy_flat.detach(),))
                        except Exception as e:
                            print(f"[WARN] add_graph(policy_net) failed: {e}")
                        try:
                            latent = policy_net(dummy_flat.detach())
                            self.writer.add_graph(action_net, (latent.detach(),))
                        except Exception as e:
                            print(f"[WARN] add_graph(action_net) failed: {e}")
                except Exception as e:
                    print(f"[WARN] add_graph partial failed: {e}")
            # Always add textual architecture as fallback
            try:
                self.writer.add_text('model/architecture', str(self.model.policy))
            except Exception:
                pass
        except Exception as e:
            print(f"[WARN] Falló inicializar SummaryWriter: {e}")

    def _on_step(self) -> bool:
        # Loggear histogramas de parámetros periódicamente
        try:
            if self.writer is None:
                return True
            if self.num_timesteps % self.log_freq == 0:
                for name, param in self.model.policy.named_parameters():
                    try:
                        self.writer.add_histogram(name, param.detach().cpu().numpy(), self.num_timesteps)
                    except Exception:
                        pass
                # Intentar loggear learning rate si existe
                try:
                    lr = None
                    if hasattr(self.model, 'lr_schedule') and callable(self.model.lr_schedule):
                        lr = float(self.model.lr_schedule(self.num_timesteps))
                    if lr is not None:
                        self.writer.add_scalar('train/learning_rate', lr, self.num_timesteps)
                except Exception:
                    pass
        except Exception as e:
            print(f"[WARN] TensorboardCallback error on step: {e}")
        return True

    def _on_training_end(self) -> None:
        if self.writer is not None:
            try:
                self.writer.close()
            except Exception:
                pass
DEFAULT_GUI = False
DEFAULT_RECORD_VIDEO = True
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

DEFAULT_OBS = ObservationType('kin') # 'kin' or 'rgb'
DEFAULT_ACT = ActionType('rpm') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'
DEFAULT_AGENTS = 1
DEFAULT_MA = False
physics=Physics.PYB # Physics.PYB or Physics.PYB_CUSTOM or Physics.PYB_WIND
CONTINUE_FROM = os.path.join(DEFAULT_OUTPUT_FOLDER,'obs21_bufferAction1_24x12_64x64_lidar_nowind_map_save-03.09.2026_15.20.02')
#CONTINUE_FROM = None # None or path to saved model folder
RANDOM_TARGETS=True

def run(multiagent=DEFAULT_MA, output_folder=DEFAULT_OUTPUT_FOLDER, gui=DEFAULT_GUI, plot=True, colab=DEFAULT_COLAB, record_video=DEFAULT_RECORD_VIDEO, local=True, continue_from=None):
    # Si se especifica un modelo para continuar, usar ese path, si no, crear uno nuevo
    print(f"Continuando entrenamiento desde: {continue_from}")
    if continue_from:
        filename = continue_from
        print(f"[INFO] Continuando entrenamiento desde: {filename}")
    else:
        filename = os.path.join(output_folder,'obs21_bufferAction1_24x12_64x64_lidar_nowind_map_save-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S"))
    if not os.path.exists(filename):
        os.makedirs(filename+'/')
        print(f"[INFO] Creando carpeta {filename}/")
    # Alternar entre entrenamiento con render (GUI) y entrenamiento rápido (vectorizado)
    if gui:
        if not multiagent:
            train_env = HoverAviary(gui=True, obs=DEFAULT_OBS, act=DEFAULT_ACT, random_targets=RANDOM_TARGETS, physics=physics)
            eval_env = HoverAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT, random_targets=RANDOM_TARGETS, physics=physics)
            eval_env = Monitor(eval_env)
        else:
            train_env = MultiHoverAviary(gui=True, num_drones=DEFAULT_AGENTS, obs=DEFAULT_OBS, act=DEFAULT_ACT, random_targets=RANDOM_TARGETS, physics=physics)
            eval_env = MultiHoverAviary(num_drones=DEFAULT_AGENTS, obs=DEFAULT_OBS, act=DEFAULT_ACT, random_targets=RANDOM_TARGETS, physics=physics)
            eval_env = Monitor(eval_env)
        use_render_callback = True
    else:
        if not multiagent:
            train_env = make_vec_env(HoverAviary,
                                    env_kwargs=dict(obs=DEFAULT_OBS, act=DEFAULT_ACT, random_targets=RANDOM_TARGETS, physics=physics),
                                    n_envs=N_ENVS,
                                    seed=0,
                                    )
            eval_env = HoverAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT, random_targets=RANDOM_TARGETS, physics=physics)
            eval_env = Monitor(eval_env)
        else:
            train_env = make_vec_env(MultiHoverAviary,
                                    env_kwargs=dict(num_drones=DEFAULT_AGENTS, obs=DEFAULT_OBS, act=DEFAULT_ACT, random_targets=RANDOM_TARGETS, physics=physics),
                                    n_envs=N_ENVS,
                                    seed=0,
                                    )
            eval_env = MultiHoverAviary(num_drones=DEFAULT_AGENTS, obs=DEFAULT_OBS, act=DEFAULT_ACT, random_targets=RANDOM_TARGETS, physics=physics)
            eval_env = Monitor(eval_env)
        use_render_callback = False

    #### Check the environment's spaces ########################
    print('[INFO] Action space:', train_env.action_space)
    print('[INFO] Observation space:', train_env.observation_space)


    #### Train the model con manejo de interrupción ###########
    if continue_from and os.path.isfile(os.path.join(filename, 'final_model.zip')):
        print(f"[INFO] Cargando modelo guardado de {os.path.join(filename, 'final_model.zip')}")
        model = PPO.load(os.path.join(filename, 'final_model.zip'), env=train_env, device=DEVICE)
        # 2. Modificar parámetros (Fine-tuning)
        # model.learning_rate = 0.0001  # Bajamos la tasa para mayor estabilidad
        # model.ent_coef = 0.001       # Reducimos la exploración aleatoria
        # model.clip_range = constant_fn(0.2)
        # El modelo ya contiene num_timesteps internamente
        model.tensorboard_log = filename+'/tb/'
    else:
        model = PPO('MlpPolicy',
                train_env,
                device=DEVICE,
                tensorboard_log=filename+'/tb/',
                n_steps=int(512*8),     # Aumentado para más muestras por actualización, mejor estimación de la ventaja, pero más memoria y menos actualizaciones por paso
                batch_size=int(256*8),    # Reducido para permitir más actualizaciones por paso, pero puede aumentar la varianza del gradiente
                n_epochs=int(10*8),       # Aumentado para más actualizaciones por paso
                gae_lambda=0.95, # Valor por defecto, buen compromiso entre bias y varianza
                    learning_rate = lambda p: 0.00005 + (0.0007 - 0.00005) * ((p - 0.25) / 0.75) if p > 0.25 else 0.00005,
                    #learning_rate=0.0001,
                    policy_kwargs=dict(
                        net_arch=[dict(pi=[24,12], vf=[64, 64])],
                        activation_fn=torch.nn.Tanh,  # Suaviza salidas
                        ##### TENSORBOARD MOD: Log Histograms y Gráfico #####
                        log_std_init=-2.0, # Valor por defecto, ayuda a la estabilidad
                        ortho_init=True, # Inicialización ortogonal para estabilidad
                        # El registro de gradientes/pesos se activa internamente si verbose=1 
                        # y log_interval es bajo, pero a veces necesitas forzarlo:
                        # SB3 registra estas métricas automáticamente si tensorboard_log está seteado.
                    ),
                    
                    ent_coef=0.001,
                    clip_range=0.2,
                    verbose=1)
        # dtype = torch.float16 # Cambiar a torch.float32 para 32 bits, torch.float16 para 16 bits
        # model.policy = model.policy.to(dtype=dtype)
        print(f"[INFO] Creando modelo en {filename}") #
        # Try to compile policy for faster forward (best-effort)
        try:
            if hasattr(torch, 'compile'):
                # Only compile if Triton is available when using CUDA to avoid runtime failures
                try:
                    triton_available = False
                    if torch.cuda.is_available():
                        try:
                            import triton  # type: ignore
                            triton_available = True
                        except Exception:
                            triton_available = False
                    else:
                        # CPU-only compile may not need Triton
                        triton_available = True
                except Exception:
                    triton_available = False

                if torch.cuda.is_available() and not triton_available:
                    print('[WARN] Triton not available; skipping torch.compile to avoid runtime errors.')
                else:
                    try:
                        model.policy = torch.compile(model.policy)
                    except Exception as e:
                        print(f"[WARN] torch.compile failed at compile time: {e}")
        except Exception as e:
            print(f"[WARN] torch.compile block failed: {e}")
        # Enable mixed precision forward (AMP) for CUDA if available
        try:
            if DEVICE == 'cuda':
                from torch.cuda.amp import autocast
                orig_forward = model.policy.forward
                def _amp_forward(*args, **kwargs):
                    with autocast(enabled=True):
                        return orig_forward(*args, **kwargs)
                model.policy.forward = _amp_forward
                # Enable TF32 where available
                try:
                    torch.backends.cudnn.allow_tf32 = True
                    torch.backends.cuda.matmul.allow_tf32 = True
                except Exception:
                    pass
                print("[INFO] AMP enabled for policy forward (inference/training autocast)")
        except Exception as e:
            print(f"[WARN] enabling AMP wrapper failed: {e}")
        
    #### Target cumulative rewards (problem-dependent) ##########
    if DEFAULT_ACT == ActionType.ONE_D_RPM:
        target_reward = 474.15 if not multiagent else 949.5
    else:
        
        target_reward = 30500 if not multiagent else 920.
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
        tb_callback = TensorboardCallback(tb_log_dir=filename+'/tb/', log_freq=2000)
        if use_render_callback:
            train_render_callback = TrainRenderCallback(train_env, sync_human_speed=False)
            model.learn(total_timesteps=int(3e7) if local else int(1e2),
                        callback=[eval_callback, train_render_callback ],
                        log_interval=100,
                        reset_num_timesteps=False if continue_from else True)
        else:
            model.learn(total_timesteps=int(3e7) if local else int(1e2),
                        callback=[eval_callback],
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
        model = PPO.load(path, device=DEVICE)

    #### Show (and record a video of) the model's performance ##
    if not multiagent:
        test_env = HoverAviary(gui=True,
                               obs=DEFAULT_OBS,
                               act=DEFAULT_ACT,
                               record=record_video,
                               initial_xyzs=np.array([[0,0,2]]),
                               initial_rpys=np.array([[0,0,0]]),
                               random_targets=RANDOM_TARGETS, physics=Physics.PYB)
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
                                            obs2[3:13],
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
                                                obs2[d][3:13],
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
