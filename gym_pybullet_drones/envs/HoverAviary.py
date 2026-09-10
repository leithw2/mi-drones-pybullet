import numpy as np
import pybullet as p
import math
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType
import matplotlib.pyplot as plt

class HoverAviary(BaseRLAviary):
    
    
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_xyzs=np.array([[0,0,1]]),
                 #initial_xyzs=np.array([[np.random.uniform(-0.5, 0.5),np.random.uniform(-0.5, 0.5), np.random.uniform(1, 1.5)]]),
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 60,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM,
                 random_targets: bool = False
                 ):
        """Initialization of a single agent RL environment.

        Using the generic single agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

        """
        self.random_targets = random_targets
        self.one_only_target = False
        self.TARGET_POS = np.array([9,9,2])
        #print("Target position: " + str(self.TARGET_POS))
        self.EPISODE_LEN_SEC = 30
        self._prev_dist = None  # Initialize the best distance to None
        self.step_count = 0
        self.score = 1
        self.actual_reward = 0
        self.time_penalty = 0
        self.TEST_BODY = None
        self.TEST_BODY_ID = None
        self.truncate_early = False
        self.point_track = None
        self.random_value = 1.6

            
        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act
                         )
    
        
        if self.GUI:
            self.fig, self.ax = plt.subplots(figsize=(6, 6))
            
            self.heatmap = self.ax.imshow(
                np.zeros((8, 8)),
                origin="lower",
                cmap="hot",
                interpolation="nearest",
                vmin=0,
                vmax=1
            )
            self.ax.invert_xaxis()
            self.fig.colorbar(self.heatmap, ax=self.ax)
            plt.ion()  # modo interactivo
            plt.show(block=False)
            
        self.prev_action = None

        # Fase inicial: vuelo conservador
        self.max_action_delta = 0.05
        self.smooth_lambda = 0.20
    
    ################################################################################
    def _draw_target_marker(self, color=[0, 1, 0]):
        if not self.GUI:
            return

        # Eliminar solamente el marcador anterior
        if hasattr(self, '_target_marker_id'):
            p.removeUserDebugItem(self._target_marker_id)

        self._target_marker_id = p.addUserDebugLine(
            self.TARGET_POS,
            [0, 0, 0],
            color,
            lineWidth=1,
            lifeTime=0
        )

        # Punto del objetivo actual
        self._target_point_id = p.addUserDebugPoints(
            pointPositions=self.TARGET_POS.reshape(1, 3),
            pointColorsRGB=[color],
            pointSize=10,
            lifeTime=0
        )
    
    def _draw_trajectory(self, trajectory):
        if not self.GUI or len(trajectory) < 2:
            return

        self._trajectory_ids = []

        for i in range(len(trajectory) - 1):
            line_id = p.addUserDebugLine(
                trajectory[i],
                trajectory[i + 1],
                [1, 0, 0],
                lineWidth=2,
                lifeTime=0
            )

            self._trajectory_ids.append(line_id)
            
    def reset(self, *args, **kwargs):
        self.score = 1
        self.time_penalty = 0
        self.prev_action = None
        
        # Habilitar orientación inicial aleatoria en Yaw
        if self.INIT_RPYS is not None:
            self.INIT_RPYS[0, 2] = np.random.uniform(-np.pi, np.pi)
            self.INIT_RPYS[0, 2] = np.random.uniform(0,0)
            

        obs, info = super().reset(*args, **kwargs)
        
        self.TEST_BODY = p.createCollisionShape(p.GEOM_SPHERE, radius=.6, physicsClientId=self.CLIENT)
        self.TEST_BODY_ID = p.createMultiBody(baseMass=0, 
                                             baseCollisionShapeIndex=self.TEST_BODY, 
                                             basePosition=[0, 0, -10],
                                             physicsClientId=self.CLIENT)

        # -------------------------------------------------------------------------
        # BENCHMARKS TRAYECTORIAS CANÓNICAS DE LA LITERATURA DE DRONES
        # -------------------------------------------------------------------------
        # Paramétrización general: p va de 0.0 a 1.0

        Amplitud = np.random.uniform(6,6)
        # print("Amplitud ", Amplitud)
        # 1. Figura en 8 (Lemniscata de Gerono - Estándar Agilicious / Mellinger)
        lemniscata_8 = lambda p: np.round(np.array([
            Amplitud * math.sin(6 * math.pi * p),                  # X: Amplitud 2m
            Amplitud * math.sin(12 * math.pi * p) / 2.0,            # Y: Doble frecuencia para el cruce
            1.2 + 0.4 * math.cos(2 * math.pi * p)             # Z: Oscilación suave de altura
        ]), 2)
        
        # 1. Figura en 8 (Lemniscata de Gerono - Estándar Agilicious / Mellinger)
        lemniscata_8_inv = lambda p: np.round(np.array([
            -Amplitud * math.sin(6 * math.pi * p),                  # X: Amplitud 2m
            -Amplitud * math.sin(12 * math.pi * p) / 2.0,            # Y: Doble frecuencia para el cruce
            1.2 + 0.4 * math.cos(2 * math.pi * p)             # Z: Oscilación suave de altura
        ]), 2)

        # 2. Curva de Lissajous 3D (Acoplamiento triaxial complejo)
        lissajous_3d = lambda p: np.round(np.array([
            4.0 * math.sin(3 * 2 * math.pi * p),              # X: Frecuencia fx = 3
            4.0 * math.cos(2 * 2 * math.pi * p),              # Y: Frecuencia fy = 2
            1.8 + 0.5 * math.sin(4 * 2 * math.pi * p)         # Z: Frecuencia fz = 4
        ]), 2)

        # 3. Espirograma 3D / Epitrocoide (Cambios rápidos de curvatura y $g$-forces)
        R_out, r_in, d_val = 4.0, 2.5, 0.7
        spirograph_3d = lambda p: np.round(np.array([
            (R_out - r_in) * math.cos(2 * math.pi * p) + d_val * math.cos((R_out - r_in) / r_in * 2 * math.pi * p),
            (R_out - r_in) * math.sin(2 * math.pi * p) - d_val * math.sin((R_out - r_in) / r_in * 2 * math.pi * p),
            2.0 + 0.4 * math.sin(6 * math.pi * p)
        ]), 2)

        # 4. Hélice Ascendente Determinista (Radio y paso constantes)
        helice_ascendente = lambda p: np.round(np.array([
            2.5 * math.cos(4 * math.pi * p),                  # X: Radio 1.5m, 2 vueltas completas
            2.5 * math.sin(4 * math.pi * p),                  # Y
            0.5 + 1.5 * p                                     # Z: Ascenso continuo de 0.5m a 2.0m
        ]), 2)

        # 5. Respuesta a Escalón Poligonal (Waypoints tipo Cuadrado Zig-Zag)
        def waypoints_square(p):
            # Divide p [0, 1] en 4 segmentos rectos
            if p < 0.25:
                t = p / 0.25
                return np.array([2.5 * t, 0.0, 1.0])
            elif p < 0.50:
                t = (p - 0.25) / 0.25
                return np.array([2.5, 2.5 * t, 1.0])
            elif p < 0.75:
                t = (p - 0.50) / 0.25
                return np.array([2.5 * (1 - t), 2.5, 1.0])
            else:
                t = (p - 0.75) / 0.25
                return np.array([0.0, 2.5 * (1 - t), 1.0])

        # -------------------------------------------------------------------------
        
        if self.random_targets:
            self.TARGET_POS = np.array([
                np.random.uniform(-self.random_value, self.random_value),
                np.random.uniform(-self.random_value, self.random_value),
                np.random.uniform(1.0, 1.0)
            ])
            r = 0.8
            for i in range(500):
                if self._is_space_clear(self.TARGET_POS, radius=r):
                    self._draw_target_marker([0, 1, 0])
                    break      
                else:
                    self.TARGET_POS = np.array([
                        np.random.uniform(-self.random_value, self.random_value),
                        np.random.uniform(-self.random_value, self.random_value),
                        np.random.uniform(1.0, 1.0)
                    ])    

        elif not self.one_only_target and not self.random_targets:
            # Lista de trayectorias benchmark disponibles
            # benchmarks = [lemniscata_8, lissajous_3d, spirograph_3d, helice_ascendente, waypoints_square]
            benchmarks = [lemniscata_8]
            
            # Selección aleatoria o manual del test (0: Lemniscata, 1: Lissajous, 2: Spirograph, 3: Hélice, 4: Cuadrado)
            self.task_idx = np.random.choice(len(benchmarks))
            
            # print("Direction ", self.task_idx )
            # Discretización razonable para dinámicas de quadcopter (entre 60 y 100 pasos por trayecto)
            self.pasos = np.random.randint(40, 60)
            self.pasos = 60
            self.point_track = self.generar_trayectoria(
                                                        benchmarks[self.task_idx],
                                                        pasos=self.pasos
                                                        )

            # print(self.point_track)
            # Dibujar TODA la trayectoria
            self._draw_trajectory(self.point_track)

            # Primer punto como objetivo actual
            self.TARGET_POS = self.point_track.pop(0)

            self._draw_target_marker([0, 1, 0]) 
           
            print(f"Benchmark Activo: ID {self.task_idx} | Puntos Restantes: {len(self.point_track)}")

        elif self.one_only_target and not self.random_targets:
            self.TARGET_POS = np.array([0.0, 0.0, 1.0])
            self.pasos = 0
            self._draw_target_marker([0, 1, 0])

        self._prev_dist = None
        self.truncate_early = False
        self.prev_action = None

        
        return self._computeObs(), info

    ################################################################################
    
    def _is_space_clear(self, pos, radius=0.4, ignore_ids=[]):
        # 1. Límites del área de vuelo
        #if pos[0] < -1 or pos[1] < -1:
        #    return False
        
        # --- NUEVO: Chequeo de Volumen (Detecta si está ADENTRO de un sólido) ---
        # Creamos una caja pequeña alrededor del punto
        aabb_min = [pos[0] - 0.05, pos[1] - 0.05, pos[2] - 0.05]
        aabb_max = [pos[0] + 0.05, pos[1] + 0.05, pos[2] + 0.05]
        overlapping = p.getOverlappingObjects(aabb_min, aabb_max, physicsClientId=self.CLIENT)
        
        if overlapping:
            for obj in overlapping:
                obj_id = obj[0]
                # Ignoramos suelo (generalmente ID 0), el dron y los IDs en ignore_ids
                if obj_id != 0 and obj_id not in self.DRONE_IDS and obj_id not in ignore_ids:
                    # Si hay algo aquí, el punto está ADENTRO de un sólido
                    return False
        # -----------------------------------------------------------------------

        # 2. Chequeo de Rayos (Detecta si hay paredes CERCA)
        directions = [
            [radius, 0, 0], [-radius, 0, 0],
            [0, radius, 0], [0, -radius, 0],
            [0, 0, radius], [0, 0, -radius]
        ]
        
        for d in directions:
            end_point = [pos[0] + d[0], pos[1] + d[1], pos[2] + d[2]]
            
            # El rayo devuelve una lista.
            ray_result = p.rayTest(pos, end_point, physicsClientId=self.CLIENT)[0]
            hit_id = ray_result[0]
            
            if hit_id != -1 and hit_id not in ignore_ids and hit_id not in self.DRONE_IDS:
                # DEBUG: Dibujar el rayo que chocó en rojo
                p.addUserDebugLine(pos, end_point, [1, 0, 0], lifeTime=0.5)
                return False
                
        return True
    
    def _computeReward(self):
        """Calcula el valor de la recompensa actual acotada y estable."""
        self.truncate_early = False
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        vel = state[10:13]          # vx, vy, vz
        angles = state[7:10]        # roll, pitch, yaw
        angle_vel = state[13:16]    # roll_rate, pitch_rate, yaw_rate
        
        delta_pos = self.TARGET_POS - pos
        dist = np.linalg.norm(delta_pos)
        
        

        if self._prev_dist is None:
            self._prev_dist = dist
            
        action_smooth_penalty = 0.0

        if self.prev_action is not None:
            action_change = self.action - self.prev_action
            action_smooth_penalty = -0.20 * np.sum(np.square(action_change))

        self.prev_action = self.action.copy()

        # 1. Recompensa Continua por Cercanía (Sintronizada)
        base_reward = np.exp(-1.5 * dist)  # Rango [0.0, 1.0] suave en lugar de cuadrático
        
        # 2. Recompensa de Progreso Continuo (Potencial)
        # Premia reducir la distancia real en vez de usar valores discretos (-2.0 / +1.5)
        progress_reward = 2.0 * (self._prev_dist - dist)
        self._prev_dist = dist

        

        # 4. Estabilización de Actitud y Velocidades Angulares
        angle_penalty = -0.05 * (abs(angles[0]) + abs(angles[1]))
        angle_vel_penalty = -0.03 * np.sum(np.square(angle_vel)) # Penaliza oscilaciones cuadráticas (temblor)

        # 5. Penalización de Tiempo Normalizada
        self.time_penalty = - 0.005

        # 6. Evaluación de Cumplimiento de Objetivo y Cambio de Target
        bonus = 0.0
        # Guardas la lectura actual antes de actualizarla con la nueva
        # ============================================================
        # LIDAR
        # self.lidar = [d0_0 ... d0_63, d1_0 ... d1_63]
        # ============================================================
        lidar = self.lidar.flatten()
        # Separar d0 y d1
        self.lidar_d0 = lidar[0:64]
        self.lidar_d1 = lidar[64:128]
        
        
        # print("self.lidar shape:", self.lidar.shape)
        # print("self.lidar_d0 shape:", self.lidar_d0.shape)
        # print("self.lidar_d1 shape:", self.lidar_d1.shape)
        # ============================================================
        # VISUALIZACIÓN 8x8
        # ============================================================

        
        if self.GUI:
            grid = self.lidar_d0.reshape(8, 8)
            
            self.heatmap.set_data(grid)
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

        # ============================================================
        # PENALIZACIÓN
        # ============================================================

        max_d0 = 0.0
        penaltyLidar = 0.0

        if self.lidar_d0 is not None and self.lidar_d1 is not None:

            # ============================================================
            # MATRICES 8x8
            # ============================================================

            d0_t0 = self.lidar_d0.reshape(8, 8)
            d1_t1 = self.lidar_d1.reshape(8, 8)

            # Máxima proximidad actual
            max_d0 = float(np.max(d0_t0))

            # ============================================================
            # RIESGO POR PROXIMIDAD
            # ============================================================

            threshold = 0.50

            penalty_dist = 0.0

            if max_d0 > threshold:

                risk = (max_d0 - threshold) / (1.0 - threshold)

                penalty_dist = -1.5 * (risk ** 2)

            # ============================================================
            # RIESGO POR APROXIMACIÓN
            # ============================================================

            delta = d0_t0 - d1_t1

            # Suavizado espacial 3x3
            padded = np.pad(delta, 1, mode='edge')

            delta_smooth = np.zeros_like(delta)

            for i in range(8):
                for j in range(8):
                    delta_smooth[i, j] = np.mean(
                        padded[i:i+3, j:j+3]
                    )

            max_delta = float(np.max(delta_smooth))

            penalty_approach = 0.0

            if max_delta > 0.2:
                penalty_approach = -0.3 * max_delta

            # ============================================================
            # PENALIZACIÓN TOTAL LiDAR
            # ============================================================

            penaltyLidar = float(
                np.clip(
                    penalty_dist + penalty_approach,
                    -3.0,
                    0.0
                )
            )


        # ============================================================
        # VELOCIDAD CERCA DE OBSTÁCULOS
        # ============================================================

        speed = np.linalg.norm(vel)

        obstacle_threshold = 0.15

        obstacle_risk = np.clip(
            (max_d0 - obstacle_threshold) /
            (1.0 - obstacle_threshold),
            0.0,
            1.0
        )

        safe_speed = 0.5

        obstacle_speed_penalty = (
            -2.0
            * obstacle_risk
            * (speed / safe_speed) ** 2
        )

        obstacle_speed_penalty = float(
            np.clip(
                obstacle_speed_penalty,
                -4.0,
                0.0
            )
        )
        # print("speed ", speed)
        # print("obstacle_risk ", obstacle_risk)
        # print("obstacle_speed_penalty ", obstacle_speed_penalty)
        # print("penaltyLidar ", penaltyLidar)

        # 3. Alineación de Yaw (Normalizada)
        quat = state[3:7].copy()
        if quat[3] < 0:
            quat = -quat
            
        heading_alignment_reward = 0.0
        radial_velocity = 0.0
        velocity_toward_target_reward = 0.0

        if dist > 0.15:

            target_dir = delta_pos / (dist + 1e-8)

            # Positivo  -> se acerca al objetivo
            # Cero      -> movimiento tangencial
            # Negativo  -> se aleja
            radial_velocity = np.dot(vel, target_dir)

            # Recompensamos únicamente el movimiento hacia el objetivo.
            # El progress_reward ya penaliza alejarse.
            velocity_toward_target_reward = (
                0.3
                * max(radial_velocity, 0.0)
                * (1.0 - obstacle_risk)
            )
        delta_xy = delta_pos[:2]
        dist_xy = np.linalg.norm(delta_xy)

        if dist_xy > 0.15:

            target_dir_xy = delta_xy / dist_xy

            rotation_matrix = np.asarray(
                p.getMatrixFromQuaternion(quat)
            ).reshape(3, 3)

            forward_xy = rotation_matrix[:2, 0]

            forward_norm = np.linalg.norm(forward_xy)

            if forward_norm > 1e-8:

                forward_xy /= forward_norm

                alignment_cos = np.dot(
                    forward_xy,
                    target_dir_xy
                )

                # Recompensa de heading condicionada al movimiento hacia el objetivo                
                velocity_factor = np.clip(
                    radial_velocity / 0.5,
                    0.0,
                    1.0
                )

                heading_alignment_reward = (
                    0.5
                    * alignment_cos
                    * velocity_factor
                    * (1.0 - obstacle_risk)
                )
        
        if self.random_targets:
            self.pasos = 15
            if dist < 0.6 and np.linalg.norm(vel) < 0.6:
                bonus = 10.0  # BONUS FIJO (sin escalar multiplicativamente por self.score)
                self.score += 1
                
                r = 0.8
                self.TARGET_POS = np.array([
                    np.random.uniform(-self.random_value, self.random_value),
                    np.random.uniform(-self.random_value, self.random_value),
                    np.random.uniform(0.5, 2.0)
                ])
                for i in range(100):
                    if self._is_space_clear(self.TARGET_POS, radius=r):
                        self._draw_target_marker([0, 0, 1])
                        break
                    else:
                        self.TARGET_POS = np.array([
                            np.random.uniform(-self.random_value, self.random_value),
                            np.random.uniform(-self.random_value, self.random_value),
                            np.random.uniform(0.5, 2.0)
                        ])
                        if i >= 99:
                            self.truncate_early = True
                
                self._prev_dist = None

        elif self.one_only_target:
            if dist < 0.6 and np.linalg.norm(vel) < 0.4:
                bonus = 5.0
                self.score += 1
                self.time_penalty = 0.0

        else:
            # MODO SEGUIMIENTO DE TRAYECTORIA (Lemniscata)
            if dist < 0.8 and np.linalg.norm(vel) < 1.8:
                bonus = 15.0  # BONUS FIJO
                self.score += 1
                self._draw_target_marker([0, 1, 0])
                
                self.TARGET_POS = self.point_track.pop(0)

                    
                self._prev_dist = None

        # Finalización por Puntuación Máxima
        if self.score == self.pasos-1 and (self.random_targets or not self.one_only_target):
            self.truncate_early = True
            bonus += 20.0  # Bonus de finalización acotado

        # Suma Total
        total_reward = (
            base_reward 
            + progress_reward 
            + heading_alignment_reward 
            + angle_penalty 
            + angle_vel_penalty 
            + self.time_penalty 
            + action_smooth_penalty
            + obstacle_speed_penalty
            + velocity_toward_target_reward
            + bonus
            + penaltyLidar
        )

        self.actual_reward += total_reward
        
        # if total_reward < -2.0:

        # print(
        #     f"""
        #     REWARD STEP NEGATIVO
        #     total       = {total_reward:.3f}
            
        #     obs_speed   = {obstacle_speed_penalty:.3f}
        #     lidar       = {penaltyLidar:.3f}
        #     velocity_toward_target_reward = {velocity_toward_target_reward:.3f}
        #     speed       = {speed:.3f}
        #     risk        = {obstacle_risk:.3f}
        #     distance    = {dist:.3f}
        #     base        = {base_reward:.3f}
        #     progress    = {progress_reward:.3f}
        #     heading     = {heading_alignment_reward:.3f}
        #     angle       = {angle_penalty:.3f}
        #     angle_vel   = {angle_vel_penalty:.3f}
        #     smooth      = {action_smooth_penalty:.3f}
        #     """
        #         )
        
        # if dist < 0.8:
        #     print(
        #         f"dist={dist:.3f} "
        #         f"speed={speed:.3f} "
        #         f"radial_v={radial_velocity:.3f} "
        #         f"base={base_reward:.3f} "
        #         f"progress={progress_reward:.3f} "
        #         f"heading={heading_alignment_reward:.3f} "
        #         f"bonus={bonus:.3f}"
        #     )
        return total_reward
        
    import math

    # 1. Definimos la FUNCIÓN que genera la lista
    def generar_trayectoria(self, formula_figura, pasos=100):
        lista_puntos = []
        for i in range(pasos + 1):
            # Aquí es donde PREPARAMOS el valor de p (de 0.0 a 1.0)
            p = (i+1) / pasos
            
            punto_base = formula_figura(p)
            punto = punto_base

            # Mantiene la posición dentro de la misma figura, pero evita
            # aceptar objetivos que estén dentro o demasiado cerca de un obstáculo.
            for _ in range(500):
                if self._is_space_clear(punto, radius=0.6):
                    break
                # Desplaza el candidato en una dirección aleatoria para salir
                # del obstáculo sin cambiar el orden de la trayectoria.
                punto = punto_base + np.random.uniform(
                    low=[-0.6, -0.6, -0.1],
                    high=[0.6, 0.6, 0.1]
                )
            else:
                raise RuntimeError(
                    f"No se encontró un punto libre para p={p:.3f} "
                    "después de 500 intentos"
                )
            
            lista_puntos.append(punto)
        return lista_puntos


    
    ################################################################################
    
    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        #print(self.time_penalty)
        
        penalty = 200 / self.score # Penalización que disminuye a medida que se alcanzan más objetivos
        state = self._getDroneStateVector(0)
        vel = state[10:13]
        
        # if self.time_penalty < -.08:
        #     print("static Truncated - reward: "  + str(self.actual_reward-penalty))
        #     print('score', self.score)
        #     self.actual_reward = 0 

        #     return True, penalty
        
        if self.truncate_early:
            print("Early Truncated - reward: "  + str(self.actual_reward))
            print('score', self.score)
            self.actual_reward = 0
            self.truncate_early = False
            return True, penalty
        
        if (abs(state[0]) > 20 or abs(state[1]) > 20 or state[2] > 80 # Truncate when the drone is too far away
        ):
            #print(  f"Truncated far away: pos {state[0:3]}, angles {state[7:10]}")
            print("far away - reward: "  + str(self.actual_reward - penalty))
            print('score', self.score)
            self.actual_reward = 0
            return True, penalty
        
        if (abs(state[7]) > 1.3 or abs(state[8]) > 1.3):

            print(
                "tilted - accumulated reward: "
                + str(self.actual_reward)
            )

            print("score", self.score)

            self.actual_reward = 0

            return True, penalty
        
        if state[2] < 0.02:
            #print(  f"Truncated height: pos {state[0:3]}, angles {state[7:10]}")
            print("height limit - reward: "  + str(self.actual_reward - penalty))
            print('score', self.score)
            self.actual_reward = 0
            return True, penalty
        

        
        if self.lidar is not None and np.max(self.lidar) > 0.95:

            print(
                "collision special!!!! - accumulated reward: "
                + str(self.actual_reward)
            )

            print("score", self.score)

            self.actual_reward = 0

            return True, penalty

        if self.obstacle_collision:

            print(
                "obstacle collision - accumulated reward: "
                + str(self.actual_reward)
            )

            print("score", self.score)

            self.actual_reward = 0

            return True, penalty
        
        if np.linalg.norm(self.TARGET_POS-state[0:3]) < .001:
            #print("target reached - reward: "  + str(self.actual_reward - penalty))
            #print('score', self.score)
            self.actual_reward = 0
            return False, 0
        else:
            return False, 0
    ################################################################################
    
    def _computeTruncated(self):
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the current episode timed out.

        """
        


            
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC*3:
            print("Time Truncated - reward: "  + str(self.actual_reward))
            self.actual_reward = 0
            print('score', self.score)
            self.step_counter = 0

            return True
        else:
            return False


    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years
