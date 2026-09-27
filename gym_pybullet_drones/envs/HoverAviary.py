import numpy as np
import pybullet as p
import pygame as pg
import math
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType
import matplotlib.pyplot as plt

class HoverAviary(BaseRLAviary):
    
    
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X250,
                 initial_xyzs=np.array([[0, 0, np.random.uniform(0.8, 2.5)]]),
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 60,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM,
                 random_targets: bool = False,
                 randomized = False,
                 rabbit_mode = "keyboard" # "script" , "keyboard"
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
        
        pg.init()
        pg.joystick.init()
        pg.display.init()
        
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
        self.randomized = randomized
        self.rabbit_mode = rabbit_mode

        self.rabbit_center = np.array([0.0, 0.0, 1.5])
        self.rabbit_target = self.rabbit_center.copy()

        self.rabbit_radius = 5.0
        self.rabbit_omega = 0.25
        self.rabbit_speed = 1.0

        self.rabbit_time = 0.0

        self.rabbit_joystick = None
        
        if self.randomized :
            initial_xyzs = np.array([[np.random.uniform(-0.8, 0.8),np.random.uniform(-0.8, 0.8), np.random.uniform(0.8, 1.5)]])


            
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
                         act=act,
                         randomized = randomized
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
        self.lidar_risk_memory = 0.0

        # Fase inicial: vuelo conservador
        self.max_action_delta = 0.05
        self.smooth_lambda = 0.20
    
    ################################################################################
    def _draw_target_marker(self, color=[0, 1, 0]):
        if not self.GUI:
            return

        # 1. Eliminar el punto anterior manualmente si existe
        if hasattr(self, '_target_point_id') and self._target_point_id is not None:
            p.removeUserDebugItem(self._target_point_id)

        dt = 0.1

        # La línea sí respeta lifeTime
        p.addUserDebugLine(
            self.TARGET_POS,
            [0, 0, 0],
            color,
            lineWidth=1,
            lifeTime=dt
        )

        # El punto se crea con lifeTime=0 y guardamos su ID para borrarlo en el siguiente paso
        self._target_point_id = p.addUserDebugPoints(
            pointPositions=self.TARGET_POS.reshape(1, 3),
            pointColorsRGB=[color],
            pointSize=10,
            lifeTime=dt
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
        self._prev_ang_vel = None
        self._probe_id = None
        self._probe_shape = None
        if self.randomized :
            self.INIT_XYZS = np.array([[np.random.uniform(-0.8, 0.8),np.random.uniform(-0.8, 0.8), np.random.uniform(0.8, 1.5)]])

        # Habilitar orientación inicial aleatoria en Yaw
        if self.INIT_RPYS is not None :
            self.INIT_RPYS[0, 2] = np.random.uniform(-np.pi, np.pi)
            #self.INIT_RPYS[0, 2] = np.random.uniform(0,0)
            

        obs, info = super().reset(*args, **kwargs)
        
        self.TEST_BODY = p.createCollisionShape(p.GEOM_SPHERE, radius=.6, physicsClientId=self.CLIENT)
        self.TEST_BODY_ID = p.createMultiBody(baseMass=0, 
                                             baseCollisionShapeIndex=self.TEST_BODY, 
                                             basePosition=[0, 0, -10],
                                             physicsClientId=self.CLIENT)
        
        if self.rabbit_mode is not None:
            # Crear una esfera visual única que actuará como marcador objetivo
            if self.GUI :
                visual_shape_id = p.createVisualShape(
                    shapeType=p.GEOM_SPHERE,
                    radius=0.15,
                    rgbaColor=[0, 1, 0, 0.8]  # Verde semi-transparente
                )
                # createMultiBody con mass=0 crea un objeto estático sin colisión
                self._target_visual_id = p.createMultiBody(
                    baseMass=0,
                    baseVisualShapeIndex=visual_shape_id,
                    basePosition=self.TARGET_POS
                )
                
                
        benchmark = True
        if benchmark:
            # -------------------------------------------------------------------------
            # BENCHMARKS TRAYECTORIAS CANÓNICAS DE LA LITERATURA DE DRONES
            # -------------------------------------------------------------------------
            # Paramétrización general: p va de 0.0 a 1.0
            if self.randomized:
                Amplitudx = np.random.uniform(4,9)
                Amplitudy = np.random.uniform(4,9)
            else : 
                Amplitudx = 6
                Amplitudy = 6
            # print("Amplitud ", Amplitud)
            # 1. Figura en 8 (Lemniscata de Gerono - Estándar Agilicious / Mellinger)
            lemniscata_8 = lambda p: np.round(np.array([
                Amplitudx * math.sin(6 * math.pi * p),                  # X: Amplitud 2m
                Amplitudy * math.sin(12 * math.pi * p) / 2.0,            # Y: Doble frecuencia para el cruce
                1.6 + 0.8 * math.cos(2 * math.pi * p)             # Z: Oscilación suave de altura
            ]), 2)
            
            # 1. Figura en 8 (Lemniscata de Gerono - Estándar Agilicious / Mellinger)
            lemniscata_8_inv = lambda p: np.round(np.array([
                -Amplitudx * math.sin(6 * math.pi * p),                  # X: Amplitud 2m
                -Amplitudy * math.sin(12 * math.pi * p) / 2.0,            # Y: Doble frecuencia para el cruce
                1.6 + 0.8 * math.cos(2 * math.pi * p)             # Z: Oscilación suave de altura
            ]), 2)

            # 2. Curva de Lissajous 3D (Acoplamiento triaxial complejo)
            lissajous_3d = lambda p: np.round(np.array([
                4.0 * math.sin(6 * 2 * math.pi * p),              # X: Frecuencia fx = 3
                4.0 * math.cos(4 * 2 * math.pi * p),              # Y: Frecuencia fy = 2
                2.2 + 0.9 * math.sin(4 * 2 * math.pi * p)         # Z: Frecuencia fz = 4
            ]), 2)

            # 3. Espirograma 3D / Epitrocoide de recorrido amplio y mayor frecuencia.
            # El radio efectivo aumenta y la curva completa dos ciclos principales.
            R_out, r_in, d_val = 6.0, 1.0, 3.0
            spiro_frequency = 1.0
            spirograph_3d = lambda p: np.round(np.array([
                (R_out - r_in) * math.cos(spiro_frequency * 2 * math.pi * p) + d_val * math.cos((R_out - r_in) / r_in * spiro_frequency * 2 * math.pi * p),
                (R_out - r_in) * math.sin(spiro_frequency * 2 * math.pi * p) - d_val * math.sin((R_out - r_in) / r_in * spiro_frequency * 2 * math.pi * p),
                2.5 + 0.8 * math.sin(6 * math.pi * p)
            ]), 2)

            # 4. Hélice Ascendente Determinista (Radio y paso constantes)
            helice_ascendente = lambda p: np.round(np.array([
                3.0 * math.cos(8 * math.pi * p),                  # X: Radio 1.5m, 2 vueltas completas
                3.0 * math.sin(8 * math.pi * p),                  # Y
                1.5 + 2.5 * p                                     # Z: Ascenso continuo de 0.5m a 2.0m
            ]), 2)

            # 5. Respuesta a Escalón Poligonal (Waypoints tipo Cuadrado Zig-Zag)
            def waypoints_square(p):
                # Divide p [0, 1] en 4 segmentos rectos
                if p < 0.25:
                    t = p / 0.5
                    return np.array([2.5 * t, 0.0, 1.5])
                elif p < 0.50:
                    t = (p - 0.25) / 0.25
                    return np.array([2.5, 2.5 * t, 1.2])
                elif p < 0.75:
                    t = (p - 0.50) / 0.25
                    return np.array([2.5 * (1 - t), 2.5, 1.0])
                else:
                    t = (p - 0.75) / 0.25
                    return np.array([0.0, 2.5 * (1 - t), 1.5])

            map_route = np.array([
                [0.0, 1.0, 1.2],

                [0.0, 12.0, 1.8],
                [0.0, 14.0, 1.2],
                [2.0, 16.0, 1.8],
                [6.0, 16.0, 1.8],
                [4.0, 20.0, 1.8],
                [4.0, 20.0, 1.8],
                [0.0, 26.0, 1.8],
                [10.0, 27.0, 1.8],

            ])
        
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
                benchmarks = [lemniscata_8, lemniscata_8_inv, lissajous_3d, spirograph_3d, helice_ascendente]
                benchmark_names = ["Lemniscata 8", "Lemniscata 8 Invertida", "Lissajous 3D", "Spirograph 3D", "Helice ascendente", "Mapa por waypoints"]
                #benchmarks = [lemniscata_8, lemniscata_8_inv]
                #benchmark_names = ["Lemniscata 8", "Lemniscata 8 Invertida"]
                
                # Selección aleatoria o manual del test (0: Lemniscata, 1: Lissajous, 2: Spirograph, 3: Hélice, 4: Cuadrado)
                self.task_idx = np.random.choice(len(benchmarks))
                print("Benchmark seleccionado: ", benchmark_names[self.task_idx])
                
                # print("Direction ", self.task_idx )
                # Menos puntos para que los waypoints consecutivos queden más separados.
                if self.randomized:
                    self.pasos = np.random.randint(50,60)
                else : 
                    self.pasos = 50
                
                self.point_track = self.generar_trayectoria(
                                                            benchmarks[self.task_idx],
                                                            pasos=self.pasos
                                                            )
                self.pasos = len(self.point_track)

                # print(self.point_track)
                # Dibujar TODA la trayectoria
                self._draw_trajectory(self.point_track)

                # Primer punto como objetivo actual
                self.TARGET_POS = self.point_track.pop(0)

                #self._draw_target_marker([0, 1, 0]) 
            
                print(f"Benchmark Activo: ID {self.task_idx} | Puntos Restantes: {len(self.point_track)}")

            elif self.one_only_target and not self.random_targets:
                self.TARGET_POS = np.array([0.0, 0.0, 1.0])
                self.pasos = 0
                self._draw_target_marker([0, 1, 0])

        self._prev_dist = None
        self.truncate_early = False
        
        self.prev_action = None
        self.lidar_risk_memory = 0.0

        
        return self._computeObs(), info

    ################################################################################
    def map_waypoint_route(p):
        num_segments = len(map_route) - 1
        segment_length = 1.0 / num_segments
        segment_index = min(int(p / segment_length), num_segments - 1)
        t = (p - segment_index * segment_length) / segment_length
        return (1 - t) * map_route[segment_index] + t * map_route[segment_index + 1]

    # -------------------------------------------------------------------------
    def _is_space_clear(self, pos, radius=0.2, ignore_ids=[]):
        if getattr(self, "_probe_id", None) is None:
            self._probe_shape = p.createCollisionShape(
                p.GEOM_SPHERE,
                radius=radius,
                physicsClientId=self.CLIENT
            )
            self._probe_id = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=self._probe_shape,
                basePosition=[0, 0, -100],
                physicsClientId=self.CLIENT
            )

        p.resetBasePositionAndOrientation(
            self._probe_id,
            np.asarray(pos, dtype=float),
            [0, 0, 0, 1],
            physicsClientId=self.CLIENT
        )

        obstacle_ids = []
        if getattr(self, "map_id", None) is not None:
            obstacle_ids.append(self.map_id)
        obstacle_ids.extend(
            obstacle_id for obstacle_id in getattr(self, "cubo_id", [])
            if obstacle_id is not None
        )
        obstacle_ids.extend(
            obstacle_id for obstacle_id in getattr(self, "dona_ids", [])
            if obstacle_id is not None
        )
        obstacle_ids = [obstacle_id for obstacle_id in obstacle_ids if obstacle_id not in ignore_ids]

        # Detecta puntos dentro de paredes y puntos demasiado cercanos a ellas.
        for obstacle_id in obstacle_ids:
            if p.getClosestPoints(
                bodyA=self._probe_id,
                bodyB=int(obstacle_id),
                distance=0.0,
                physicsClientId=self.CLIENT
            ):
                return False

        # Margen adicional para mallas cóncavas o paredes delgadas.
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
    
    @staticmethod
    def _smooth_lidar_map(lidar_map):
        """Aplica un kernel 3x3 para suavizar la lectura 8x8 del LiDAR."""
        lidar_map = np.asarray(lidar_map, dtype=float).reshape(8, 8)
        kernel = np.array([[1, 2, 1],
                           [2, 4, 2],
                           [1, 2, 1]], dtype=float) / 16.0
        padded = np.pad(lidar_map, ((1, 1), (1, 1)), mode="edge")
        smoothed = np.zeros_like(lidar_map, dtype=float)
        for i in range(8):
            for j in range(8):
                smoothed[i, j] = float(np.sum(padded[i:i + 3, j:j + 3] * kernel))
        return smoothed

    def _get_obstacle_guidance(self, vel_local):
        """Return persistent obstacle risk and the safer lateral direction."""
        if self.lidar is None:
            return 0.0, 0.0

        current_lidar = np.asarray(
            self.lidar[0, :64],
            dtype=float
        ).reshape(8, 8)

        current_lidar = self._smooth_lidar_map(current_lidar)

        closest_risk = float(np.max(current_lidar))

        strongest_rays = np.partition(
            current_lidar.reshape(-1),
            -8
        )[-8:]

        # CAMBIO: antes se usaba directamente esta combinación como obstacle_risk
        raw_risk = float(np.clip(
            0.7 * closest_risk
            + 0.3 * np.mean(strongest_rays),
            0.0,
            1.0
        ))

        # CAMBIO: antes obstacle_risk = raw_risk
        # Ahora hay una zona muerta hasta ~1.2 m para un LiDAR de 4 m.
        # raw_risk = 0.70 equivale aproximadamente a 1.2 m.
        minimum_risk = 0.70

        obstacle_risk = float(np.clip(
            (raw_risk - minimum_risk)
            / (1.0 - minimum_risk),
            0.0,
            1.0
        ))

        left_risk = float(
            np.mean(current_lidar[:, :3])
        )

        right_risk = float(
            np.mean(current_lidar[:, 5:])
        )

        safer_side = float(
            np.sign(
                (1.0 - right_risk)
                - (1.0 - left_risk)
            )
        )

        self.lidar_risk_memory = max(
            obstacle_risk,
            0.92 * self.lidar_risk_memory
        )

        return self.lidar_risk_memory, safer_side

    def _computeReward(self):
        """Reward local orientado a seguir la trayectoria de forma suave y controlada."""

        self.truncate_early = False
        self._update_rabbit_target()
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        vel_global = state[10:13]
        ang_vel_global = state[13:16]
        quat = state[3:7]
        angles = state[7:10]

        rotation_matrix = np.asarray(
            p.getMatrixFromQuaternion(quat)
        ).reshape(3, 3)

        vel = rotation_matrix.T @ vel_global
        ang_vel = rotation_matrix.T @ ang_vel_global
        roll, pitch = angles[0], angles[1]

        delta_pos = self.TARGET_POS - pos
        dist = float(np.linalg.norm(delta_pos))

        if self._prev_dist is None:
            self._prev_dist = dist

        progress = self._prev_dist - dist
        self._prev_dist = dist

        target_dir = delta_pos / (dist + 1e-8)
        target_local = rotation_matrix.T @ target_dir

        target_xy = target_local[:2]
        target_xy_norm = np.linalg.norm(target_xy)

        heading_alignment = 1.0

        if target_xy_norm > 1e-8:
            target_xy = target_xy / target_xy_norm
            heading_alignment = float(
                np.dot(np.array([1.0, 0.0]), target_xy)
            )

        heading_alignment = float(
            np.clip(heading_alignment, -1.0, 1.0)
        )

        forward_progress = float(
            np.dot(vel[:2], target_xy)
        )

        forward_speed = float(
            max(vel[0], 0.0)
        )

        speed_magnitude = float(
            np.linalg.norm(vel_global)
        )

        # ============================================================
        # GUÍA DE OBSTÁCULOS
        # ============================================================

        obstacle_risk, safer_side = self._get_obstacle_guidance(vel)

        collision_penalty = -2.5 * obstacle_risk

        if obstacle_risk > 0.35:
            collision_penalty -= (
                4.0
                * (obstacle_risk - 0.35)
                / 0.65
            )

        if obstacle_risk > 0.75:
            collision_penalty -= (
                4.0
                * (obstacle_risk - 0.75)
                / 0.25
            )

        collision_penalty = float(
            np.clip(
                collision_penalty,
                -10.0,
                0.0
            )
        )

        braking_penalty = (
            -3.0
            * obstacle_risk
            * max(forward_speed - 0.35, 0.0)
        )

        safe_lateral_speed = safer_side * vel[1]

        avoidance_reward = (
            2.5
            * obstacle_risk
            * np.clip(
                safe_lateral_speed,
                0.0,
                1.0
            )
        )

        wrong_side_penalty = (
            -1.5
            * obstacle_risk
            * np.clip(
                -safe_lateral_speed,
                0.0,
                1.0
            )
        )

        # ============================================================
        # CONTROL DE VELOCIDAD MÁXIMA
        # ============================================================

        target_speed_limit = 1.2

        excess_speed = max(
            0.0,
            speed_magnitude - target_speed_limit
        )

        excess_speed_penalty = (
            -1.5
            * (excess_speed ** 2)
        )

        # ============================================================
        # TAREA PRINCIPAL
        # ============================================================

        base_reward = 0.0

        approach_reward = (
            5.0
            * max(progress, 0.0)
            * (1.0 - 0.65 * obstacle_risk)
        )

        no_approach_penalty = 0.0

        if progress <= 0.0 and dist > 0.8:
            no_approach_penalty -= (
                1.5
                * (
                    1.0
                    + min(dist, 8.0) / 8.0
                )
            )

        if forward_progress < 0.05 and dist > 1.0:
            no_approach_penalty -= 0.5

        if heading_alignment < 0.2 and dist > 1.0:
            no_approach_penalty -= 0.2

        heading_reward = (
            1.2
            * max(heading_alignment, 0.0)
            * max(0.0, 1.0 - dist / 8.0)
        )

        speed_reward = (
            0.4
            * max(forward_progress, 0.0)
            * max(0.0, 1.0 - dist / 6.0)
        )

        reverse_penalty = (
            -0.8
            * min(forward_progress, 0.0)
        )

        time_penalty = -0.045

        # ============================================================
        # INICIALIZACIÓN DE ESTADOS PREVIOS
        # ============================================================

        action_now = np.asarray(
            self.action,
            dtype=np.float32
        ).flatten()

        if (
            not hasattr(self, "_prev_action")
            or self._prev_action is None
            or self._prev_action.shape != action_now.shape
        ):
            self._prev_action = action_now.copy()

        if (
            not hasattr(self, "_prev_ang_vel")
            or self._prev_ang_vel is None
            or self._prev_ang_vel.shape != ang_vel.shape
        ):
            self._prev_ang_vel = ang_vel.copy()

        # ============================================================
        # 1. SUAVIDAD DE ACCIÓN
        # ============================================================

        action_delta = (
            action_now - self._prev_action
        )

        action_change_penalty = (
            -0.25
            * float(np.mean(action_delta ** 2))
        )

        action_change_penalty = float(
            np.clip(
                action_change_penalty,
                -0.6,
                0.0
            )
        )

        # ============================================================
        # 2. TEMBLORES / CAMBIOS BRUSCOS DE VELOCIDAD ANGULAR
        # ============================================================

        angular_delta = (
            ang_vel - self._prev_ang_vel
        )

        jitter_penalty = (
            -0.20
            * float(
                np.sum(angular_delta[:2] ** 2)
            )
        )

        jitter_penalty = float(
            np.clip(
                jitter_penalty,
                -0.8,
                0.0
            )
        )

        # ============================================================
        # 3. VELOCIDAD ANGULAR CONTINUA
        # ============================================================

        ang_vel_penalty = (
            -0.15
            * float(
                np.linalg.norm(ang_vel[:2])
            )
        )

        ang_vel_penalty = float(
            np.clip(
                ang_vel_penalty,
                -0.5,
                0.0
            )
        )

        # ============================================================
        # 4. MANTENER HORIZONTE
        # ============================================================

        horizon_limit = np.deg2rad(10.0)

        excess_roll = max(
            abs(roll) - horizon_limit,
            0.0
        )

        excess_pitch = max(
            abs(pitch) - horizon_limit,
            0.0
        )

        horizon_penalty = (
            -0.4
            * (
                excess_roll
                + excess_pitch
            )
        )

        horizon_penalty = float(
            np.clip(
                horizon_penalty,
                -0.5,
                0.0
            )
        )

        # ============================================================
        # 5. RECOMPENSA POR ESTABILIDAD
        # ============================================================

        attitude_error = np.sqrt(
            roll ** 2 + pitch ** 2
        )

        angular_activity = np.linalg.norm(
            ang_vel[:2]
        )

        smooth_flight_reward = (
            0.15
            * np.exp(
                -4.0 * attitude_error
            )
            * np.exp(
                -3.0 * angular_activity
            )
        )

        # Guardar estados para el siguiente paso
        self._prev_action = action_now.copy()
        self._prev_ang_vel = ang_vel.copy()

        # ============================================================
        # REWARD TOTAL
        # ============================================================

        total_reward = (
            base_reward
            + approach_reward
            + heading_reward
            + speed_reward
            + reverse_penalty
            + no_approach_penalty
            + jitter_penalty
            + ang_vel_penalty
            + excess_speed_penalty
            + smooth_flight_reward
            + action_change_penalty
            + collision_penalty
            + braking_penalty
            + avoidance_reward
            + wrong_side_penalty
            + time_penalty
            + horizon_penalty
        )

        # ============================================================
        # WAYPOINT
        # ============================================================

        bonus = 0.0

        if dist < 1.0 and speed_magnitude < 2.0:

            bonus = 20.0
            self.score += 1

            if (
                not self.one_only_target
                and getattr(
                    self,
                    "point_track",
                    None
                ) is not None
                and len(self.point_track) > 0
            ):

                self.TARGET_POS = (
                    self.point_track.pop(0)
                )

                self._draw_target_marker(
                    [0, 1, 0]
                )

            self._prev_dist = None

        # ============================================================
        # FINAL DE LA TRAYECTORIA
        # ============================================================

        if (
            self.point_track is not None
            and not self.point_track
            and not self.one_only_target
        ):

            self.truncate_early = True
            bonus += 30.0

        total_reward += bonus

        self.actual_reward += total_reward

        return total_reward
        
    import math
    
    def _update_rabbit_target(self):
        """Actualiza TARGET_POS: programado, teclado o joystick."""
        pg.event.pump()
        if self.rabbit_mode is None:
            return

        dt = 1.0 / self.CTRL_FREQ

        # ============================================================
        # 1. CONEJO PROGRAMADO
        # ============================================================

        if self.rabbit_mode == "script":

            t = self.step_counter / self.CTRL_FREQ

            self.TARGET_POS = self.rabbit_center + np.array([
                self.rabbit_radius * np.cos(self.rabbit_omega * t),
                self.rabbit_radius * np.sin(self.rabbit_omega * t),
                0.5 * np.sin(0.5 * self.rabbit_omega * t)
            ])

        # ============================================================
        # 2. CONEJO CONTROLADO POR TECLADO
        # ============================================================

        # ============================================================
        # 2. CONEJO CONTROLADO POR TECLADO (PyBullet Native API)
        # ============================================================

        elif self.rabbit_mode == "keyboard":

            # Obtener el mapa de eventos del teclado desde PyBullet
            keys = p.getKeyboardEvents()

            x = 0.0
            y = 0.0
            z = 0.0

            # PyBullet usa constantes enteras o el valor ord() para las teclas.
            # Verificamos si la tecla está en estado pulsado (KEY_WAS_TRIGGERED) 
            # o mantenida (KEY_IS_DOWN).
            
            # Flecha Izquierda
            if p.B3G_LEFT_ARROW in keys and (keys[p.B3G_LEFT_ARROW] & (p.KEY_IS_DOWN | p.KEY_WAS_TRIGGERED)):
                x = -2.0
            # Flecha Derecha
            if p.B3G_RIGHT_ARROW in keys and (keys[p.B3G_RIGHT_ARROW] & (p.KEY_IS_DOWN | p.KEY_WAS_TRIGGERED)):
                x = 2.0
            # Flecha Arriba
            if p.B3G_UP_ARROW in keys and (keys[p.B3G_UP_ARROW] & (p.KEY_IS_DOWN | p.KEY_WAS_TRIGGERED)):
                y = 2.0
            # Flecha Abajo
            if p.B3G_DOWN_ARROW in keys and (keys[p.B3G_DOWN_ARROW] & (p.KEY_IS_DOWN | p.KEY_WAS_TRIGGERED)):
                y = -2.0

            # Tecla 'q' (usando el código ASCII ord('q'))
            if ord('q') in keys and (keys[ord('q')] & (p.KEY_IS_DOWN | p.KEY_WAS_TRIGGERED)):
                z = 2.0
            # Tecla 'a' (usando el código ASCII ord('a'))
            if ord('a') in keys and (keys[ord('a')] & (p.KEY_IS_DOWN | p.KEY_WAS_TRIGGERED)):
                z = -2.0

            self.TARGET_POS[0] += x * self.rabbit_speed * dt
            self.TARGET_POS[1] += y * self.rabbit_speed * dt
            self.TARGET_POS[2] += z * self.rabbit_speed * dt

        # ============================================================
        # 3. CONEJO CONTROLADO POR JOYSTICK
        # ============================================================

        elif self.rabbit_mode == "joystick":


            if self.rabbit_joystick is None:


                if pg.joystick.get_count() == 0:
                    raise RuntimeError(
                        "No se detectó ningún joystick."
                    )

                self.rabbit_joystick = pg.joystick.Joystick(0)
                self.rabbit_joystick.init()

                print(
                    "Joystick conectado:",
                    self.rabbit_joystick.get_name()
                )

            pg.event.pump()

            # Stick izquierdo
            x = self.rabbit_joystick.get_axis(0)
            y = -self.rabbit_joystick.get_axis(1)

            # Stick derecho vertical para altura
            z = -self.rabbit_joystick.get_axis(3)

            # Dead-zone
            deadzone = 0.10

            if abs(x) < deadzone:
                x = 0.0

            if abs(y) < deadzone:
                y = 0.0

            if abs(z) < deadzone:
                z = 0.0

            self.TARGET_POS[0] += x * self.rabbit_speed * dt
            self.TARGET_POS[1] += y * self.rabbit_speed * dt
            self.TARGET_POS[2] += z * self.rabbit_speed * dt

        else:
            raise ValueError(
                f"rabbit_mode desconocido: {self.rabbit_mode}"
            )

        # ============================================================
        # LÍMITES
        # ============================================================

        self.TARGET_POS[0] = np.clip(
            self.TARGET_POS[0], -15.0, 15.0
        )

        self.TARGET_POS[1] = np.clip(
            self.TARGET_POS[1], -15.0, 15.0
        )

        self.TARGET_POS[2] = np.clip(
            self.TARGET_POS[2], 0.5, 5.0
        )


        
        p.resetBasePositionAndOrientation(
        self._target_visual_id,
        self.TARGET_POS,
        [0, 0, 0, 1]  # Cuaternión identidad
        )
            
        
        
        
    # 1. Definimos la FUNCIÓN que genera la lista
    def generar_trayectoria(self, formula_figura, pasos=100):
        lista_puntos = []
        min_spacing = 0.4
        incremento = 0.01
        for i in range(pasos + 1):
            # Aquí es donde PREPARAMOS el valor de p (de 0.0 a 1.0)
            p = (i+1) / pasos
            
            punto_base = formula_figura(p)
            punto = punto_base

            # Mantiene la posición dentro de la misma figura, pero evita
            # aceptar objetivos que estén dentro o demasiado cerca de un obstáculo.
            
            for _ in range(500):
                if self._is_space_clear(punto, radius=0.2):
                    break
                # Desplaza el candidato en una dirección aleatoria para salir
                # del obstáculo sin cambiar el orden de la trayectoria.
                punto = punto_base + np.random.uniform(
                    low=[-0.4 - incremento, -0.4 - incremento, -0.1],
                    high=[0.4 + incremento, 0.4 + incremento, 0.1]
                )
                incremento += 0.01
                
            else:
                raise RuntimeError(
                    f"No se encontró un punto libre para p={p:.3f} "
                    "después de 500 intentos"
                )
            
            if not lista_puntos or np.linalg.norm(punto - lista_puntos[-1]) >= min_spacing:
                lista_puntos.append(punto)

        if lista_puntos and np.linalg.norm(lista_puntos[-1] - punto_base) >= min_spacing:
            lista_puntos.append(punto_base)

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
            print('step counter', self.step_counter)
            self.actual_reward = 0
            self.truncate_early = False
            return True, penalty * 0
        
        if (abs(state[0]) > 20 or abs(state[1]) > 20 or state[2] > 80 # Truncate when the drone is too far away
        ):
            #print(  f"Truncated far away: pos {state[0:3]}, angles {state[7:10]}")
            print("far away - reward: "  + str(self.actual_reward - penalty))
            print('score', self.score)
            print('step counter', self.step_counter)
            self.actual_reward = 0
            return True, penalty
        
        if (abs(state[7]) > 1.3 or abs(state[8]) > 1.3):

            print(
                "tilted - accumulated reward: "
                + str(self.actual_reward)
            )

            print("score", self.score)
            print('step counter', self.step_counter)

            self.actual_reward = 0

            return True, penalty
        
        if state[2] < 0.02:
            #print(  f"Truncated height: pos {state[0:3]}, angles {state[7:10]}")
            print("height limit - reward: "  + str(self.actual_reward - penalty))
            print('score', self.score)
            print('step counter', self.step_counter)
            self.actual_reward = 0
            return True, penalty
        

        
        if self.lidar is not None and np.max(self.lidar) > 0.95:

            print(
                "collision special!!!! - accumulated reward: "
                + str(self.actual_reward)
            )

            print("score", self.score)
            print('step counter', self.step_counter)
            self.actual_reward = 0

            return True, penalty

        if self.obstacle_collision:

            print(
                "obstacle collision - accumulated reward: "
                + str(self.actual_reward)
            )

            print("score", self.score)
            print('step counter', self.step_counter)

            self.actual_reward = 0

            return True, penalty
        
        if np.linalg.norm(self.TARGET_POS-state[0:3]) < .001:
            #print("target reached - reward: "  + str(self.actual_reward - penalty))
            #print('score', self.score)
            print('step counter', self.step_counter)
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
