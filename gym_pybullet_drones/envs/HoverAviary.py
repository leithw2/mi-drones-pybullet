import numpy as np
import pybullet as p
import pygame as pg
import math
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType
from gym_pybullet_drones.utils.powerloop_track import PowerloopTrack
import matplotlib.pyplot as plt

class HoverAviary(BaseRLAviary):
    POWERLOOP_GATE_CROSSING_BONUS = 50.0
    POWERLOOP_MIN_GATE_HEADING_ALIGNMENT = 0.5

    
    
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
                 rabbit_mode = None, # "script" , "keyboard"
                 track_scale: float = 1.0,
                 track_position_variation: float = 0.02,
                 track_orientation_variation: float = 0.02,
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
        self.track_scale = track_scale
        self.track_position_variation = track_position_variation
        self.track_orientation_variation = track_orientation_variation

        self.rabbit_center = np.array([0.0, 0.0, 1.5])
        self.rabbit_target = self.rabbit_center.copy()

        self.rabbit_radius = 5.0
        self.rabbit_omega = 0.25
        self.rabbit_speed = 1.0

        self.rabbit_time = 0.0

        self.rabbit_joystick = None
        self.verbose = 1
        self.wrong_gate_cross = False
        self._prev_gate_plane_dists_dict = {}
        self.is_powerloop = False
                
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
        self.gate_data = None
    
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
        for line_id in getattr(self, "_trajectory_ids", []):
            p.removeUserDebugItem(line_id, physicsClientId=self.CLIENT)

        if not self.GUI or len(trajectory) < 2:
            self._trajectory_ids = []
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
        seed = kwargs.get("seed")
        if seed is None and args:
            seed = args[0]
        if seed is not None:
            self._episode_rng = np.random.RandomState(seed)
        elif not hasattr(self, "_episode_rng"):
            self._episode_rng = self.np_random
        rng = self._episode_rng

        self.score = 1
        self.time_penalty = 0
        self._prev_ang_vel = None
        self._probe_id = None
        self._probe_shape = None
        self.wrong_gate = False
        self._prev_gate_plane_dists_dict = {}

        if hasattr(self, "track"):
            self.track.current_gate_idx = 0
        
        
        if self.randomized :
            self.INIT_XYZS = np.array([[rng.uniform(-0.1, 0.1)+.8 ,rng.uniform(-0.1, 0.1) + 2, rng.uniform(0.8, 1.0)]])
            self.INIT_XYZS[0, :2] *= self.track_scale

        # Habilitar orientación inicial aleatoria en Yaw
        if self.INIT_RPYS is not None:
            self.INIT_RPYS[0, 2] = rng.uniform(-np.pi/6, np.pi/6)
            

        preserved_scene_ids = set(
            getattr(self, "_powerloop_static_body_ids", ())
        )
        preserved_scene_ids.update(self._get_registered_obstacle_body_ids())
        if preserved_scene_ids:
            obs, info = self.reset_preserving_bodies(
                seed=seed,
                preserved_body_ids=preserved_scene_ids,
            )
        else:
            obs, info = super().reset(*args, **kwargs)
        
        if self.TEST_BODY_ID is None:
            self.TEST_BODY = p.createCollisionShape(
                p.GEOM_SPHERE,
                radius=.6,
                physicsClientId=self.CLIENT,
            )
            self.TEST_BODY_ID = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=self.TEST_BODY,
                basePosition=[0, 0, -10],
                physicsClientId=self.CLIENT,
            )
        
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
            # Parametrización general: p va de 0.0 a 1.0
            if self.randomized:
                Amplitudx = rng.uniform(1, 3)
                Amplitudy = rng.uniform(1, 3)
            else: 
                Amplitudx = 6.0
                Amplitudy = 6.0

            # 1. Lemniscata 8
            lemniscata_8 = lambda p: np.round(np.array([
                Amplitudx * math.sin(6 * math.pi * p),
                Amplitudy * math.sin(12 * math.pi * p) / 2.0,
                1.6 + 0.8 * math.cos(2 * math.pi * p)
            ]), 2)
            
            # 2. Lemniscata 8 Invertida
            lemniscata_8_inv = lambda p: np.round(np.array([
                -Amplitudx * math.sin(6 * math.pi * p),
                -Amplitudy * math.sin(12 * math.pi * p) / 2.0,
                1.6 + 0.8 * math.cos(2 * math.pi * p)
            ]), 2)

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
                # Opciones de benchmark, incluyendo el Powerloop Track
                #benchmarks = [lemniscata_8, lemniscata_8_inv, "powerloop"]
                #benchmark_names = ["Lemniscata 8", "Lemniscata 8 Invertida", "Powerloop Track"]
                
                benchmarks = ["powerloop"]
                benchmark_names = ["Powerloop Track"]
                
                # Selección aleatoria o asignada
                self.task_idx = rng.choice(len(benchmarks))
                self.selected_benchmark = benchmarks[self.task_idx]
                # print("Benchmark seleccionado: ", benchmark_names[self.task_idx])
                
                if self.randomized:
                    self.pasos = rng.choice(np.arange(50, 60))
                else: 
                    self.pasos = 50
                
                if self.selected_benchmark == "powerloop":
                    self.is_powerloop = True
                    if not hasattr(self, 'track'):
                        self.track = PowerloopTrack(
                            show_labels=self.GUI,
                            enable_collisions=True,
                            physics_client_id=self.CLIENT,
                            scale=self.track_scale,
                            position_variation=self.track_position_variation,
                            orientation_variation=self.track_orientation_variation,
                            seed=(
                                int(seed)
                                if seed is not None
                                else int(rng.randint(0, 2**32))
                            ),
                        )

                    if not getattr(self, "_powerloop_static_body_ids", None):
                        self.track.create()
                        self._powerloop_static_body_ids = {
                            int(body_id)
                            for structure in self.track.gate_structures
                            for body_id in structure["base_bodies"] + structure["front_bodies"]
                        }
                    else:
                        self.track.current_gate_idx = 0
                    
                    self.gate_data = self.track.get_gate_data(
                        self.track.track_points
                    )

                    base_gate_positions = np.array([
                        gate["position"] for gate in self.gate_data
                    ], dtype=np.float32)

                    base_gate_normals = np.array([
                        gate["normal"] for gate in self.gate_data
                    ], dtype=np.float32)
                    
                    # Número total de waypoints = N
                    self.num_waypoints = 21   # 3 vueltas de 7 puertas

                    repetitions = int(np.ceil(
                        self.num_waypoints / len(base_gate_positions)
                    ))

                    self.gate_positions = np.tile(
                        base_gate_positions,
                        (repetitions, 1)
                    )[:self.num_waypoints]

                    self.gate_normals = np.tile(
                        base_gate_normals,
                        (repetitions, 1)
                    )[:self.num_waypoints]
                    
                    base_physical_ids = np.array([
                    gate["physical_id"]
                    for gate in self.gate_data
                    ], dtype=np.int32)
                    
                    base_physical_ids = np.array([
                        int(gate["physical_id"]) for gate in self.gate_data if gate["type"] == "gate"
                    ])

                    self.gate_physical_ids = np.resize(
                        base_physical_ids,
                        len(self.gate_positions)
                    )

                    self.current_gate_idx = 0

                    self.point_track = [
                        np.array(wp, dtype=np.float32)
                        for wp in self.gate_positions
                    ]

                    # ============================================================
                    # NUEVO: N WAYPOINTS TOTALES
                    # ============================================================
                    self.num_waypoints = 21   # 7 gates x 3 vueltas

                    base_waypoints = np.array(
                        PowerloopTrack.get_waypoints(
                            self.track.track_points
                        ),
                        dtype=np.float32
                    )

                    num_track_waypoints = len(base_waypoints)

                    # Repetir la pista tantas veces como sea necesario
                    repetitions = int(
                        np.ceil(
                            self.num_waypoints
                            / num_track_waypoints
                        )
                    )

                    self.point_track = [
                        np.array(wp, dtype=np.float32)
                        for wp in np.tile(
                            base_waypoints,
                            (repetitions, 1)
                        )[:self.num_waypoints]
                    ]

                    self.current_gate_idx = 0
                else:
                    self.point_track = self.generar_trayectoria(
                        self.selected_benchmark,
                        pasos=self.pasos
                    )

                self.pasos = len(self.point_track)

                # Dibujar la ruta y asignar el primer objetivo
                if (
                    not self.is_powerloop
                    or not getattr(self, "_powerloop_trajectory_drawn", False)
                ):
                    self._draw_trajectory(self.point_track)
                    if self.is_powerloop:
                        self._powerloop_trajectory_drawn = True
                self.TARGET_POS = self.point_track.pop(0)

                #print(f"Benchmark Activo: ID {self.task_idx} | Puntos Restantes: {len(self.point_track)}")

            elif self.one_only_target and not self.random_targets:
                self.TARGET_POS = np.array([0.0, 0.0, 1.0])
                self.pasos = 0
                self._draw_target_marker([0, 1, 0])

        self._prev_dist = None
        self.truncate_early = False
        
        self.prev_action = None
        self.lidar_risk_memory = 0.0
        self.wrong_gate = False

        self._prev_dist = None

        self._prev_gate_plane_dists = None

        self.wrong_gate_detected = False
        self.wrong_gate_idx = None

        self.terminate_on_wrong_gate = False
        
        return self._computeObs(), info


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
        minimum_risk = 0.80

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
        reward_rate_scale = (
            1.0 / self.CTRL_FREQ if self.is_powerloop else 1.0
        )
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


        # ============================================================
        # NUEVO: PARA UNA PUERTA, USAR SU NORMAL COMO DIRECCIÓN
        # DE ORIENTACIÓN
        # ============================================================

        if self.is_powerloop:

            
            current_gate = self.gate_data[self.current_gate_idx]

            if current_gate["type"] == "gate":

                gate_normal = np.asarray(
                    current_gate["normal"],
                    dtype=np.float32
                )

                gate_normal /= (
                    np.linalg.norm(gate_normal) + 1e-8
                )

                # Normal de la puerta expresada en el frame del dron
                gate_normal_local = (
                    rotation_matrix.T @ gate_normal
                )

                # El +X del dron debe apuntar hacia la normal
                gate_heading_alignment = float(
                    np.clip(
                        gate_normal_local[0],
                        -1.0,
                        1.0
                    )
                )

                # Usamos la orientación de la puerta
                heading_alignment = gate_heading_alignment
        

        # ============================================================
        # NUEVO: PROGRESO HACIA LA PUERTA
        # ============================================================

        if self.is_powerloop and current_gate["type"] == "gate":

            # Velocidad global proyectada sobre la normal de la puerta.
            # Positivo = avanzando en el sentido correcto para atravesarla.
            forward_progress = float(
                np.dot(vel , gate_normal_local )
            )

        else:

            # Para waypoints conservamos el comportamiento anterior.
            forward_progress = float(
                np.dot(vel[:2], target_xy)
            )

        if self.is_powerloop and current_gate["type"] == "gate":

            forward_speed = float(
                max(np.dot(vel, gate_normal_local ), 0.0)
            )

        else:

            forward_speed = float(
                max(vel[0], 0.0)
            )

        speed_magnitude = float(
            np.linalg.norm(vel)
        )

        # ============================================================
        # GUÍA DE OBSTÁCULOS
        # ============================================================

        obstacle_risk, safer_side = self._get_obstacle_guidance(vel)

        collision_penalty = -0.5 * obstacle_risk

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
                -15.0,
                0.0
            )
        ) * reward_rate_scale

        braking_penalty = (
            -2.0
            * obstacle_risk
            * max(forward_speed - 0.35, 0.0)
        ) * reward_rate_scale

        safe_lateral_speed = safer_side * vel[1]

        avoidance_reward = (
            2.5
            * obstacle_risk
            * np.clip(
                safe_lateral_speed,
                0.0,
                1.0
            )
        ) * reward_rate_scale

        wrong_side_penalty = (
            -1.5
            * obstacle_risk
            * np.clip(
                -safe_lateral_speed,
                0.0,
                1.0
            )
        ) * reward_rate_scale

        # ============================================================
        # CONTROL DE VELOCIDAD MÁXIMA
        # ============================================================

        target_speed_limit = 2.0

        excess_speed = max(
            0.0,
            speed_magnitude - target_speed_limit
        )

        excess_speed_penalty = (
            -1.5
            * (excess_speed ** 2)
        ) * reward_rate_scale

        # ============================================================
        # TAREA PRINCIPAL
        # ============================================================

        base_reward = 0.0
        approach_reward = 10.0 * progress
        approach_reward = (
            8.0
            * max(progress, 0.0)
            * (1.0 - 0.65 * obstacle_risk)
        )

        no_approach_penalty = 0.0
        #print("progress--------- ", progress)
        if progress <= 0.00 and dist > 0.8:
            no_approach_penalty -= (
                1.5
                * (
                    1.0
                    + min(dist, 8.0) / 8.0
                )
            )

        if forward_progress < 0.05 and dist > 1.0:
            no_approach_penalty -= 0.3

        if heading_alignment < 0.2 and dist > 1.0:
            no_approach_penalty -= 0.1
        no_approach_penalty *= reward_rate_scale

        heading_reward = (
            1.2
            * max(heading_alignment, 0.0)
            * max(0.0, 1.0 - dist / 8.0)
        ) * reward_rate_scale

        speed_reward = (
            0.4
            * max(forward_progress, 0.0)
            * max(0.0, 1.0 - dist / 6.0)
        ) * reward_rate_scale

        reverse_penalty = (
            1.0
            * min(forward_progress, 0.0)
        ) * reward_rate_scale

        time_penalty = -0.003

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
        ) * reward_rate_scale

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
        ) * reward_rate_scale

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
        ) * reward_rate_scale

        # ============================================================
        # 4. MANTENER HORIZONTE
        # ============================================================

        horizon_limit = np.deg2rad(20.0)

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
        ) * reward_rate_scale

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
        ) * reward_rate_scale

        # ============================================================
        # WAYPOINT / GATE VALIDATION & REWARD
        # ============================================================

        # 1. Asegurar que las variables locales de posición y velocidad sean 1D
        target_pos_1d = np.array(self.TARGET_POS).flatten()
        current_pos_1d = (
            np.array(self.pos).flatten()
        )
        current_v_1d = (
            np.array(self.vel).flatten()
        )

        dist = np.linalg.norm(target_pos_1d - current_pos_1d)
        speed_mag = np.linalg.norm(current_v_1d)

        bonus = 0.0
        wrong_direction_penalty = 0.0
        gate_wait_penalty = 0
        

        if self.is_powerloop:
            # ============================================================
            # ESTADO DE LA RUTA AL INICIO DEL STEP
            # ============================================================
            #
            # IMPORTANTE:
            # current_gate_idx NO se modifica mientras recorremos
            # gate_data.
            #
            # Esto evita que Gate 3/6 y TARGET_POS queden desincronizados.
            # ============================================================

            gate_idx_at_start = self.current_gate_idx

            if gate_idx_at_start < len(self.gate_data):

                current_target_info = self.gate_data[
                    gate_idx_at_start
                ]

                if self.verbose >= 2:
                    print(
                        f"\n[ROUTE STATE] "
                        f"route_idx={gate_idx_at_start} | "
                        f"type={current_target_info.get('type')} | "
                        f"physical_id={current_target_info.get('physical_id', None)} | "
                        f"position={current_target_info.get('position', None)}"
                    )
                
                is_current_waypoint = (
                    current_target_info["type"] == "waypoint"
                )

                # --------------------------------------------------------
                # Identidad física de la puerta objetivo
                # --------------------------------------------------------

                current_physical_id = None

                if not is_current_waypoint:
                    current_physical_id = int(
                        current_target_info["physical_id"]
                    )

                # --------------------------------------------------------
                # Variables de resultado
                # --------------------------------------------------------

                correct_crossing = False
                wrong_direction_current_gate = False
                wrong_gate_detected = False
                wrong_gate_physical_id = None
                correct_crossing = False

                # Guarda qué puerta física ya fue cruzada correctamente en ESTE timestep
                correct_crossing_physical_id = None

                gate_velocity_current = 0.0
                self.gate_inside = False

                # ========================================================
                # A. WAYPOINT VIRTUAL
                # ========================================================
                waypoint_radius = 1.5
                if is_current_waypoint:
                    waypoint_dist = np.linalg.norm(pos - np.asarray(current_target_info["position"]))

                    if waypoint_dist < waypoint_radius:
                        if self.verbose >= 2:
                            print(
                                f"[WAYPOINT REACHED] route_idx={gate_idx_at_start} "
                                f"| dist={waypoint_dist:.3f}"
                            )

                        # ¿Es el último elemento de la pista?
                        if gate_idx_at_start == len(self.gate_data) - 1:

                            print("[TRACK COMPLETE] Reiniciando pista")
                            bonus += 50.0
                            self.score += 1
                            self.current_gate_idx = 0

                            self.TARGET_POS = np.asarray(
                                self.gate_data[0]["position"],
                                dtype=np.float32
                            )

                            self._prev_dist = None
                            self._prev_gate_plane_dists_dict = {}

                        else:
                            self.current_gate_idx = gate_idx_at_start + 1

                            self.TARGET_POS = np.asarray(
                                self.gate_data[self.current_gate_idx]["position"],
                                dtype=np.float32
                            )

                            self._prev_dist = None

                # ========================================================
                # B. HISTORIAL DE PLANOS
                # ========================================================

                if not hasattr(
                    self,
                    "_prev_gate_plane_dists_dict"
                ):
                    self._prev_gate_plane_dists_dict = {}

                # ========================================================
                # C. PROCESAR TODAS LAS PUERTAS FÍSICAS
                # ========================================================
                correct_crossing_physical_id = None
                
                for physical_idx, gate in enumerate(
                    self.gate_data
                ):

                    if gate.get("type") != "gate":
                        continue

                    gate_pid = int(
                        gate["physical_id"]
                    )

                    # ----------------------------------------------------
                    # POSICIÓN Y NORMAL GLOBAL
                    # ----------------------------------------------------

                    gate_position_global = np.asarray(
                        gate["position"],
                        dtype=np.float32
                    )

                    gate_normal_global = np.asarray(
                        gate["normal"],
                        dtype=np.float32
                    )

                    gate_normal_global /= (
                        np.linalg.norm(
                            gate_normal_global
                        ) + 1e-8
                    )

                    # ----------------------------------------------------
                    # OFFSET DRON -> PUERTA
                    # ----------------------------------------------------

                    gate_offset_global = (
                        gate_position_global
                        - current_pos_1d
                    )

                    gate_offset_local = (
                        rotation_matrix.T
                        @ gate_offset_global
                    )

                    # ----------------------------------------------------
                    # NORMAL EN FRAME LOCAL
                    # ----------------------------------------------------

                    gate_normal_local = (
                        rotation_matrix.T
                        @ gate_normal_global
                    )

                    gate_normal_local /= (
                        np.linalg.norm(
                            gate_normal_local
                        ) + 1e-8
                    )

                    # ----------------------------------------------------
                    # EJES LOCALES DE LA PUERTA
                    # ----------------------------------------------------

                    world_up = np.array(
                        [0.0, 0.0, 1.0],
                        dtype=np.float32
                    )

                    if abs(
                        np.dot(
                            gate_normal_global,
                            world_up
                        )
                    ) > 0.99:

                        world_up = np.array(
                            [0.0, 1.0, 0.0
                        ], dtype=np.float32)

                    gate_x_global = np.cross(
                        world_up,
                        gate_normal_global
                    )

                    gate_x_global /= (
                        np.linalg.norm(
                            gate_x_global
                        ) + 1e-8
                    )

                    gate_y_global = np.cross(
                        gate_normal_global,
                        gate_x_global
                    )

                    gate_y_global /= (
                        np.linalg.norm(
                            gate_y_global
                        ) + 1e-8
                    )

                    gate_x_local = (
                        rotation_matrix.T
                        @ gate_x_global
                    )

                    gate_y_local = (
                        rotation_matrix.T
                        @ gate_y_global
                    )

                    # ====================================================
                    # DISTANCIA FIRMADA AL PLANO
                    # ====================================================

                    current_plane_dist = float(
                        np.dot(
                            -gate_offset_local,
                            gate_normal_local
                        )
                    )

                    # ====================================================
                    # HISTORIAL
                    # ====================================================

                    if physical_idx not in (
                        self._prev_gate_plane_dists_dict
                    ):

                        self._prev_gate_plane_dists_dict[
                            physical_idx
                        ] = current_plane_dist

                    previous_plane_dist = (
                        self._prev_gate_plane_dists_dict[
                            physical_idx
                        ]
                    )

                    # ====================================================
                    # VELOCIDAD ATRAVESANDO LA PUERTA
                    # ====================================================

                    gate_velocity = float(
                        np.dot(
                            vel,
                            gate_normal_local
                        )
                    )

                    # ====================================================
                    # POSICIÓN LATERAL / VERTICAL
                    # ====================================================

                    lateral_offset = abs(
                        np.dot(
                            -gate_offset_local,
                            gate_x_local
                        )
                    )

                    vertical_offset = abs(
                        np.dot(
                            -gate_offset_local,
                            gate_y_local
                        )
                    )

                    is_inside = (
                        lateral_offset <= 0.65
                        and vertical_offset <= 0.65
                    )

                    # ====================================================
                    # CRUCE HACIA ADELANTE
                    # ====================================================

                    crossing_forward = (
                        previous_plane_dist < 0.0
                        and current_plane_dist >= 0.0
                        and is_inside
                        and gate_velocity > 0.05
                    )

                    # ====================================================
                    # CRUCE HACIA ATRÁS
                    # ====================================================

                    crossing_backward = (
                        previous_plane_dist > 0.0
                        and current_plane_dist <= 0.0
                        and is_inside
                        and gate_velocity < -0.05
                    )

                    # ====================================================
                    # ¿ES LA PUERTA ACTUAL DE LA RUTA?
                    #
                    # IMPORTANTE:
                    # usamos gate_idx_at_start.
                    #
                    # Aunque otra puerta sea cruzada después,
                    # la identidad de la puerta objetivo NO cambia
                    # durante este timestep.
                    # ====================================================

                    if self.verbose >= 2:
                        print(
                            f"[GATE CHECK] "
                            f"data_idx={physical_idx} | "
                            f"physical_id={gate_pid} | "
                            f"is_target_object={gate is current_target_info} | "
                            f"prev_plane={previous_plane_dist:.3f} | "
                            f"curr_plane={current_plane_dist:.3f} | "
                            f"velocity={gate_velocity:.3f} | "
                            f"inside={is_inside}"
                        )
                    
                    is_current_gate = (
                        not is_current_waypoint
                        and physical_idx == gate_idx_at_start
                    )
                    
                    if self.verbose == 2:
                        print(
                            f"[GATE ROLE] "
                            f"data_idx={physical_idx} | "
                            f"physical_id={gate_pid} | "
                            f"CURRENT={is_current_gate}"
                        )
                    
                    
                    
                    # ====================================================
                    # PUERTA ACTUAL
                    # ====================================================

                    if is_current_gate:

                        self.gate_inside = is_inside
                        gate_velocity_current = gate_velocity

                        if crossing_forward and is_inside and gate_velocity > 0.1:
                            if gate_normal_local[0] >= self.POWERLOOP_MIN_GATE_HEADING_ALIGNMENT:
                                if self.verbose >= 2:
                                    print(
                                        f"[CROSS FORWARD] data_idx={physical_idx} "
                                        f"| physical_id={gate_pid} "
                                        f"| heading_alignment={gate_normal_local[0]:.3f}"
                                    )
                                correct_crossing = True
                                correct_crossing_physical_id = gate_pid
                            else:
                                if self.verbose >= 1:
                                    print(
                                        f"[WRONG GATE ORIENTATION] data_idx={physical_idx} "
                                        f"| physical_id={gate_pid} "
                                        f"| heading_alignment={gate_normal_local[0]:.3f} "
                                        f"| required>={self.POWERLOOP_MIN_GATE_HEADING_ALIGNMENT:.3f}"
                                    )
                                wrong_direction_current_gate = True

                        if (
                            crossing_backward
                            and is_inside
                            and gate_velocity < -0.1
                        ):
                            if self.verbose >= 2:
                                print(
                                    f"[CROSS BACKWARD] data_idx={physical_idx} "
                                    f"| physical_id={gate_pid} "
                                    f"| CURRENT={is_current_gate}"
                                )

                            if is_current_gate:
                                wrong_direction_current_gate = True

                    # ====================================================
                    # PUERTA QUE NO ES LA ACTUAL
                    # ====================================================

                    else:

                        # ============================================================
                        # PUERTA NO ACTUAL
                        #
                        # Si comparte physical_id con la puerta actual,
                        # es la MISMA puerta física (ej. Gate 3 / Gate 6).
                        # NO debe considerarse una puerta equivocada.
                        # ============================================================

                        same_physical_gate = (
                            gate_pid == current_physical_id
                        )

                        crossing_any_direction = (
                            crossing_forward
                            or crossing_backward
                        )

                        if (
                            crossing_any_direction
                            and is_inside
                            and not same_physical_gate
                        ):

                            wrong_gate_detected = True
                            wrong_gate_physical_id = gate_pid

                            if self.verbose == 2:
                                print(
                                    f"[WRONG GATE DETECTED] "
                                    f"route_idx={gate_idx_at_start} | "
                                    f"target_physical_id={current_physical_id} | "
                                    f"crossed_data_idx={physical_idx} | "
                                    f"crossed_physical_id={gate_pid}"
                                )
                    # ====================================================
                    # ACTUALIZAR HISTORIAL
                    # ====================================================

                    self._prev_gate_plane_dists_dict[
                        physical_idx
                    ] = current_plane_dist

                # ========================================================
                # D. ACTUALIZAR LA RUTA DESPUÉS DE EVALUAR TODAS
                # LAS PUERTAS
                # ========================================================

                if correct_crossing:

                    self.score += 1
                    bonus += self.POWERLOOP_GATE_CROSSING_BONUS

                    next_gate_idx = (
                        gate_idx_at_start + 1
                    ) % len(self.gate_data)

                    self.current_gate_idx = (
                        next_gate_idx
                    )

                    if next_gate_idx < len(
                        self.gate_positions
                    ):

                        self.TARGET_POS = np.asarray(
                            self.gate_positions[
                                next_gate_idx
                            ],
                            dtype=np.float32
                        )

                    self._prev_dist = None

                # ========================================================
                # E. CRUCE INCORRECTO
                # ========================================================

                if wrong_gate_detected:

                    wrong_direction_penalty = -20.0
                    self.wrong_gate_cross = True

                    if self.verbose == 2:
                        print(
                            f"[WRONG GATE] "
                            f"Puerta física {wrong_gate_physical_id} "
                            f"cruzada. "
                            f"Objetivo de ruta: "
                            f"{gate_idx_at_start}"
                        )

                elif wrong_direction_current_gate:

                    wrong_direction_penalty = -20.0
                    self.wrong_gate_cross = True

                    if self.verbose == 2:
                        print(
                            f"[BACKWARD CROSSING] "
                            f"Puerta actual: "
                            f"{gate_idx_at_start}"
                        )
        else:

            # ============================================================
            # OTROS BENCHMARKS: LÓGICA ORIGINAL
            # ============================================================

            if dist < waypoint_radius and valid_speed and valid_direction:

                bonus = 30 + (self.score - 1) * 20

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

                    next_target = self.point_track.pop(0)

                    self.TARGET_POS = (
                        np.array(next_target).flatten()
                    )

                    self._draw_target_marker(
                        [0, 1, 0]
                    )

                self._prev_dist = None

        # ============================================================
        # FINAL DE LA TRAYECTORIA
        # ============================================================
        if self.score > 14 :
            self.truncate_early = True
            bonus += 30.0
        if (
            self.point_track is not None
            and not self.point_track
            and not self.one_only_target
        ):

            self.truncate_early = True
            bonus += 30.0

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
            + wrong_direction_penalty
        )
        
        if self.verbose == 2:
            print("\n========== REWARD ==========")
            print(f"base_reward:           {base_reward:.3f}")
            print(f"approach_reward:       {approach_reward:.3f}")
            print(f"heading_reward:        {heading_reward:.3f}")
            print(f"speed_reward:          {speed_reward:.3f}")
            print(f"reverse_penalty:       {reverse_penalty:.3f}")
            print(f"no_approach_penalty:   {no_approach_penalty:.3f}")
            print(f"jitter_penalty:        {jitter_penalty:.3f}")
            print(f"ang_vel_penalty:       {ang_vel_penalty:.3f}")
            print(f"excess_speed_penalty:  {excess_speed_penalty:.3f}")
            print(f"smooth_flight_reward:  {smooth_flight_reward:.3f}")
            print(f"action_change_penalty: {action_change_penalty:.3f}")
            print(f"collision_penalty:     {collision_penalty:.3f}")
            print(f"braking_penalty:       {braking_penalty:.3f}")
            print(f"avoidance_reward:      {avoidance_reward:.3f}")
            print(f"wrong_side_penalty:    {wrong_side_penalty:.3f}")
            print(f"time_penalty:          {time_penalty:.3f}")
            print(f"horizon_penalty:       {horizon_penalty:.3f}")
            print(f"wrong_direction:       {wrong_direction_penalty:.3f}")
            print(f"TOTAL REWARD:          {total_reward:.3f}")
            print("============================\n")
        
        
        total_reward += bonus

        self.actual_reward += total_reward

        return total_reward
        
    
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
            if self.verbose >=1:
                
                print("Early Truncated - reward: "  + str(self.actual_reward))
                print('score', self.score)
                print('step counter', self.step_counter)
            self.actual_reward = 0
            self.truncate_early = False
            return True, penalty * 0
        
        if (abs(state[0]) > 20 or abs(state[1]) > 20 or state[2] > 80 # Truncate when the drone is too far away
        ):
            #print(  f"Truncated far away: pos {state[0:3]}, angles {state[7:10]}")
            if self.verbose >= 1:
                            
                print("far away - reward: "  + str(self.actual_reward - penalty))
                print('score', self.score)
                print('step counter', self.step_counter)
            self.actual_reward = 0
            return True, penalty
        
        if (abs(state[7]) > 1.7 or abs(state[8]) > 1.7):

            if self.verbose >= 1:
                print(
                    "tilted - accumulated reward: "
                    + str(self.actual_reward)
                )
                print("score", self.score)
                print("step counter", self.step_counter)

            self.actual_reward = 0

            return True, penalty
        
        if state[2] < 0.02:
            #print(  f"Truncated height: pos {state[0:3]}, angles {state[7:10]}")
            if self.verbose >=1:
                            
                print("height limit - reward: "  + str(self.actual_reward - penalty))
                print('score', self.score)
                print('step counter', self.step_counter)
            self.actual_reward = 0
            return True, penalty
        

        
        if self.lidar is not None and np.max(self.lidar) > 0.95:
            if self.verbose >=1:
                            
                print(
                    "collision special!!!! - accumulated reward: "
                    + str(self.actual_reward)
                )

                print("score", self.score)
                print('step counter', self.step_counter)
            self.actual_reward = 0

            return True, penalty

        if self.obstacle_collision:

            if self.verbose >=1:
                            
                print(
                    "obstacle collision - accumulated reward: "
                    + str(self.actual_reward)
                )

                print("score", self.score)
                print('step counter', self.step_counter)

            self.actual_reward = 0

            return True, penalty
        
        if self.wrong_gate_cross:
        
            if self.verbose >=1:
                            
                print(
                    "wrong_gate_cross - accumulated reward: "
                    + str(self.actual_reward)
                )

                print("score", self.score)
                print('step counter', self.step_counter)

            self.actual_reward = 0
            self.wrong_gate_cross = False
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