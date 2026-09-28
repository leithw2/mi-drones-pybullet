import os
import numpy as np
import pybullet as p
import pkg_resources
from scipy.spatial.transform import Rotation as R
from gymnasium import spaces
from collections import deque
# Importación del modelo de ruido desde tu archivo sensor_noise.py
from gym_pybullet_drones.utils.SensorNoiseModel import SensorNoiseModel, IMUNoiseConfig, LaserNoiseConfig, OdometryNoiseConfig
from gym_pybullet_drones.utils.powerloop_track import PowerloopTrack
from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType, ImageType
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

class BaseRLAviary(BaseAviary):
    """Base single and multi-agent environment class for reinforcement learning."""
    
    ################################################################################

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=1,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 240,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM,
                 randomized = False,
                 enable_noise = False

                 
                 ):
        """Initialization of a generic single and multi-agent RL environment.

        Attributes `vision_attributes` and `dynamics_attributes` are selected
        based on the choice of `obs` and `act`; `obstacles` is set to True 
        and overridden with landmarks for vision applications; 
        `user_debug_gui` is set to False for performance.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        num_drones : int, optional
            The desired number of drones in the aviary.
        neighbourhood_radius : float, optional
            Radius used to compute the drones' adjacency matrix, in meters.
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
            The type of action space (1 or 3D; RPMS, thurst and torques, waypoint or velocity with PID control; etc.)

        """
        #### Create a buffer ########
        self.ACTION_BUFFER_SIZE = int(1) # buffer of the last n actions, to be added to the observation space for non-Markovian formulations of the problem
        self.LIDAR_BUFFER_SIZE = int(2) # buffer of the last n lidar readings, to be added to the observation space for non-Markovian formulations of the problem and to help with obstacle avoidance in vision-based tasks
        self.action_buffer = deque(maxlen=self.ACTION_BUFFER_SIZE)
        self.lidar_buffer = deque(maxlen=self.LIDAR_BUFFER_SIZE)
        self.action = np.zeros(4)
        ####
        # Initialize TARGET_POS to avoid attribute errors
        self.TARGET_POS = np.zeros(3) if num_drones == 1 else np.zeros((num_drones, 3))
        vision_attributes = True if obs == ObservationType.RGB else False
        self.OBS_TYPE = obs
        self.ACT_TYPE = act
        self.OBSTACLE_TYPE = "powertrack" # "cubes", "moving_cubes", "donas", "map"
        self.moving_obstacle_time = 0.0
        self.moving_obstacle_centers = np.empty((0, 3), dtype=float)
        self.moving_obstacle_motion = []
        self.prev_v = np.zeros( 3) # placeholder for previous velocity to calculate acceleration for the IMU readings in the kinematic observation space, initialized at zero
        
        self.posBo = [None for i in range(15)] # placeholder for obstacle positions in case we want to remove them later
        self.cubo_id = [None for i in range(15)] # placeholder for obstacle ids in case we want to remove them later
        self.dona_ids = [] # IDs of torus obstacles used for collision detection
        self.map_id = None # ID of the map mesh used for collision detection
        self.randomized = randomized
        
        # 1. Bandera para activar/desactivar ruido rápidamente
        self.enable_noise = enable_noise

        # 2. Instanciación del modelo de ruido con configuraciones por defecto
        # (o puedes personalizar los valores de std/bias aquí)
        self.noise_model = SensorNoiseModel(
            imu_config=IMUNoiseConfig(),
            laser_config=LaserNoiseConfig(),
            odom_config=OdometryNoiseConfig(),
            seed=None
        )
        #### Create integrated controllers #########################
        if act in [ActionType.PID, ActionType.VEL, ActionType.ONE_D_PID]:
            os.environ['KMP_DUPLICATE_LIB_OK']='True'
            if drone_model in [DroneModel.CF2X, DroneModel.CF2X250, DroneModel.CF2P]:
                self.ctrl = [DSLPIDControl(drone_model=DroneModel.CF2X) for i in range(num_drones)]
            else:
                print("[ERROR] in BaseRLAviary.__init()__, no controller is available for the specified drone_model")
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record, 
                         obstacles=True, # Add obstacles for RGB observations and/or FlyThruGate
                         user_debug_gui=False, # Remove of RPM sliders from all single agent learning aviaries
                         vision_attributes=vision_attributes,
                         randomized = randomized,
                         )
        #### Set a limit on the maximum target speed ###############
        if act == ActionType.VEL:
            self.SPEED_LIMIT = 0.03 * self.MAX_SPEED_KMH * (1000/3600)
        
        self.prev_action = None
        self.max_action_delta = 0.05
        self.smooth_lambda = 0.20
        self.BRUSHLESS_MOTOR_TIME_CONSTANT = 0.05
        self.brushless_rpm = np.full((self.NUM_DRONES, 4), self.HOVER_RPM)
        self.gate_normal_arrow_ids = [-1, -1, -1]
        self.GUI = gui
        
        

    ################################################################################
    def _addObstacles(self):
        """Add obstacles aligned with the Lemniscate path for ToF RL training."""
        valid_obstacle_types = {"cubes", "moving_cubes", "donas", "map", "none", "powertrack"}
        if self.OBSTACLE_TYPE not in valid_obstacle_types:
            raise ValueError(
                f"OBSTACLE_TYPE debe ser uno de: {sorted(valid_obstacle_types)}"
            )
        randoms_obstacles = True
        self.cubo_id = []
        self.dona_ids = []
        self.map_id = None
        self.moving_obstacle_time = 0.0
        self.moving_obstacle_centers = np.empty((0, 3), dtype=float)
        self.moving_obstacle_motion = []
        if self.OBSTACLE_TYPE in ("cubes", "moving_cubes", "donas"):
            if randoms_obstacles:
                # 1. Definir posiciones según las 3 Zonas Estratégicas:
                # ZONA 1 (Intersección Central): Cerca del centro (0,0) pero desfasado
                # ZONA 2 (Pasillos en las curvas): Pares laterales
                # ZONA 3 (Bloqueos en la curva): Sobre el trazo
                

                if self.randomized :
                    posx = lambda : np.random.uniform(-.1,.1)
                    posy = lambda : np.random.uniform(-.1,.1)
                    posz = lambda : np.random.uniform(-0.5, 2)
                else:
                    posx = lambda : 0
                    posy = lambda : 0
                    posz = lambda : 0
                
                self.posBo = [
                    [2.4 + posx(), 3.2 + posy(), 1.0 + posz()],
                    [5.4 + posx(), -1.4 + posy(), 1.0 + posz()],
                    [-5.0 + posx(), 0.4 + posy(), 1.0 + posz()],
                    [-2.3 + posx(), -2.3 + posy(), 1.0 + posz()],
                ]
                self.moving_obstacle_centers = np.asarray(self.posBo, dtype=float)
                
                # Los IDs se indexan por cubo durante la creación.
                self.cubo_id = {}
                
                # 3. Crear los cuerpos estáticos en la simulación
                create_cubes = self.OBSTACLE_TYPE in ("cubes", "moving_cubes")
                for i, pos in enumerate(self.posBo if create_cubes else []):
                    if self.randomized :
                        ancho = np.random.uniform(.2,.24)
                        largo = np.random.uniform(.2,.24)
                        alto  = np.random.uniform(1.5,2.7)
                    else:
                        ancho = 0.15
                        largo = 0.15
                        alto  = 1.8
                    # 2. Crear las formas en PyBullet
                    # halfExtents=[0.15, 0.15, 1.8] genera una columna de 30x30 cm x 3.6m de alto
                    col_id = p.createCollisionShape(
                        p.GEOM_BOX, 
                        halfExtents=[ancho, largo, alto], 
                        physicsClientId=self.CLIENT
                    )
                    vis_id = p.createVisualShape(
                        p.GEOM_BOX, 
                        halfExtents=[ancho, largo, alto], 
                        rgbaColor=[0.8, 0.2, 0.2, 1], 
                        physicsClientId=self.CLIENT
                    )
                    self.cubo_id[i] = p.createMultiBody(
                        baseMass=0,  # <--- CRÍTICO: 0 lo hace estático (inamovible al chocar)
                        baseCollisionShapeIndex=col_id,
                        baseVisualShapeIndex=vis_id,
                        basePosition=pos,
                        physicsClientId=self.CLIENT
                    )

                    if self.OBSTACLE_TYPE == "moving_cubes":
                        rng = np.random.default_rng(i + 1001)
                        self.moving_obstacle_motion.append({
                            "amplitude": np.array([
                                rng.uniform(0.8, 1.8),
                                rng.uniform(0.8, 1.8),
                                rng.uniform(0.25, 0.75),
                            ]),
                            "frequency": rng.uniform(0.05, 0.25, size=3),
                            "phase": rng.uniform(0.0, 2.0 * np.pi, size=3),
                            "rotation_amplitude": rng.uniform(
                                np.deg2rad(15.0), np.deg2rad(40.0), size=3
                            ),
                            "rotation_frequency": rng.uniform(0.12, 0.35, size=3),
                            "rotation_phase": rng.uniform(0.0, 2.0 * np.pi, size=3),
                        })
                    
                    # Las donas solo se usan con cubos estáticos.
                if self.OBSTACLE_TYPE == "donas":
                    self.dona_ids = self.generar_5_donas(radio_int_min=0.7)
        elif self.OBSTACLE_TYPE == "map":
                # Tu código previo para cargar la malla del mapa .obj
                visual_id = p.createVisualShape(
                    shapeType=p.GEOM_MESH,
                    fileName=pkg_resources.resource_filename('gym_pybullet_drones', 'assets/map.obj'),
                    meshScale=[4, 4, 4],
                    physicsClientId=self.CLIENT
                )

                collision_id = p.createCollisionShape(
                    shapeType=p.GEOM_MESH,
                    fileName=pkg_resources.resource_filename('gym_pybullet_drones', 'assets/map.obj'),
                    meshScale=[4, 4, 4],
                    flags=p.GEOM_FORCE_CONCAVE_TRIMESH,
                    physicsClientId=self.CLIENT
                )

                self.map_id = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=collision_id,
                    baseVisualShapeIndex=visual_id,
                    basePosition=[4 * 4.5, 4 * 4.5, 0.01],
                    baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                    physicsClientId=self.CLIENT
                )
                
        elif self.OBSTACLE_TYPE == "powertrack":
            track = PowerloopTrack( show_labels=True ) 
            track.create()

    def _updateMovingObstacles(self, dt):
        """Move cube obstacles smoothly inside their configured motion margins."""
        if self.OBSTACLE_TYPE != "moving_cubes" or not self.moving_obstacle_motion:
            return

        self.moving_obstacle_time += dt
        for index, obstacle_id in enumerate(self.cubo_id.values()):
            motion = self.moving_obstacle_motion[index]
            angular_position = (
                2.0 * np.pi * motion["frequency"] * self.moving_obstacle_time
                + motion["phase"]
            )
            position = self.moving_obstacle_centers[index] + motion["amplitude"] * np.sin(angular_position)
            angular_rotation = (
                2.0 * np.pi * motion["rotation_frequency"] * self.moving_obstacle_time
                + motion["rotation_phase"]
            )
            orientation = p.getQuaternionFromEuler(
                motion["rotation_amplitude"] * np.sin(angular_rotation)
            )
            p.resetBasePositionAndOrientation(
                int(obstacle_id), position, orientation, physicsClientId=self.CLIENT
            )

                    

    def crear_malla_dona(self, radio_interior, grosor_tubo, num_secciones=24, num_segmentos_tubo=12):
        R = radio_interior + grosor_tubo
        r = grosor_tubo
        vertices = []
        indices = []

        for i in range(num_secciones):
            u = i * 2 * np.pi / num_secciones
            cos_u, sin_u = np.cos(u), np.sin(u)
            for j in range(num_segmentos_tubo):
                v = j * 2 * np.pi / num_segmentos_tubo
                cos_v, sin_v = np.cos(v), np.sin(v)
                x = (R + r * cos_v) * cos_u
                y = (R + r * cos_v) * sin_u
                z = r * sin_v
                vertices.append([x, y, z])

        for i in range(num_secciones):
            i_next = (i + 1) % num_secciones
            for j in range(num_segmentos_tubo):
                j_next = (j + 1) % num_segmentos_tubo
                v1 = i * num_segmentos_tubo + j
                v2 = i_next * num_segmentos_tubo + j
                v3 = i_next * num_segmentos_tubo + j_next
                v4 = i * num_segmentos_tubo + j_next
                indices.extend([v1, v2, v3, v1, v3, v4])

        return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.int32)

    def guardar_obj_temporal(self, vertices, indices, filename="dona_temp.obj"):
        with open(filename, "w") as f:
            for v in vertices:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for i in range(0, len(indices), 3):
                f.write(f"f {indices[i]+1} {indices[i+1]+1} {indices[i+2]+1}\n")
        return filename

    def generar_5_donas(self, radio_int_min=0.7):
        """
        Genera e inserta 5 donas delgadas cóncavas en la simulación de PyBullet.
        """
        donas_ids = []
        if self.randomized :
            posx = lambda : np.random.uniform(-0.5, 0.5)
            posy = lambda : np.random.uniform(-0.5, 0.5)
            posz = lambda : np.random.uniform(-0.5, 0.5)
        else:
            posx = lambda : 0
            posy = lambda : 0
            posz = lambda : 0
        
        
        posicion = [
                    [ 2.4 + posx(),  3.2 + posy(), 2.0 + posz()],
                    [ 2.6 + posx(),  1.5 + posy(), 2.0 + posz()],
                    [ 1.4 + posx(), -1.2 + posy(), 2.0 + posz()],
                    [-5.0 + posx(),  0.4 + posy(), 2.0 + posz()],
                    [ 5.0 + posx(),  0.4 + posy(), 2.0 + posz()],

                    [-2.3 + posx(), -2.3 + posy(), 2.0 + posz()],
                    [ 5.4 + posx(), -1.4 + posy(), 2.0 + posz()],                   
                    [-1.0 + posx(),  2.0 + posy(), 2.0 + posz()],
                ]
        
        for i in range(posicion.__len__()):
            # 1. Parámetros aleatorios (Radio interno >= 3.0, tubo delgado)
            
            if self.randomized :
                radio_interior = np.random.uniform(radio_int_min, radio_int_min + .3)
                grosor_tubo = np.random.uniform(0.05, 0.25)
            else:
                radio_interior = radio_int_min
                grosor_tubo = 0.1

            # 2. Posición espacial aleatoria a lo largo de la ruta del dron
            posicion 

            # 3. Orientación (pitch, roll, yaw) y color RGB aleatorio
            if self.randomized :
                rot_euler = [
                    np.random.uniform(np.pi/2, np.pi/2),
                    np.random.uniform(0, 0),
                    np.random.uniform(0, 2*np.pi)
                ]
                color_rgba = [np.random.random(), np.random.random(), np.random.random(), 1.0]
            else:
                rot_euler = [
                    0,
                    np.pi/2,
                    0
                ]
                color_rgba = [np.random.random(), np.random.random(), np.random.random(), 1.0]


            # 4. Construcción de archivo .obj e importación con malla cóncava
            filename = f"dona_obstaculo_{i}.obj"
            vertices, indices = self.crear_malla_dona(radio_interior, grosor_tubo)
            obj_path = self.guardar_obj_temporal(vertices, indices, filename)

            col_id = p.createCollisionShape(
                shapeType=p.GEOM_MESH,
                fileName=obj_path,
                flags=p.GEOM_FORCE_CONCAVE_TRIMESH
            )
            vis_id = p.createVisualShape(
                shapeType=p.GEOM_MESH,
                fileName=obj_path,
                rgbaColor=color_rgba
            )

            quat = p.getQuaternionFromEuler(rot_euler)
            
            dona_id = p.createMultiBody(
                baseMass=0,  # Estático (masa 0)
                baseCollisionShapeIndex=col_id,
                baseVisualShapeIndex=vis_id,
                basePosition=posicion[i],
                baseOrientation=quat
            )
            donas_ids.append(dona_id)

        return donas_ids
    ################################################################################

    def _actionSpace(self):
        """Returns the action space of the environment.

        Returns
        -------
        spaces.Box
            A Box of size NUM_DRONES x 4, 3, or 1, depending on the action type.

        """
        if self.ACT_TYPE in [ActionType.RPM, ActionType.BRUSHLESS_THRUST, ActionType.VEL]:
            size = 4
        elif self.ACT_TYPE==ActionType.PID:
            size = 3
        elif self.ACT_TYPE in [ActionType.ONE_D_RPM, ActionType.ONE_D_PID]:
            size = 1
        else:
            print("[ERROR] in BaseRLAviary._actionSpace()")
            exit()
        act_lower_bound = np.array([-1*np.ones(size) for i in range(self.NUM_DRONES)])
        act_upper_bound = np.array([+1*np.ones(size) for i in range(self.NUM_DRONES)])
        #
        for i in range(self.ACTION_BUFFER_SIZE):
            self.action_buffer.append(np.zeros((self.NUM_DRONES,size)))
        #
        return spaces.Box(low=act_lower_bound, high=act_upper_bound, dtype=np.float32)

    ################################################################################

    def _preprocessAction(self,
                          action
                          ):
        """Pre-processes the action passed to `.step()` into motors' RPMs.

        Parameter `action` is processed differenly for each of the different
        action types: the input to n-th drone, `action[n]` can be of length
        1, 3, or 4, and represent RPMs, desired thrust and torques, or the next
        target position to reach using PID control.

        Parameter `action` is processed differenly for each of the different
        action types: `action` can be of length 1, 3, or 4 and represent 
        RPMs, desired thrust and torques, the next target position to reach 
        using PID control, a desired velocity vector, etc.

        Parameters
        ----------
        action : ndarray
            The input action for each drone, to be translated into RPMs.

        Returns
        -------
        ndarray
            (NUM_DRONES, 4)-shaped array of ints containing to clipped RPMs
            commanded to the 4 motors of each drone.

        """
        self.action_buffer.append(action)
        if self.ACT_TYPE == ActionType.BRUSHLESS_THRUST and self.step_counter == 0:
            self.brushless_rpm.fill(self.HOVER_RPM)
        
        self.action = action.copy()


        self.prev_action = action.copy()
        rpm = np.zeros((self.NUM_DRONES,4))
        for k in range(action.shape[0]):
            target = action[k, :]
            if self.ACT_TYPE == ActionType.RPM:
                rpm[k,:] = np.array(self.HOVER_RPM * (1+0.05*target))
            elif self.ACT_TYPE == ActionType.BRUSHLESS_THRUST:
                target = np.clip(np.asarray(target, dtype=float), -1.0, 1.0)
                max_motor_thrust = self.MAX_THRUST / 4.0
                hover_motor_thrust = self.GRAVITY / 4.0
                motor_thrust = np.where(
                    target <= 0.0,
                    hover_motor_thrust * (target + 1.0),
                    hover_motor_thrust
                    + (max_motor_thrust - hover_motor_thrust) * target
                )
                target_rpm = np.sqrt(np.maximum(motor_thrust, 0.0) / self.KF)
                alpha = 1.0 - np.exp(-self.CTRL_TIMESTEP / self.BRUSHLESS_MOTOR_TIME_CONSTANT)
                self.brushless_rpm[k, :] += alpha * (target_rpm - self.brushless_rpm[k, :])
                rpm[k, :] = self.brushless_rpm[k, :]
            elif self.ACT_TYPE == ActionType.PID:
                state = self._getDroneStateVector(k)
                next_pos = self._calculateNextStep(
                    current_position=state[0:3],
                    destination=target,
                    step_size=1,
                    )
                rpm_k, _, _ = self.ctrl[k].computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                        cur_pos=state[0:3],
                                                        cur_quat=state[3:7],
                                                        cur_vel=state[10:13],
                                                        cur_ang_vel=state[13:16],
                                                        target_pos=next_pos
                                                        )
                rpm[k,:] = rpm_k
            elif self.ACT_TYPE == ActionType.VEL:
                state = self._getDroneStateVector(k)
                if np.linalg.norm(target[0:3]) != 0:
                    v_unit_vector = target[0:3] / np.linalg.norm(target[0:3])
                else:
                    v_unit_vector = np.zeros(3)
                temp, _, _ = self.ctrl[k].computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                        cur_pos=state[0:3],
                                                        cur_quat=state[3:7],
                                                        cur_vel=state[10:13],
                                                        cur_ang_vel=state[13:16],
                                                        target_pos=state[0:3], # same as the current position
                                                        target_rpy=np.array([0,0,state[9]]), # keep current yaw
                                                        target_vel=self.SPEED_LIMIT * np.abs(target[3]) * v_unit_vector # target the desired velocity vector
                                                        )
                rpm[k,:] = temp
            elif self.ACT_TYPE == ActionType.ONE_D_RPM:
                rpm[k,:] = np.repeat(self.HOVER_RPM * (1+0.05*target), 4)
            elif self.ACT_TYPE == ActionType.ONE_D_PID:
                state = self._getDroneStateVector(k)
                res, _, _ = self.ctrl[k].computeControl(control_timestep=self.CTRL_TIMESTEP,
                                                        cur_pos=state[0:3],
                                                        cur_quat=state[3:7],
                                                        cur_vel=state[10:13],
                                                        cur_ang_vel=state[13:16],
                                                        target_pos=state[0:3]+0.1*np.array([0,0,target[0]])
                                                        )
                rpm[k,:] = res
            else:
                print("[ERROR] in BaseRLAviary._preprocessAction()")
                exit()
        return rpm

    ################################################################################

    def _observationSpace(self):
        if self.OBS_TYPE == ObservationType.RGB:
            return spaces.Box(
                low=0,
                high=255,
                shape=(self.IMG_RES[0], self.IMG_RES[1], 4),
                dtype=np.uint8
            )

        elif self.OBS_TYPE == ObservationType.KIN:
            lo = -np.inf
            hi = np.inf

            # ============================================================
            # CAMBIO: 14 -> 17 variables
            #
            # Antes:
            #   3 u_dir
            #   1 d_norm
            #   4 quaternion
            #   3 linear vel
            #   3 angular vel
            #
            # Ahora:
            #   3 u_dir
            #   1 d_norm
            #   3 gate_normal_local   <-- NUEVO
            #   4 quaternion
            #   3 linear vel
            #   3 angular vel
            # ============================================================

            obs_lower_bound = np.array([
                -1.0, -1.0, -1.0,      # u_dir
                0.0,                    # d_norm
                -1.0, -1.0, -1.0,      # gate_normal_local <-- NUEVO
                -1.0, -1.0, -1.0, -1.0, # quaternion
                lo, lo, lo,             # linear vel
                lo, lo, lo              # angular vel
            ], dtype=np.float32)

            obs_lower_bound = np.tile(
                obs_lower_bound,
                self.NUM_DRONES
            )

            obs_upper_bound = np.array([
                1.0, 1.0, 1.0,         # u_dir
                1.0,                    # d_norm
                1.0, 1.0, 1.0,         # gate_normal_local <-- NUEVO
                1.0, 1.0, 1.0, 1.0,    # quaternion
                hi, hi, hi,             # linear vel
                hi, hi, hi              # angular vel
            ], dtype=np.float32)

            obs_upper_bound = np.tile(
                obs_upper_bound,
                self.NUM_DRONES
            )

            act_lo = -1
            act_hi = 1

            lidar_lo = -1
            lidar_hi = 1

            lidar_size = 128

            self.lidar_buffer.clear()

            # Action history
            for _ in range(self.ACTION_BUFFER_SIZE):
                if self.ACT_TYPE == ActionType.RPM:
                    action_size = 4
                elif self.ACT_TYPE == ActionType.PID:
                    action_size = 4
                elif self.ACT_TYPE == ActionType.VEL:
                    action_size = 4
                elif self.ACT_TYPE == ActionType.ONE_D_RPM:
                    action_size = 1
                elif self.ACT_TYPE == ActionType.ONE_D_PID:
                    action_size = 1
                else:
                    action_size = 4

                obs_lower_bound = np.hstack([
                    obs_lower_bound,
                    np.full(
                        self.NUM_DRONES * action_size,
                        act_lo,
                        dtype=np.float32
                    )
                ])

                obs_upper_bound = np.hstack([
                    obs_upper_bound,
                    np.full(
                        self.NUM_DRONES * action_size,
                        act_hi,
                        dtype=np.float32
                    )
                ])

            # Lidar history
            for _ in range(self.LIDAR_BUFFER_SIZE):
                obs_lower_bound = np.hstack([
                    obs_lower_bound,
                    np.full(
                        self.NUM_DRONES * lidar_size,
                        lidar_lo,
                        dtype=np.float32
                    )
                ])

                obs_upper_bound = np.hstack([
                    obs_upper_bound,
                    np.full(
                        self.NUM_DRONES * lidar_size,
                        lidar_hi,
                        dtype=np.float32
                    )
                ])

            return spaces.Box(
                low=obs_lower_bound,
                high=obs_upper_bound,
                dtype=np.float32
            )
                ############################################################
        else:
            print("[ERROR] in BaseRLAviary._observationSpace()")
    
    ################################################################################


    def _computeObs(self):
        if self.OBS_TYPE == ObservationType.RGB:
            return self._getDroneImages()

        elif self.OBS_TYPE == ObservationType.KIN:

            # ============================================================
            # CAMBIO: 14 -> 17
            # ============================================================
            obs_17 = np.zeros(
                (self.NUM_DRONES, 17),
                dtype=np.float32
            )

            current_lidar = self.lidar.copy()

            if self.enable_noise:
                current_lidar = self.noise_model.apply_laser_noise(
                    current_lidar
                )

            for i in range(self.NUM_DRONES):

                obs = self._getDroneStateVector(i)

                quat_world = obs[3:7]

                current_v_global = obs[10:13]

                ang_vel_global = obs[13:16]

                rot_matrix = np.array(
                    p.getMatrixFromQuaternion(quat_world)
                ).reshape(3, 3)

                # ========================================================
                # TARGET POSITION
                # ========================================================

                if hasattr(self, 'TARGET_POS'):

                    if self.NUM_DRONES == 1:
                        delta_pos = self.TARGET_POS - obs[0:3]

                    else:
                        delta_pos = self.TARGET_POS[i] - obs[0:3]

                else:
                    delta_pos = np.zeros(3)

                dist_total = np.linalg.norm(delta_pos)

                d_min = 0.05

                if dist_total < d_min:
                    u_unit_global = np.zeros(3)

                else:
                    u_unit_global = delta_pos / dist_total

                if np.linalg.norm(u_unit_global[:2]) < 0.08:

                    u_unit_global[0] = 0
                    u_unit_global[1] = 0
                    u_unit_global[2] = np.sign(
                        u_unit_global[2]
                    )

                # Dirección hacia el target expresada en frame local
                u_unit_local = rot_matrix.T @ u_unit_global

                # ========================================================
                # CAMBIO NUEVO:
                # NORMAL DE LA PUERTA EN FRAME LOCAL
                #
                # self.gate_normals contiene la normal de cada gate
                # expresada en coordenadas globales.
                #
                # Se transforma usando la misma rotación que ya utilizas
                # para velocidad y dirección.
                # ========================================================

                if (
                    hasattr(self, 'gate_normals')
                    and hasattr(self, 'current_gate_idx')
                    and self.current_gate_idx < len(self.gate_normals)
                ):

                    gate_normal_global = np.array(
                        self.gate_normals[self.current_gate_idx],
                        dtype=np.float32
                    )

                    gate_normal_local = (
                        rot_matrix.T @ gate_normal_global
                    )

                    # Normalización por seguridad
                    gate_normal_norm = np.linalg.norm(
                        gate_normal_local
                    )

                    if gate_normal_norm > 1e-6:
                        gate_normal_local /= gate_normal_norm

                    else:
                        gate_normal_local = np.zeros(
                            3,
                            dtype=np.float32
                        )

                else:
                    # Para benchmarks que todavía no tengan normales
                    gate_normal_local = np.zeros(
                        3,
                        dtype=np.float32
                    )
                # print("gate_normal_local ", gate_normal_local)
                
                # ============================================================
                # FLECHA DE LA NORMAL DE LA PUERTA
                # ============================================================
                if self.GUI :
                    if np.linalg.norm(gate_normal_local) > 1e-6:

                        # Normal local -> mundo
                        gate_normal_world = (
                            gate_normal_local
                        )

                        gate_normal_world /= (
                            np.linalg.norm(gate_normal_world) + 1e-8
                        )

                        arrow_length = 1.5
                        arrow_head_length = 0.35
                        arrow_head_width = 0.20

                        start = obs[0:3]
                        end = (
                            obs[0:3]
                            + arrow_length * gate_normal_world
                        )

                        # Dirección hacia atrás para las puntas
                        backward = (
                            -gate_normal_world
                            * arrow_head_length
                        )

                        # Vector perpendicular para abrir las puntas
                        reference = np.array([
                            0.0,
                            0.0,
                            1.0
                        ])

                        side = np.cross(
                            gate_normal_world,
                            reference
                        )

                        side_norm = np.linalg.norm(side)

                        if side_norm < 1e-6:
                            reference = np.array([
                                1.0,
                                0.0,
                                0.0
                            ])

                            side = np.cross(
                                gate_normal_world,
                                reference
                            )

                            side_norm = np.linalg.norm(side)

                        side /= side_norm

                        arrow_head_1 = (
                            end
                            + backward
                            + arrow_head_width * side
                        )

                        arrow_head_2 = (
                            end
                            + backward
                            - arrow_head_width * side
                        )

                        # --------------------------------------------------------
                        # Crear o actualizar las 3 líneas
                        # --------------------------------------------------------

                        lines = [
                            (start, end),
                            (end, arrow_head_1),
                            (end, arrow_head_2)
                        ]

                        for j, (line_start, line_end) in enumerate(lines):

                            if self.gate_normal_arrow_ids[j] == -1:

                                self.gate_normal_arrow_ids[j] = (
                                    p.addUserDebugLine(
                                        line_start,
                                        line_end,
                                        lineColorRGB=[1, 0, 0],
                                        lineWidth=4.0,
                                        lifeTime=0
                                    )
                                )

                            else:

                                p.addUserDebugLine(
                                    line_start,
                                    line_end,
                                    lineColorRGB=[1, 0, 0],
                                    lineWidth=4.0,
                                    lifeTime=0,
                                    replaceItemUniqueId=(
                                        self.gate_normal_arrow_ids[j]
                                    )
                                )
                # ========================================================
                # VELOCIDADES LOCALES
                # ========================================================

                current_v_local = (
                    rot_matrix.T @ current_v_global
                )

                vel_angle_local = (
                    rot_matrix.T @ ang_vel_global
                )

                # ========================================================
                # QUATERNION LOCAL
                # ========================================================

                r_world = R.from_quat(quat_world)

                _, _, yaw = r_world.as_euler('xyz')

                if (
                    not hasattr(self, 'initial_yaws')
                    or self.step_counter == 0
                ):

                    self.initial_yaws = [
                        R.from_quat(
                            self._getDroneStateVector(j)[3:7]
                        ).as_euler('xyz')[2]
                        for j in range(self.NUM_DRONES)
                    ]

                q_yaw_inv = R.from_euler(
                    'z',
                    -yaw
                )

                q_local = (
                    q_yaw_inv * r_world
                ).as_quat()

                if q_local[3] < 0:
                    q_local = -q_local

                # ========================================================
                # NORMALIZACIÓN DE DISTANCIA
                # ========================================================

                S = 5.0

                d_norm = np.tanh(
                    dist_total / S
                )

                # ========================================================
                # NOISE
                # ========================================================

                if self.enable_noise:

                    u_unit_local, current_v_local = (
                        self.noise_model.apply_odometry_noise(
                            u_unit_local,
                            current_v_local
                        )
                    )

                    u_norm = np.linalg.norm(
                        u_unit_local
                    )

                    if u_norm > 1e-6:
                        u_unit_local /= u_norm

                    _, vel_angle_local, _ = (
                        self.noise_model.apply_imu_noise(
                            acc=np.zeros(3),
                            gyro=vel_angle_local,
                            angles=None
                        )
                    )

                    euler_local = R.from_quat(
                        q_local
                    ).as_euler('xyz')

                    _, _, euler_local = (
                        self.noise_model.apply_imu_noise(
                            acc=np.zeros(3),
                            gyro=np.zeros(3),
                            angles=euler_local
                        )
                    )

                    q_local = R.from_euler(
                        'xyz',
                        euler_local
                    ).as_quat()

                    if q_local[3] < 0:
                        q_local = -q_local

                # ========================================================
                # CAMBIO: OBSERVACIÓN FINAL 17 ELEMENTOS
                # ========================================================

                obs_17[i, :] = np.hstack([
                    u_unit_local,       # 3
                    d_norm,             # 1
                    gate_normal_local,  # 3 <-- NUEVO
                    q_local,            # 4
                    current_v_local,    # 3
                    vel_angle_local     # 3
                ]).reshape(17,)

            # ============================================================
            # LIDAR BUFFER
            # ============================================================

            self.lidar_buffer.append(
                current_lidar.copy()
            )

            ret = np.array([
                obs_17[i, :]
                for i in range(self.NUM_DRONES)
            ]).astype('float32')

            # ============================================================
            # ACTION BUFFER
            # ============================================================

            for action in self.action_buffer:

                ret = np.hstack([
                    ret,
                    action
                ])

            # ============================================================
            # LIDAR BUFFER
            # ============================================================

            for lidar in self.lidar_buffer:

                ret = np.hstack([
                    ret,
                    lidar
                ])

            if self.NUM_DRONES == 1:
                return ret[0]

            return ret
            ############################################################
        else:
            print("[ERROR] in BaseRLAviary._computeObs()")
