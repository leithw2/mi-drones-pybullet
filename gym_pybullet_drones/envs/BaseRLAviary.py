import os
import numpy as np
import pybullet as p
import pkg_resources
from scipy.spatial.transform import Rotation as R
from gymnasium import spaces
from collections import deque


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
                 act: ActionType=ActionType.RPM
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
        ####
        # Initialize TARGET_POS to avoid attribute errors
        self.TARGET_POS = np.zeros(3) if num_drones == 1 else np.zeros((num_drones, 3))
        vision_attributes = True if obs == ObservationType.RGB else False
        self.OBS_TYPE = obs
        self.ACT_TYPE = act
        self.prev_v = np.zeros( 3) # placeholder for previous velocity to calculate acceleration for the IMU readings in the kinematic observation space, initialized at zero
        
        self.posBo = [None for i in range(15)] # placeholder for obstacle positions in case we want to remove them later
        self.cubo_id = [None for i in range(15)] # placeholder for obstacle ids in case we want to remove them later
        
        #### Create integrated controllers #########################
        if act in [ActionType.PID, ActionType.VEL, ActionType.ONE_D_PID]:
            os.environ['KMP_DUPLICATE_LIB_OK']='True'
            if drone_model in [DroneModel.CF2X, DroneModel.CF2P]:
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
                         )
        #### Set a limit on the maximum target speed ###############
        if act == ActionType.VEL:
            self.SPEED_LIMIT = 0.03 * self.MAX_SPEED_KMH * (1000/3600)
        
        self.prev_action = None
        self.max_action_delta = 0.05
        self.smooth_lambda = 0.20

    ################################################################################
    def _addObstacles(self):
        """Add obstacles aligned with the Lemniscate path for ToF RL training."""
        randoms_obstacles = True
        if self.OBSTACLES:
            if randoms_obstacles:
                # 1. Definir posiciones según las 3 Zonas Estratégicas:
                # ZONA 1 (Intersección Central): Cerca del centro (0,0) pero desfasado
                # ZONA 2 (Pasillos en las curvas): Pares laterales
                # ZONA 3 (Bloqueos en la curva): Sobre el trazo
                
                self.posBo = [
                    [1.4,  -1.2, 1.8],  # Zona 2: Pasillo Curva Derecha (Pared interna)
                    [6.4,  1.4, 0.0],  # Zona 2: Pasillo Curva Derecha (Pared interna)
                    [6.4, -1.4, 1.8*0.1],  # Zona 2: Pasillo Curva Derecha (Pared externa)
                    [-6.8, 0.0, 1.8],  # Zona 3: Bloqueo Directo Curva Izquierda
                    [0.0,  1.5, 1.8],   # Zona 3: Bloqueo Directo Curva Superior
                    
                    [ 2.3, 2.3, 1.8],  # Zona 2: Pasillo Curva Derecha (Pared interna)
                                        
                    [ -2.3, -2.3, 1.8],  # Zona 2: Pasillo Curva Derecha (Pared interna)

                    [-2.0, 1.4, 1.8*1.4],  # Zona 2: Pasillo Curva Derecha (Pared interna)
                    [ 5.0, 0.4, 1.8],  # Zona 2: Pasillo Curva Derecha (Pared externa)
                    [-5.0, 0.4, 1.8*1.3],  # Zona 3: Bloqueo Directo Curva Izquierda
                    [ 0.0, 1.5, 1.8]   # Zona 3: Bloqueo Directo Curva Superior
                ]
                
                # Inicializar diccionario/lista de IDs si no existe
                self.cubo_id = {} if not hasattr(self, 'cubo_id') else self.cubo_id
                
                # 2. Crear las formas en PyBullet
                # halfExtents=[0.15, 0.15, 1.8] genera una columna de 30x30 cm x 3.6m de alto
                col_id = p.createCollisionShape(
                    p.GEOM_BOX, 
                    halfExtents=[0.15, 0.15, 0.9], 
                    physicsClientId=self.CLIENT
                )
                vis_id = p.createVisualShape(
                    p.GEOM_BOX, 
                    halfExtents=[0.15, 0.15, 0.9], 
                    rgbaColor=[0.8, 0.2, 0.2, 1], 
                    physicsClientId=self.CLIENT
                )
                
                # 3. Crear los cuerpos estáticos en la simulación
                for i, pos in enumerate(self.posBo):
                    self.cubo_id[i] = p.createMultiBody(
                        baseMass=0,  # <--- CRÍTICO: 0 lo hace estático (inamovible al chocar)
                        baseCollisionShapeIndex=col_id,
                        baseVisualShapeIndex=vis_id,
                        basePosition=pos,
                        physicsClientId=self.CLIENT
                    )
                    
            else:
                # Tu código previo para cargar la malla del mapa .obj
                visual_id = p.createVisualShape(
                    shapeType=p.GEOM_MESH,
                    fileName=pkg_resources.resource_filename('gym_pybullet_drones', 'assets/map.obj'),
                    meshScale=[2, 2, 2],
                    physicsClientId=self.CLIENT
                )

                collision_id = p.createCollisionShape(
                    shapeType=p.GEOM_MESH,
                    fileName=pkg_resources.resource_filename('gym_pybullet_drones', 'assets/map.obj'),
                    meshScale=[2, 2, 2],
                    flags=p.GEOM_FORCE_CONCAVE_TRIMESH,
                    physicsClientId=self.CLIENT
                )

                mapa_id = p.createMultiBody(
                    baseMass=0,
                    baseCollisionShapeIndex=collision_id,
                    baseVisualShapeIndex=visual_id,
                    basePosition=[2 * 4.5, 2 * 4.5, 0.01],
                    baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                    physicsClientId=self.CLIENT
                )
    # def _addObstacles(self):
    #     """Add obstacles to the environment.

    #     Only if the observation is of type RGB, 4 landmarks are added.
    #     Overrides BaseAviary's method.

    #     """
                    
    #     randoms_obstacles= True
    #     if self.OBSTACLES:
    #         if randoms_obstacles:

    #             self.posBo[0] = [1.8,0,1.8]
    #             self.posBo[1]  = [-1.8,0,1.8]
    #             self.posBo[2]  = [0,1.8,1.8]
    #             self.posBo[3]  = [0,-1.8,1.8]
    #             self.posBo[4]  = [10,0,1.8]
                
                
    #             #print("obstaculos!!!!!!!!!!!!!!")
    #             # create multiple random boxes in the environment
    #             col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.35, 0.35, 1.8], physicsClientId=self.CLIENT)
    #             vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.35, 0.35, 1.8], rgbaColor=[0.8, 0.2, 0.2, 1], physicsClientId=self.CLIENT)
    #             for i in range(len(self.posBo)):
    #                 self.cubo_id[i] = p.createMultiBody(baseMass=1, # 0 lo hace estático e inamovible
    #                             baseCollisionShapeIndex=col_id,
    #                             baseVisualShapeIndex=vis_id,
    #                             #basePosition=[np.random.uniform(1, 3.5) * np.random.choice([-1, 1]), np.random.uniform(1, 3.5) * np.random.choice([-1, 1]), 0.8],
    #                             basePosition=self.posBo[i],
    #                             physicsClientId=self.CLIENT)

                
    #         else:
    #             # 1. Crear la forma visual
    #             visual_id = p.createVisualShape(
    #                 shapeType=p.GEOM_MESH,
    #                 fileName=pkg_resources.resource_filename('gym_pybullet_drones', 'assets/map.obj'),
    #                 meshScale=[2, 2, 2]
    #             )

    #             # 2. Crear la forma de colisión FORZANDO malla cóncava (Trimesh)
    #             collision_id = p.createCollisionShape(physicsClientId=self.CLIENT,
    #                 shapeType=p.GEOM_MESH,
    #                 fileName=pkg_resources.resource_filename('gym_pybullet_drones', 'assets/map.obj'),
    #                 meshScale=[2, 2, 2],
    #                 flags=p.GEOM_FORCE_CONCAVE_TRIMESH  # <--- ESTO SOLUCIONA EL "AIRE SÓLIDO"
    #             )

    #             # 3. Crear el cuerpo en el mundo
    #             mapa_id = p.createMultiBody(physicsClientId=self.CLIENT,
    #                 baseMass=0,
    #                 baseCollisionShapeIndex=collision_id,
    #                 baseVisualShapeIndex=visual_id,
    #                 basePosition=[2*4.5, 2*4.5, 0.01],
    #                 baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
    #             )
    ################################################################################

    def _actionSpace(self):
        """Returns the action space of the environment.

        Returns
        -------
        spaces.Box
            A Box of size NUM_DRONES x 4, 3, or 1, depending on the action type.

        """
        if self.ACT_TYPE in [ActionType.RPM, ActionType.VEL]:
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
        
        self.action = action.copy()   # ← AÑADE ESTA LÍNEA

        rpm = np.zeros((self.NUM_DRONES,4))

        self.prev_action = action.copy()
        rpm = np.zeros((self.NUM_DRONES,4))
        for k in range(action.shape[0]):
            target = action[k, :]
            if self.ACT_TYPE == ActionType.RPM:
                rpm[k,:] = np.array(self.HOVER_RPM * (1+0.05*target))
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
        """Returns the observation space of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (NUM_DRONES,H,W,4) or (NUM_DRONES,12) depending on the observation type.

        """
        if self.OBS_TYPE == ObservationType.RGB:
            return spaces.Box(low=0,
                              high=255,
                              shape=(self.NUM_DRONES, self.IMG_RES[1], self.IMG_RES[0], 4), dtype=np.uint8)
        elif self.OBS_TYPE == ObservationType.KIN:
            # OBS SPACE OF SIZE
            lo = -np.inf
            hi = np.inf
            obs_lower_bound = np.array([
                -1.0, -1.0, -1.0,  # u_dir (x, y, z): vector unitario
                0.0,              # d_norm: normalizado entre 0 y 1 con tanh
                -1.0, -1.0, -1.0, -1.0,  # quaternion (x, y, z, w)
                lo,   lo,   lo,   # velocidad lineal (vx, vy, vz)
                lo,   lo,   lo    # velocidad angular (wx, wy, wz)
            ], dtype=np.float32)
            obs_lower_bound = np.tile(obs_lower_bound, (self.NUM_DRONES, 1))

            # Definición del vector de límites superiores (14 elementos)
            obs_upper_bound = np.array([
                1.0,  1.0,  1.0,  # u_dir
                1.0,              # d_norm
                1.0,  1.0,  1.0,  1.0,  # quaternion (x, y, z, w)
                hi,   hi,   hi,   # velocidad lineal
                hi,   hi,   hi    # velocidad angular
            ], dtype=np.float32)
            obs_upper_bound = np.tile(obs_upper_bound, (self.NUM_DRONES, 1))
            # Add action buffer to observation space
            act_lo = -1
            act_hi = +1
            lidar_lo = -1
            lidar_hi = 1
            lidar_size = 128
            self.lidar_buffer.clear()
            for i in range(self.LIDAR_BUFFER_SIZE):
                self.lidar_buffer.append(np.zeros((self.NUM_DRONES, lidar_size)))

            for i in range(self.ACTION_BUFFER_SIZE):
                if self.ACT_TYPE in [ActionType.RPM, ActionType.VEL]:
                    action_size = 4
                elif self.ACT_TYPE==ActionType.PID:
                    action_size = 3
                elif self.ACT_TYPE in [ActionType.ONE_D_RPM, ActionType.ONE_D_PID]:
                    action_size = 1
                obs_lower_bound = np.hstack([obs_lower_bound, np.full((self.NUM_DRONES, action_size), act_lo)])
                obs_upper_bound = np.hstack([obs_upper_bound, np.full((self.NUM_DRONES, action_size), act_hi)])

            # Add the 8x8 lidar history (64 readings per buffer step).
            for i in range(self.LIDAR_BUFFER_SIZE):
                obs_lower_bound = np.hstack([obs_lower_bound, np.full((self.NUM_DRONES, lidar_size), lidar_lo)])
                obs_upper_bound = np.hstack([obs_upper_bound, np.full((self.NUM_DRONES, lidar_size), lidar_hi)])
            return spaces.Box(low=obs_lower_bound, high=obs_upper_bound, dtype=np.float32)
            ############################################################
        else:
            print("[ERROR] in BaseRLAviary._observationSpace()")
    
    ################################################################################


    def _computeObs(self):
        """Returns the current observation of the environment.

        Returns
        -------
        ndarray
            A Box() of shape (NUM_DRONES,H,W,4) or (NUM_DRONES,14) depending on the observation type.
        """
        if self.OBS_TYPE == ObservationType.RGB:
            if self.step_counter % self.IMG_CAPTURE_FREQ == 0:
                for i in range(self.NUM_DRONES):
                    self.rgb[i], self.dep[i], self.seg[i] = self._getDroneImages(i, segmentation=False)
                    
                    if self.RECORD:
                        self._exportImage(img_type=ImageType.RGB,
                                        img_input=self.rgb[i],
                                        path=self.ONBOARD_IMG_PATH + "drone_" + str(i),
                                        frame_num=int(self.step_counter / self.IMG_CAPTURE_FREQ)
                                        )
            return np.array([self.rgb[i] for i in range(self.NUM_DRONES)]).astype('float32')

        elif self.OBS_TYPE == ObservationType.KIN:
            obs_14 = np.zeros((self.NUM_DRONES, 14))
            
            for i in range(self.NUM_DRONES):
                obs = self._getDroneStateVector(i)
                # --- Guardar Yaw de inicio (poner antes del loop de drones o dentro de 'if self.step_counter == 0') ---
                
                
                # 1. Orientación Global (Cuaternión de PyBullet: [x, y, z, w])
                quat_world = obs[3:7]
                current_v_global = obs[10:13]

                # 2. Matriz de Rotación Body -> World
                rot_matrix = np.array(p.getMatrixFromQuaternion(quat_world)).reshape(3, 3)

                # 3. Calcular Vector de Posición Relativa al Objetivo
                if hasattr(self, 'TARGET_POS'):
                    if self.NUM_DRONES == 1:
                        delta_pos = self.TARGET_POS - obs[0:3]
                    else:
                        delta_pos = self.TARGET_POS[i] - obs[0:3]
                else:
                    delta_pos = np.zeros(3)

                dist_total = np.linalg.norm(delta_pos)

                # 4. Vector unitario hacia el objetivo en marco GLOBAL
                d_min = 0.05  # Zona muerta de 5 cm
                if dist_total < d_min:
                    u_unit_global = np.zeros(3)
                else:
                    u_unit_global = delta_pos / dist_total


                # Eliminar ruido horizontal si el movimiento es puramente en Z
                if np.linalg.norm(u_unit_global[:2]) < 0.08:
                    u_unit_global[0] = 0.0
                    u_unit_global[1] = 0.0
                    u_unit_global[2] = np.sign(u_unit_global[2])
                # --- TRANSFORMACIONES A MARCO LOCAL (BODY FRAME) ---
                
                # A. Dirección al objetivo en marco LOCAL
                u_unit_local = rot_matrix.T @ u_unit_global

                # B. Velocidad lineal en marco LOCAL (Avanzar, Lateral, Vertical)
                current_v_local = rot_matrix.T @ current_v_global

                # C. CUATERNIÓN LOCAL (Invariante al Yaw del Mapa)
                # Extraer Euler global
                r_world = R.from_quat(quat_world)
                _, _, yaw = r_world.as_euler('xyz')
                if not hasattr(self, 'initial_yaws') or self.step_counter == 0:
                                    self.initial_yaws = [R.from_quat(self._getDroneStateVector(j)[3:7]).as_euler('xyz')[2] for j in range(self.NUM_DRONES)]
                
                # --- Dentro del loop 'for i in range(self.NUM_DRONES):' ---
                yaw_inc_deg = np.degrees(np.arctan2(np.sin(yaw - self.initial_yaws[i]), np.cos(yaw - self.initial_yaws[i])))

                # --- Print a añadir ---
                #print(f"Yaw Incremental (Spawn): {yaw_inc_deg:.2f}°")
                # Inverso del Yaw absoluto (para 'cancelar' la rotación de brújula)
                q_yaw_inv = R.from_euler('z', -yaw)

                # Cuaternión local (solo contiene Roll y Pitch)
                q_local = (q_yaw_inv * r_world).as_quat()  # Retorna [x, y, z, w]

                # Corrección de "Sign Flip" (Asegurar continuidad de q == -q en RL)
                # Mantiene el escalar w siempre positivo
                if q_local[3] < 0:
                    q_local = -q_local

                # D. Distancia escalar normalizada [0, 1)
                S = 5.0
                d_norm = np.tanh(dist_total / S)

                # E. Velocidades Angulares (PQR)
                pqr_local = obs[13:16]
                # --- IMPRESIONES DE VERIFICACIÓN DE ESTADO LOCAL ---
                azimuth_local_deg = np.degrees(np.arctan2(u_unit_local[1], u_unit_local[0]))
                elevation_local_deg = np.degrees(np.arcsin(np.clip(u_unit_local[2], -1.0, 1.0)))

                # print(f"\n--- [DRONE {i}] ESTADO LOCAL ---")
                # print(f"Distancia Real:         {dist_total:.3f} m")
                # print(f"u_unit_local:           [{u_unit_local[0]:.3f}, {u_unit_local[1]:.3f}, {u_unit_local[2]:.3f}]")
                # print(f"Azimut Local (Objetivo):{azimuth_local_deg:.2f}°")
                # print(f"Elevación Local:        {elevation_local_deg:.2f}°")
                # print(f"Yaw Incremental (Spawn):{yaw_inc_deg:.2f}°")  # <-- NUEVO PRINT
                # print(f"q_local [x,y,z,w]:      [{q_local[0]:.3f}, {q_local[1]:.3f}, {q_local[2]:.3f}, {q_local[3]:.3f}]")
                # print(f"v_local [fwd,lat,ver]:  [{current_v_local[0]:.3f}, {current_v_local[1]:.3f}, {current_v_local[2]:.3f}] m/s")
                # --- CONSTRUCCIÓN DEL VECTOR DE 14 ELEMENTOS LOCAL Y CONTINUO ---
                # Indices:
                # [0:3]   -> u_unit_local (Dirección al objetivo)
                # [3]     -> d_norm (Distancia normalizada)
                # [4:8]   -> q_local (Cuaternión de actitud local [x, y, z, w])
                # [8:11]  -> current_v_local (Velocidad lineal local)
                # [11:14] -> pqr_local (Velocidad angular local)
                obs_14[i, :] = np.hstack([
                    u_unit_local,      # 3
                    d_norm,            # 1
                    q_local,           # 4
                    current_v_local,   # 3
                    pqr_local          # 3
                ]).reshape(14,)
                
                # print(f"obs_14[{i}]: {obs_14[i, :]}")

            self.lidar_buffer.append(self.lidar.copy())
            
            ret = np.array([obs_14[i, :] for i in range(self.NUM_DRONES)]).astype('float32')

            # Agregar buffers de acción y LiDAR a la observación
            for i in range(self.ACTION_BUFFER_SIZE):
                ret = np.hstack([ret, np.array([self.action_buffer[i][j, :] for j in range(self.NUM_DRONES)])])
            for i in range(self.LIDAR_BUFFER_SIZE):
                ret = np.hstack([ret, np.array([self.lidar_buffer[i][j, :] for j in range(self.NUM_DRONES)])])
                
            return ret
            ############################################################
        else:
            print("[ERROR] in BaseRLAviary._computeObs()")
