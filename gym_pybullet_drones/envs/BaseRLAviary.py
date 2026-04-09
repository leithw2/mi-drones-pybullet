import os
import numpy as np
import pybullet as p
import pkg_resources

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
        self.OBSERVATION_BUFFER_SIZE = int(2)
        self.action_buffer = deque(maxlen=self.ACTION_BUFFER_SIZE)
        self.lidar_buffer = deque(maxlen=self.LIDAR_BUFFER_SIZE)
        self.observation_buffer = deque(maxlen=self.OBSERVATION_BUFFER_SIZE)
        ####
        # Initialize TARGET_POS to avoid attribute errors
        self.TARGET_POS = np.zeros(3) if num_drones == 1 else np.zeros((num_drones, 3))
        vision_attributes = True if obs == ObservationType.RGB else False
        self.OBS_TYPE = obs
        self.ACT_TYPE = act
        
        self.posBo = [None for i in range(4)] # placeholder for obstacle positions in case we want to remove them later
        self.cubo_id = [None for i in range(4)] # placeholder for obstacle ids in case we want to remove them later
        
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

    ################################################################################

    def _addObstacles(self):
        """Add obstacles to the environment.

        Only if the observation is of type RGB, 4 landmarks are added.
        Overrides BaseAviary's method.

        """
                    
        randoms_obstacles= True
        if self.OBSTACLES:
            if randoms_obstacles:
                # p.loadURDF("block.urdf",
                #         [np.random.uniform(1.5, 2) * np.random.choice([-1, 1]), np.random.uniform(1.5, 2) * np.random.choice([-1, 1]), 0.5],
                #         p.getQuaternionFromEuler([0, 0, 0]),
                #         physicsClientId=self.CLIENT, globalScaling=30
                #       )
                
                # Definir la colisión (1 metro de lado = halfExtents de 0.5)
                # self.posBo[0] = [-2,-2,.8]
                # self.posBo[1]  = [2,-2,.8]
                # self.posBo[2]  = [-2,2,.8]
                # self.posBo[3]  = [2,2,.8]
                # self.posBo[0] = [2.5,-2.5,1.8]
                # self.posBo[1]  = [-30,-30,1.8]
                # self.posBo[2]  = [-2.5,2.5,1.8]
                # self.posBo[3]  = [30,30,1.8]
                self.posBo[0] = [4,2,1.8]
                self.posBo[1]  = [6,3,1.8]
                self.posBo[2]  = [3,5,1.8]
                self.posBo[3]  = [6,6,1.8]
                
                
                #print("obstaculos!!!!!!!!!!!!!!")
                # create multiple random boxes in the environment
                col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.4, 0.4, 1.8], physicsClientId=self.CLIENT)
                vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.4, 0.4, 1.8], rgbaColor=[0.8, 0.2, 0.2, 1], physicsClientId=self.CLIENT)
                for i in range(len(self.posBo)):
                    self.cubo_id[i] = p.createMultiBody(baseMass=1, # 0 lo hace estático e inamovible
                                baseCollisionShapeIndex=col_id,
                                baseVisualShapeIndex=vis_id,
                                #basePosition=[np.random.uniform(1, 3.5) * np.random.choice([-1, 1]), np.random.uniform(1, 3.5) * np.random.choice([-1, 1]), 0.8],
                                basePosition=self.posBo[i],
                                physicsClientId=self.CLIENT)
                
                # col_id2 = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.4, 0.4, .3], physicsClientId=self.CLIENT)
                # vis_id2 = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.4, 0.4, .3], rgbaColor=[0.8, 0.2, 0.2, 1], physicsClientId=self.CLIENT)
                # p.createMultiBody(baseMass=1, # 0 lo hace estático e inamovible
                #                 baseCollisionShapeIndex=col_id2,
                #                 baseVisualShapeIndex=vis_id2,
                #                 #basePosition=[np.random.uniform(1, 3.5) * np.random.choice([-1, 1]), np.random.uniform(1, 3.5) * np.random.choice([-1, 1]), 0.8],
                #                 basePosition=[1,1,.3],
                #                 physicsClientId=self.CLIENT)
                
                # col_id3 = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.4, 0.4, .4], physicsClientId=self.CLIENT)
                # vis_id3 = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.4, 0.4, .4], rgbaColor=[0.8, 0.2, 0.2, 1], physicsClientId=self.CLIENT)
                # p.createMultiBody(baseMass=1, # 0 lo hace estático e inamovible
                #                 baseCollisionShapeIndex=col_id3,
                #                 baseVisualShapeIndex=vis_id3,
                #                 #basePosition=[np.random.uniform(1, 3.5) * np.random.choice([-1, 1]), np.random.uniform(1, 3.5) * np.random.choice([-1, 1]), 0.8],
                #                 basePosition=[-1,-1,.4],
                #                 physicsClientId=self.CLIENT)
                
                # p.loadURDF("cube_small.urdf",
                #     [np.random.uniform(2.5, 3),np.random.uniform(2.5, 3), 0.5],
                #     p.getQuaternionFromEuler([0, 0, 0]),
                #     physicsClientId=self.CLIENT, globalScaling=20
                #     )
                # p.loadURDF("cube_small.urdf",
                #     [np.random.uniform(3.5, 4),np.random.uniform(3.5, 4), 0.5],
                #     p.getQuaternionFromEuler([0, 0, 0]),
                #     physicsClientId=self.CLIENT, globalScaling=20
                #     )
                # p.loadURDF("cube_small.urdf",
                #     [np.random.uniform(4.5, 5.5),np.random.uniform(4.5, 5.5), 0.5],
                #     p.getQuaternionFromEuler([0, 0, 0]),
                #     physicsClientId=self.CLIENT, globalScaling=20
                #     )
                # p.loadURDF("duck_vhacd.urdf",
                #         [np.random.uniform(1.5, 2) * np.random.choice([-1, 1]), np.random.uniform(1.5, 2) * np.random.choice([-1, 1]), 0.5],
                #         p.getQuaternionFromEuler([1, 0, 0]),
                #         physicsClientId=self.CLIENT, globalScaling=20
                #       )
                # p.loadURDF("teddy_vhacd.urdf",
                #            [ np.random.uniform(1.5, 2) * np.random.choice([-1, 1]), np.random.uniform(1.5, 2) * np.random.choice([-1, 1]), 0.5],
                #            p.getQuaternionFromEuler([2, 0, 0]),
                #            physicsClientId=self.CLIENT, globalScaling=20
                #            )

                
            else:
                # 1. Crear la forma visual
                visual_id = p.createVisualShape(
                    shapeType=p.GEOM_MESH,
                    fileName=pkg_resources.resource_filename('gym_pybullet_drones', 'assets/map.obj'),
                    meshScale=[2, 2, 2]
                )

                # 2. Crear la forma de colisión FORZANDO malla cóncava (Trimesh)
                collision_id = p.createCollisionShape(physicsClientId=self.CLIENT,
                    shapeType=p.GEOM_MESH,
                    fileName=pkg_resources.resource_filename('gym_pybullet_drones', 'assets/map.obj'),
                    meshScale=[2, 2, 2],
                    flags=p.GEOM_FORCE_CONCAVE_TRIMESH  # <--- ESTO SOLUCIONA EL "AIRE SÓLIDO"
                )

                # 3. Crear el cuerpo en el mundo
                mapa_id = p.createMultiBody(physicsClientId=self.CLIENT,
                    baseMass=0,
                    baseCollisionShapeIndex=collision_id,
                    baseVisualShapeIndex=visual_id,
                    basePosition=[2*4.5, 2*4.5, 0.01],
                    baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                )
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
            obs_lower_bound = np.array([[lo,lo,lo, lo,lo,lo, lo,lo,lo, lo,lo,lo] for i in range(self.NUM_DRONES)])
            obs_upper_bound = np.array([[hi,hi,hi, hi,hi,hi, hi,hi,hi, hi,hi,hi] for i in range(self.NUM_DRONES)])
            #obs_lower_bound = np.array([[lo,lo,lo, lo,lo,lo, lo,lo,lo, lo,lo,lo] for i in range(self.NUM_DRONES)])
            #obs_upper_bound = np.array([[hi,hi,hi, hi,hi,hi, hi,hi,hi, hi,hi,hi] for i in range(self.NUM_DRONES)])
            # Add action buffer to observation space
            act_lo = -1
            act_hi = +1
            lidar_lo = 0
            lidar_hi = 1
            
            # Compute action size
            if self.ACT_TYPE in [ActionType.RPM, ActionType.VEL]:
                size = 4
            elif self.ACT_TYPE==ActionType.PID:
                size = 3
            elif self.ACT_TYPE in [ActionType.ONE_D_RPM, ActionType.ONE_D_PID]:
                size = 1
            else:
                size = 1  # default
            
            for i in range(self.LIDAR_BUFFER_SIZE):
                self.lidar_buffer.append(np.zeros((self.NUM_DRONES, 5)))
            for i in range(self.ACTION_BUFFER_SIZE):
                self.action_buffer.append(np.zeros((self.NUM_DRONES,size)))
                
            for i in range(self.ACTION_BUFFER_SIZE):
                obs_lower_bound = np.hstack([obs_lower_bound, np.array([[act_lo]*size for i in range(self.NUM_DRONES)])])
                obs_upper_bound = np.hstack([obs_upper_bound, np.array([[act_hi]*size for i in range(self.NUM_DRONES)])])
                    
            # Agregamos espacio para el historial del LIDAR (5 observaciones por cada paso en el buffer)
            for i in range(self.LIDAR_BUFFER_SIZE):
                obs_lower_bound = np.hstack([obs_lower_bound, np.full((self.NUM_DRONES, 5), lidar_lo)])
                obs_upper_bound = np.hstack([obs_upper_bound, np.full((self.NUM_DRONES, 5), lidar_hi)])

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
            A Box() of shape (NUM_DRONES,H,W,4) or (NUM_DRONES,12) depending on the observation type.

        """
        if self.OBS_TYPE == ObservationType.RGB:
            if self.step_counter%self.IMG_CAPTURE_FREQ == 0:
                for i in range(self.NUM_DRONES):
                    self.rgb[i], self.dep[i], self.seg[i] = self._getDroneImages(i,
                                                                                 segmentation=False
                                                                                 )
                    #### Printing observation to PNG frames example ############
                    if self.RECORD:
                        self._exportImage(img_type=ImageType.RGB,
                                          img_input=self.rgb[i],
                                          path=self.ONBOARD_IMG_PATH+"drone_"+str(i),
                                          frame_num=int(self.step_counter/self.IMG_CAPTURE_FREQ)
                                          )
            return np.array([self.rgb[i] for i in range(self.NUM_DRONES)]).astype('float32')
        elif self.OBS_TYPE == ObservationType.KIN:
            # OBS SPACE OF SIZE 15: 12 originales + 3 para TARGET_POS
            obs_12 = np.zeros((self.NUM_DRONES,12))
            #lidar = self.lidar if hasattr(self, 'lidar') else np.zeros(5) # placeholder for lidar readings in case the environment doesn't have a lidar sensor, to avoid attribute errors and allow the use of the same observation space for all environments regardless of the presence of a lidar sensor. The shape of the lidar reading is (NUM_DRONES, 5) because we have 5 rays in our simple lidar sensor, but it can be changed as needed.
            for i in range(self.NUM_DRONES):
                obs = self._getDroneStateVector(i)
                # Concatenar TARGET_POS a la observación
                if hasattr(self, 'TARGET_POS'):
                    if self.NUM_DRONES == 1:
                        distance = self.TARGET_POS - obs[0:3]
                        
                    else:
                        distance = self.TARGET_POS[i] - obs[0:3]
                else:
                    distance = np.zeros(3)
                lidar = self.lidar if hasattr(self, 'lidar') else np.zeros(5)
                # lastes_distance es la distancia al target de la observación anterior, que se guarda en un buffer para ser incluido en la observación actual y darle al agente información sobre hacia dónde se dirigía en el paso anterior, lo cual puede ser útil para aprender a evitar obstáculos y para problemas de control más complejos donde la observación actual no es suficiente para determinar la acción óptima (problemas no-Markovianos). En este caso, se incluye en la observación actual para ayudar al agente a aprender a evitar obstáculos, ya que la distancia al target puede estar dentro o cerca de un obstáculo y el agente puede necesitar aprender a desviarse de ese target para evitar chocar contra el obstáculo.
                self.observation_buffer.append(distance)
                lastes_distance = self.observation_buffer[-2] if len(self.observation_buffer) > 1 else np.zeros(3)
                obs_12[i, :] = np.hstack([distance, lastes_distance,  obs[7:10], obs[13:16]]).reshape(12,)
                #print("distance:", distance)
                #print("lastes_distance:", lastes_distance)
                l = np.tile(lidar, (self.NUM_DRONES, 1))
                self.lidar_buffer.append(l)
            ret = np.array([obs_12[i, :] for i in range(self.NUM_DRONES)]).astype('float32')
            # Add action buffer and lidar buffer to observation space
            for i in range(self.ACTION_BUFFER_SIZE):
                ret = np.hstack([ret, np.array([self.action_buffer[i][j, :] for j in range(self.NUM_DRONES)])])
            for i in range(self.LIDAR_BUFFER_SIZE):
                ret = np.hstack([ret, np.array([self.lidar_buffer[i][j, :] for j in range(self.NUM_DRONES)])])
                
            #print(ret)
            return ret
            ############################################################
        else:
            print("[ERROR] in BaseRLAviary._computeObs()")
