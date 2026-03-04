import numpy as np
import pybullet as p

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class HoverAviary(BaseRLAviary):
    
    
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_xyzs=np.array([[0,0,1]]),
                 initial_rpys=np.array([[0,0,0]]),
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
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
        self.TARGET_POS = np.array([0,0,1])
        self.EPISODE_LEN_SEC = 30
        self._best_dist = None  # Initialize the best distance to None
        self.step_count = 0
        
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
        
    ################################################################################
    def _draw_target_marker(self):
        # Elimina el marcador anterior si existe
        #print("Dibujando marcador de objetivo en:", self.TARGET_POS)
        if hasattr(self, '_target_marker_id'):
            p.removeUserDebugItem(self._target_marker_id)
        # Dibuja una esfera pequeña en TARGET_POS
        self._target_marker_id = p.addUserDebugLine(
            self.TARGET_POS,
            [0, 0, 0.0],
            [1, 0, 0],  # color rojo
            lineWidth=1,
            lifeTime=0  # 0 = permanente hasta que se borre
        )
        
        point_debug = p.addUserDebugPoints(
                            pointPositions=self.TARGET_POS.reshape(1,3),
                            pointColorsRGB=[[0,1,0]],
                            pointSize=10,
                            lifeTime=0
                        )      
        
        # self._target_text_id = p.addUserDebugText(str(self.TARGET_POS), self.TARGET_POS, [0,0,0], 0.5)

    def reset(self, *args, **kwargs):
        # Cambia el objetivo a un punto aleatorio en cada episodio
        self.TEST_BODY = p.createCollisionShape(p.GEOM_SPHERE, radius=0.01)
        self.TEST_BODY_ID = p.createMultiBody(baseMass=0, 
                                      baseCollisionShapeIndex=self.TEST_BODY, 
                                      basePosition=[0, 0, -10], # Escondido bajo el suelo
                                      physicsClientId=self.CLIENT)
        if self.random_targets:
            self.TARGET_POS = np.array([np.random.uniform(0, 2),np.random.uniform(0,2),np.random.uniform(0.5, 2.5)])
            r = .4  # radio del cubo de colisión
            for i in range(1000):  # Intenta encontrar un punto aleatorio sin colisiones
                    if self._is_space_clear(self.TARGET_POS, radius=r):
                        self._draw_target_marker()
                        break
                    else:
                        print(f"colisiones: {self.TARGET_POS}")
                        self.TARGET_POS = np.array([np.random.uniform(0, 2),np.random.uniform(0,2),np.random.uniform(0.5, 2.5)])      

                        
        else:
            self.TARGET_POS = np.array([0,0,1])
        
        
        
        self._best_dist = None  # Reinicia la mejor distancia
        obs = super().reset(*args, **kwargs)
        if self.GUI:  # Solo dibujar si hay GUI
            self._draw_target_marker()
        return obs
    """Single agent RL problem: hover at position."""

    ################################################################################
    
    def _is_space_clear(self, pos, radius=0.15, ignore_ids=[]):
        """
        Verifica si una posición en el espacio está libre de obstáculos reales (mallas).
        
        Args:
            pos (list/np.array): Posición [x, y, z] a testear.
            radius (float): Radio de seguridad alrededor del punto.
            ignore_ids (list): IDs de PyBullet que NO deben contar como colisión (ej. el marcador del target).
            
        Returns:
            bool: True si el espacio está limpio, False si hay algo cerca.
        """
        # 1. Teletransportar el sensor a la ubicación de prueba
        p.resetBasePositionAndOrientation(self.TEST_BODY_ID, pos, [0, 0, 0, 1], physicsClientId=self.CLIENT)
        
        bodys = []
        for i in range(p.getNumBodies(physicsClientId=self.CLIENT)):
            # 2. Obtener puntos más cercanos contra todos los objetos en la escena
            puntos = p.getClosestPoints(bodyA=self.TEST_BODY_ID, 
                                    bodyB=p.getBodyUniqueId(i), 
                                    distance=radius, 
                                    physicsClientId=self.CLIENT)
            if len(puntos) > 0:
                break
        
        # 3. Mover el sensor de vuelta al "limbo" para que no estorbe
        p.resetBasePositionAndOrientation(self.TEST_BODY_ID, [0, 0, -10], [0, 0, 0, 1], physicsClientId=self.CLIENT)

        # 4. Filtrar resultados
        for p_contact in puntos:
            id_detectado = p_contact[2] # ID del objeto con el que chocó
            # Si el objeto detectado no es el sensor mismo ni está en la lista de ignorados
            if id_detectado != self.TEST_BODY_ID and id_detectado not in ignore_ids:
                return False # Se detectó un obstáculo real
        
        return True # El espacio está despejado
    
    
    
    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        self.step_count + 1
        state = self._getDroneStateVector(0)
        pos = state[0:3]
        vel = state[10:13]  # vx, vy, vz
        angles = state[7:10]  # roll, pitch, yaw
        dist = np.linalg.norm(self.TARGET_POS - pos)
        #print(f"Distance to target: {dist}")

        # Inicializar la mejor distancia si es la primera vez
        if self._best_dist is None:
            self._best_dist = dist

        # Recompensa por acercarse y penalización por alejarse
        reward_dist = 0.0
        if dist < self._best_dist:
            reward_dist = 0.5  # Mayor recompensa por acercarse
            self._best_dist = dist
        elif dist > self._best_dist + 0.05:
            reward_dist = -3  # Mayor penalización por alejarse


        base_reward = max(0.00, (30 - dist**2)*0.08)
        #print(f"Base reward: {base_reward}")
        #print(f"Distance: {dist}")
        # Penalización por velocidad (para evitar tambaleo)
        speed_penalty = -0.1 * np.linalg.norm(vel)

        # Penalización por inclinación (roll y pitch, no yaw)
        angle_penalty = -.2 * (abs(angles[0]) + abs(angles[1]))

        # Recompensa extra si está muy cerca y estable
        bonus = 0.0
        if self.random_targets:
            r = 0.4  # radio del cubo de colisión
            if dist < 0.08 and np.linalg.norm(vel) < 0.2 and abs(angles[0]) < 0.2 and abs(angles[1]) < 0.2:
                old_target = self.TARGET_POS.copy()
                bonus = 1500
                if self.random_targets:
                    for i in range(1000):  # Intenta encontrar un punto aleatorio sin colisiones
                            self.TARGET_POS = np.array([old_target[0]+np.random.uniform(0, 2), old_target[1]+np.random.uniform(0, 2), np.random.uniform(-0.5, 0.5)])        
                            if not self._is_space_clear(self.TARGET_POS, radius=r):
                                self.TARGET_POS = np.array([old_target[0]+np.random.uniform(-0, 2), old_target[1]+np.random.uniform(0, 2), np.random.uniform(-0.5, 0.5)])        
                            else:
                                # print(f"Objetivo colocado sin colisiones: {self.TARGET_POS}")       
                                break
                self._draw_target_marker()
                print(f"New target position: {self.TARGET_POS}")
        else:
            
            if dist < 0.05 and np.linalg.norm(vel) < 0.1 and abs(angles[0]) < 0.1 and abs(angles[1]) < 0.1:
                bonus = 2
                print("Hovering achieved!")
            else:
                #print("try Hovering!")
                pass
                

        if state[2] < 0.05:
            penalty = -100
        else:
            penalty = -0.01
            
        if self.lidar is not None and np.min(self.lidar) < 0.3: # the drone is about to collide with something
            #print(f"penalty: obstacle detected at distance {self.lidar}")
            penalty = -5
        #print(f"dist: {dist}, reward_dist: {reward_dist}, speed_penalty: {speed_penalty}, angle_penalty: {angle_penalty}, bonus: {bonus}, base_reward: {base_reward}, Total: {base_reward + penalty + reward_dist + speed_penalty + angle_penalty + bonus}")
        return base_reward + penalty + reward_dist + speed_penalty + angle_penalty + bonus
        

    ################################################################################
    
    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        state = self._getDroneStateVector(0)
        if (abs(state[0]) > 10 or abs(state[1]) > 10 or state[2] > 2.5 # Truncate when the drone is too far away
        ):
            #print(  f"Truncated far away: pos {state[0:3]}, angles {state[7:10]}")
            return True
        
        if (abs(state[7]) > .7 or abs(state[8]) > .7 # Truncate when the drone is too tilted
        ):
            #print(  f"Truncated tilted: pos {state[0:3]}, angles {state[7:10]}")
            return True
        
        if state[2] < 0.05:
            #print(  f"Truncated height: pos {state[0:3]}, angles {state[7:10]}")
            return True
        
        if self.lidar is not None and np.min(self.lidar) < 0.15: # Truncate if the drone is about to collide with something
            print(f"Truncated: obstacle detected at distance {self.lidar}")
            return True
        
        if np.linalg.norm(self.TARGET_POS-state[0:3]) < .001:
            return False 
        else:
            return False
    ################################################################################
    
    def _computeTruncated(self):
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the current episode timed out.

        """
        
        
            
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
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
