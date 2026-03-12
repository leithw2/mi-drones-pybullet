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
        self.TARGET_POS = np.array([0,2,1])
        #print("Target position: " + str(self.TARGET_POS))
        self.EPISODE_LEN_SEC = 30
        self._best_dist = None  # Initialize the best distance to None
        self.step_count = 0
        self.score = 1
        self.actual_reward = 0
        self.time_penalty = 0
        self.TEST_BODY = None
        self.TEST_BODY_ID = None
        self.truncate_early = False

        
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
    def _draw_target_marker(self, color=[0,1,0]):
        # Elimina el marcador anterior si existe
        #print("Dibujando marcador de objetivo en:", self.TARGET_POS)
        if hasattr(self, '_target_marker_id'):
            p.removeUserDebugItem(self._target_marker_id)
        # Dibuja una esfera pequeña en TARGET_POS
        self._target_marker_id = p.addUserDebugLine(
            self.TARGET_POS,
            [0, 0, 0],
            color,  # color rojo
            lineWidth=1,
            lifeTime=10  # 0 = permanente hasta que se borre
        )
        
        point_debug = p.addUserDebugPoints(
                            pointPositions=self.TARGET_POS.reshape(1,3),
                            pointColorsRGB=[color],
                            pointSize=10,
                            lifeTime=1
                        )      
        #self._target_text_id = p.addUserDebugText(str(self.TARGET_POS), self.TARGET_POS, [0,0,0], 0.5)

    def reset(self, *args, **kwargs):
        obs = super().reset(*args, **kwargs)
        #print("control freq: " + str(self.CTRL_FREQ), "pyb freq: " + str(self.PYB_FREQ), "episode len sec: " + str(self.EPISODE_LEN_SEC))
        self.score = 1
        self.time_penalty = 0
        #self.TARGET_POS = np.array([0,2,1])
        #self.TARGET_POS = np.array([np.random.uniform(-0.3, 0.1),np.random.uniform(.50, 5.0),np.random.uniform(0.8, 1.2)])      
        # Cambia el objetivo a un punto aleatorio en cada episodio
        self.TEST_BODY = p.createCollisionShape(p.GEOM_SPHERE, radius=.6)
        self.TEST_BODY_ID = p.createMultiBody(baseMass=0, 
                                      baseCollisionShapeIndex=self.TEST_BODY, 
                                      basePosition=[0, 0, -10], # Escondido bajo el suelo
                                      physicsClientId=self.CLIENT)
        if self.random_targets:
            self.TARGET_POS = np.array([np.random.uniform(-6, 1),np.random.uniform(-6, 1),np.random.uniform(0.5, 1.2)])
            r = 0.4  # radio del cubo de colisión
            for i in range(500):  # Intenta encontrar un punto aleatorio sin colisiones
                    if self._is_space_clear(self.TARGET_POS, radius=r):
                        self._draw_target_marker([0, 1, 0])
                        #self.TARGET_POS = np.array([np.random.uniform(-0.3, 2),np.random.uniform(0.5, 2),np.random.uniform(0.5, 1.2)])
                        break      
                                                
                    else:
                        #print(f"colisiones: {self.TARGET_POS}")
                        #self._draw_target_marker([1, 0, 0])
                        self.TARGET_POS = np.array([np.random.uniform(-6, 1),np.random.uniform(-6, 1),np.random.uniform(0.5, 1.2)])      

        
        #print("Nuevo objetivo: " + str(self.TARGET_POS))
        self._best_dist = None  # Reinicia la mejor distancia
       
        if self.GUI:  # Solo dibujar si hay GUI
            pass
        return obs
    """Single agent RL problem: hover at position."""

    ################################################################################
    
    def _is_space_clear(self, pos, radius=0.6, ignore_ids=[]):
        # 1. Límites del área de vuelo
        # if pos[0] < -1 or pos[1] < -1:
        #     return False
        
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
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        self.truncate_early = False
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
        if dist < self._best_dist - 0.01:  # Si se acerca al objetivo
            reward_dist = 2  # Mayor recompensa por acercarse
            self._best_dist = dist
        elif dist > self._best_dist: 
            reward_dist = -0.5  # Mayor penalización por alejarse
            #print(f"Distance increased from {self._best_dist:.3f} to {dist:.3f} - reward: {reward_dist}")


        base_reward = max(0.00, (30 - dist**2)*0.08)
        #print(f"Base reward: {base_reward}")
        #print(f"Distance: {dist}")
        # Penalización por velocidad (para evitar tambaleo)
        speed_penalty = -1 * np.linalg.norm(vel)

        # Penalización por inclinación (roll y pitch, no yaw)
        angle_penalty = -1 * (abs(angles[0]) + abs(angles[1]))

        # penalizar por tiempo acumulativo sin actualizar objetivo
        self.time_penalty = self.time_penalty -10 / (self.PYB_FREQ * self.EPISODE_LEN_SEC)  # Penalización que aumenta con el tiempo
        #print(f"Time penalty: {self.time_penalty :.8f}")
        
        # Recompensa extra si está muy cerca y estable
        bonus = 0.0
        if self.random_targets:
            if dist < 0.2 and np.linalg.norm(vel) < 0.2 and abs(angles[0]) < 0.2 and abs(angles[1]) < 0.2:
                old_target = self.TARGET_POS.copy()
                bonus = 190 * self.score
                self.time_penalty = 0
                self.score += 1
                if self.score == 6:
                    print("¡Puntuación 6 alcanzada!" + " objetivo alcanzado: " + str(old_target))
                if self.score == 9:
                    print("¡Puntuación 9 alcanzada!" + " objetivo alcanzado: " + str(old_target))
                if self.score == 12:
                    print("¡Puntuación 12 alcanzada!" + " objetivo alcanzado: " + str(old_target))
                if self.score == 15:
                    print("¡Puntuación 15 alcanzada!" + " objetivo alcanzado: " + str(old_target))
                r = 0.6  # radio del cubo de colisión
                
                self.TARGET_POS = np.array([ (np.random.uniform(-4, 4)),np.random.uniform(-4, 4), np.random.uniform(0.5, 1.2)])
                for i in range(100):  # Intenta encontrar un punto aleatorio sin colisiones    
                    if self._is_space_clear(self.TARGET_POS, radius=r):
                        self._draw_target_marker([0, 0, 1])
                        break
                        #self.TARGET_POS = np.array([old_target[0]+np.random.uniform(-.5, 2), old_target[1]+np.random.uniform(-0.5, 2), np.random.uniform(0.5, 2)])
                        
                    else:
                        #self._draw_target_marker([1, 0, 0])
                        self.TARGET_POS = np.array([ (np.random.uniform(-4, 4)), np.random.uniform(-4, 4), np.random.uniform(0.5, 1.2)])
                        if i>=99:
                            print("No se encontró un nuevo objetivo sin colisiones después de 100 intentos. Manteniendo el mismo objetivo.")
                            bonus = 1000  # No dar la recompensa si no se puede colocar un nuevo objetivo
                            self.truncate_early = True
                            

        else:
            #print(f"New target position: {self.TARGET_POS}")
            if dist < .1 and np.linalg.norm(vel) < 0.1 and abs(angles[0]) < 0.1 and abs(angles[1]) < 0.1:
                bonus = 6.2
                #print("Hovering achieved!")
                time_penalty = 0
            else:
                #print("try Hovering!")
                pass
                

        penaltyLidar = 0
        if self.lidar is not None and np.min(self.lidar) < 0.4: # the drone is about to collide with something
            #print(f"penalty: obstacle detected at distance {self.lidar}")
            penaltyLidar = -1* (0.3 - np.min(self.lidar)) # penalización proporcional a lo cerca que esté el obstáculo, con un máximo de -5 cuando el obstáculo está a 0.0m de distancia
        total_reward = base_reward + self.time_penalty  + reward_dist + speed_penalty + angle_penalty + bonus + penaltyLidar
        #print("Total reward: " + str(total_reward) )
        self.actual_reward = self.actual_reward + total_reward
        #print(f"Reward breakdown: base {base_reward:.3f}, time_penalty {self.time_penalty:.3f}, reward_dist {reward_dist:.3f}, speed_penalty {speed_penalty:.3f}, angle_penalty {angle_penalty:.3f}, bonus {bonus:.3f}, penaltyLidar {penaltyLidar:.3f} - total: {total_reward:.3f}")
        if self.score == 10:
            print("¡Puntuación máxima alcanzada! Reiniciando entorno.")
            self.truncate_early = True
            bonus = bonus + 1000  # Dar una gran recompensa por alcanzar la puntuación máxima
            
        return total_reward
        

    ################################################################################
    
    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        
        penalty = 4500 / self.score # Penalización que disminuye a medida que se alcanzan más objetivos
        state = self._getDroneStateVector(0)
        if (abs(state[0]) > 50 or abs(state[1]) > 50 or state[2] > 2.5 # Truncate when the drone is too far away
        ):
            #print(  f"Truncated far away: pos {state[0:3]}, angles {state[7:10]}")
            print("far away - reward: "  + str(self.actual_reward - penalty))
            self.actual_reward = 0
            return True, penalty
        
        if (abs(state[7]) > .6 or abs(state[8]) > .6 # Truncate when the drone is too tilted
        ):
            #print(  f"Truncated tilted: pos {state[0:3]}, angles {state[7:10]}")
            print(" tilted - reward: "  + str(self.actual_reward - penalty))
            self.actual_reward = 0
            return True, penalty
        
        if state[2] < 0.05:
            #print(  f"Truncated height: pos {state[0:3]}, angles {state[7:10]}")
            print("height limit - reward: "  + str(self.actual_reward - penalty))
            self.actual_reward = 0
            return True, penalty
        
        if self.lidar is not None and np.min(self.lidar) < 0.1: # Truncate if the drone is about to collide with something
            # print(f"Truncated: obstacle detected at distance {self.lidar}")
            print("collision special!!!! - reward: "  + str(self.actual_reward - penalty))
            self.actual_reward = self.actual_reward - 5000
            return True, penalty + 5000
        
        #print("target pos: " + str(self.TARGET_POS) + " drone pos: " + str(state[0:3]) + " distance to target: " + str(np.linalg.norm(self.TARGET_POS-state[0:3])))
        if np.linalg.norm(self.TARGET_POS-state[0:3]) < .001:
            print("target reached - reward: "  + str(self.actual_reward - penalty))
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
        
        if self.truncate_early:
            print("Early Truncated - reward: "  + str(self.actual_reward))
            self.actual_reward = 0
            self.truncate_early = False
            return True
            
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            print("Time Truncated - reward: "  + str(self.actual_reward))
            self.actual_reward = 0

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
