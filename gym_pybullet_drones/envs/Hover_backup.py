import numpy as np
import pybullet as p
import math
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class HoverAviary(BaseRLAviary):
    
    
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_xyzs=np.array([[np.random.uniform(-0.5, 0.5),np.random.uniform(-0.5, 0.5), np.random.uniform(1, 1.5)]]),
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
        self._best_dist = None  # Initialize the best distance to None
        self._previous_dist = None
        self.step_count = 0
        self.score = 1
        self.actual_reward = 0
        self.time_penalty = 0
        self.TEST_BODY = None
        self.TEST_BODY_ID = None
        self.truncate_early = False
        self.point_track = None
        self.random_value = 1

        if initial_rpys is None:
            initial_rpys = np.array([[0, 0, np.random.uniform(-np.pi, np.pi)]])
        
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
        # print("Dibujando marcador de objetivo en:", self.TARGET_POS)
        if not self.GUI:
            return
        if hasattr(self, '_target_marker_id'):
            p.removeUserDebugItem(self._target_marker_id)
            self._target_marker_id = []
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
        return 

    def reset(self, *args, **kwargs):
        
        self.score = 1
        self.time_penalty = 0

        self.INIT_RPYS[0, 2] = np.random.uniform(-np.pi, np.pi)

        obs, info = super().reset(*args, **kwargs)  # ⚠️ PRIMERO esto      
        
        # Cambia el objetivo a un punto aleatorio en cada episodio
        self.TEST_BODY = p.createCollisionShape(p.GEOM_SPHERE, radius=.6, physicsClientId=self.CLIENT)
        self.TEST_BODY_ID = p.createMultiBody(baseMass=0, 
                                      baseCollisionShapeIndex=self.TEST_BODY, 
                                      basePosition=[0, 0, -10], # Escondido bajo el suelo
                                      physicsClientId=self.CLIENT)
        if self.random_targets:
            
            self.TARGET_POS = np.array([np.random.uniform(-self.random_value, self.random_value),np.random.uniform(-self.random_value, self.random_value),np.random.uniform(1., 1)])
            r = 0.8  # radio del cubo de colisión
            for i in range(500):  # Intenta encontrar un punto aleatorio sin colisiones
                    if self._is_space_clear(self.TARGET_POS, radius=r):
                        self._draw_target_marker([0, 1, 0])
                        #self.TARGET_POS = np.array([np.random.uniform(-0.3, 2),np.random.uniform(0.5, 2),np.random.uniform(0.5, 1.2)])
                        break      
                                                
                    else:
                        #print(f"colisiones: {self.TARGET_POS}")
                        #self._draw_target_marker([1, 0, 0])
                        self.TARGET_POS = np.array([np.random.uniform(-self.random_value, self.random_value),np.random.uniform(-self.random_value, self.random_value),np.random.uniform(1., 1)])      
    
        self._best_dist = None  # Reinicia la mejor distancia
        self._previous_dist = None

        # Hélice Ascendente redondeada
        helice_ascendente = lambda p: np.round(np.array([
            np.random.uniform(.5,1.5) * math.cos(10 * math.pi * p),      # X
            np.random.uniform(.5,1.5)  * math.sin(10 * math.pi * p),      # Y
            p*5 + 1                            # Z
        ]), 2)

        # Montaña Rusa redondeada con dirección aleatoria
        roller_angle = np.random.uniform(0, 2 * math.pi)
        roller_angle = -np.pi
        roller_coaster = lambda p: np.round(np.array([
            np.random.uniform(20,20) * p * math.cos(roller_angle),                            # X
            np.random.uniform(20,20) * p * math.sin(roller_angle),                            # Y
            1 * math.sin(10 * math.pi * p) + 1   # Z
        ]), 2)
        
        if not self.one_only_target and not self.random_targets:
            self.tast = np.random.choice(2,1)
            self.tast = 1
            self.pasos = np.random.random_integers(50,65)
            self.point_track = self.generar_trayectoria(helice_ascendente if self.tast == 0 else roller_coaster, pasos=self.pasos)
            self.TARGET_POS = self.point_track.pop(0)
            print("roller_coaster" , self.tast )
        
        
        elif(self.one_only_target and not self.random_targets):
            self.TARGET_POS = np.array([0,0,2])
            self.pasos = 0
            #print("one_only_target" , self.TARGET_POS)
            self._draw_target_marker([0, 1, 0])
        self.truncate_early = False
        #print(self.TARGET_POS)
        return self._computeObs(), info
    """Single agent RL problem: hover at position."""

    ################################################################################
    
    def _is_space_clear(self, pos, radius=0.2, ignore_ids=[]):
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
        angle_vel = state[13:16]  # roll_rate, pitch_rate, yaw_rate
        delta_pos = self.TARGET_POS - pos
        dist = np.linalg.norm(delta_pos)
        target_dir = delta_pos / (dist + 1e-8)
        #print(f"Distance to target: {dist}")
        # if self.GUI:
        #     p.resetDebugVisualizerCamera(
        #         cameraDistance=2,    # Distancia desde el dron
        #         cameraYaw=45,          # Ángulo de rotación horizontal
        #         cameraPitch=-30,       # Ángulo de inclinación vertical
        #         cameraTargetPosition=pos)
            
            
        if self._previous_dist is None:
            self._previous_dist = dist

        # Dense progress reward: temporary detours for turning are allowed,
        # while repeatedly moving away from the target is still discouraged.
        progress = float(np.clip(self._previous_dist - dist, -0.2, 0.2))
        progress_reward = 8.0 * progress
        self._previous_dist = dist

        base_reward = max(0.0, (10.0 - dist**2) * 0.08)
        speed = np.linalg.norm(vel)
        velocity_alignment_reward = 0.15 * np.dot(vel, target_dir)
        speed_penalty = -0.03 * speed

        # Reward the body x-axis pointing toward the target without Euler yaw.
        # PyBullet quaternions are ordered as (x, y, z, w); the first column
        # of its rotation matrix is the body-forward axis in world coordinates.
        delta_xy = delta_pos[:2]
        dist_xy = np.linalg.norm(delta_xy)
        if dist_xy > 1e-8:
            target_dir_xy = delta_xy / dist_xy
            rotation_matrix = np.asarray(p.getMatrixFromQuaternion(state[3:7])).reshape(3, 3)
            forward_xy = rotation_matrix[:2, 0]
            forward_norm = np.linalg.norm(forward_xy)
            if forward_norm > 1e-8:
                forward_xy = forward_xy / forward_norm
                heading_alignment_reward = 0.3 * np.dot(forward_xy, target_dir_xy)
                print (np.dot(forward_xy, target_dir_xy))
            else:
                heading_alignment_reward = 0.0
        else:
            heading_alignment_reward = 0.0

        # Keep the vehicle level; yaw is rewarded only for facing the target.
        angle_penalty = -0.08 * (abs(angles[0]) + abs(angles[1]))
        angle_vel_penalty = -0.02 * (abs(angle_vel[0]) + abs(angle_vel[1]))
        # penalizar por tiempo acumulativo sin actualizar objetivo
        self.time_penalty =  self.time_penalty -.2 / (self.PYB_FREQ * self.EPISODE_LEN_SEC)  # Penalización que aumenta con el tiempo
        #print(f"Time penalty: {self.time_penalty :.8f}")
        
        # Recompensa extra si está muy cerca y estable
        bonus = 0.0
        if self.random_targets:
            self.pasos = 15
            if dist < 0.2 and np.linalg.norm(vel) < 0.4 :
                old_target = self.TARGET_POS.copy()
                bonus = 50 -(50 * self.score/20)  # Recompensa decreciente a medida que se alcanzan más objetivos
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
                r = 0.8  # radio del cubo de colisión
                
                self.TARGET_POS = np.array([ (np.random.uniform(-self.random_value, self.random_value)),np.random.uniform(-self.random_value, self.random_value), np.random.uniform(0.3, 1.5)])
                for i in range(100):  # Intenta encontrar un punto aleatorio sin colisiones    
                    if self._is_space_clear(self.TARGET_POS, radius=r):
                        self._draw_target_marker([0, 0, 1])

                        break
                        #self.TARGET_POS = np.array([old_target[0]+np.random.uniform(-.5, 2), old_target[1]+np.random.uniform(-0.5, 2), np.random.uniform(0.5, 2)])
                        
                    else:
                        #self._draw_target_marker([1, 0, 0])
                        self.TARGET_POS = np.array([ (np.random.uniform(-self.random_value, self.random_value)), np.random.uniform(-self.random_value, self.random_value), np.random.uniform(0.3, 1.5)])
                        if i>=99:
                            print("No se encontró un nuevo objetivo sin colisiones después de 100 intentos. Manteniendo el mismo objetivo.")
                            bonus = 1000  # No dar la recompensa si no se puede colocar un nuevo objetivo
                            self.truncate_early = True
                self._best_dist = None  # Reinicia la mejor distancia para el nuevo objetivo
                self._previous_dist = np.linalg.norm(self.TARGET_POS - pos)

        elif(self.one_only_target):
            if dist < .2 and np.linalg.norm(vel) < 0.2 :
                bonus = 0.2
                #print("Hovering achieved!")
                self.score = self.score + 1
                self.time_penalty = 0
            else:
                #print("try Hovering!")
                pass
        else:
            if dist < 0.4 and np.linalg.norm(vel) < 0.2: 
                self.TARGET_POS = self.point_track.pop(0)
                self.time_penalty = 0
                bonus = (10) * (self.score*1)
                #print("target: ", self.TARGET_POS )
                self._draw_target_marker([0, 1, 0])
                self.score += 1
        
        #print(np.max(self.lidar))
        penaltyLidar = 0
        # if self.lidar is not None and np.max(self.lidar) > 0.7: # the drone is about to collide with something
        #     #print(f"penalty: obstacle detected at distance {self.lidar}")
            
        #     penaltyLidar = -1* (0.7 - np.max(self.lidar)) # penalización proporcional a lo cerca que esté el obstáculo, con un máximo de -5 cuando el obstáculo está a 0.0m de distancia
        
        if self.score == 20 and (self.random_targets or not self.one_only_target):
            print("¡Puntuación máxima alcanzada! Reiniciando entorno.")
            self.truncate_early = True
            self.time_penalty = 0
            bonus = bonus + 500  # Dar una gran recompensa por alcanzar la puntuación máxima
        total_reward = (base_reward + self.time_penalty + progress_reward
                + velocity_alignment_reward + speed_penalty
            + heading_alignment_reward + angle_vel_penalty
            + angle_penalty + bonus + penaltyLidar)
        #print(np.max(self.lidar))
        
        self.actual_reward = self.actual_reward + total_reward
        #print(f"Reward breakdown: base {base_reward:.3f}, progress {progress_reward:.3f}, velocity {velocity_alignment_reward:.3f}, total {total_reward:.3f}")
        
            
        
        return total_reward
        
    import math

    # 1. Definimos la FUNCIÓN que genera la lista
    def generar_trayectoria(self, formula_figura, pasos=50):
        lista_puntos = []
        for i in range(pasos + 1):
            # Aquí es donde PREPARAMOS el valor de p (de 0.0 a 1.0)
            p = i / pasos 
            
            # Aquí LLAMAMOS a la lambda que nos pases
            punto_base = formula_figura(p)
            punto = punto_base

            # Mantiene la posición dentro de la misma figura, pero evita
            # aceptar objetivos que estén dentro o demasiado cerca de un obstáculo.
            for _ in range(500):
                if self._is_space_clear(punto, radius=0.8):
                    break
                # Desplaza el candidato en una dirección aleatoria para salir
                # del obstáculo sin cambiar el orden de la trayectoria.
                punto = punto_base + np.random.uniform(
                    low=[-0.3, -0.3, -0.4],
                    high=[0.3, 0.3, 0.4]
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
        
        if self.time_penalty < -.08:
            print("static Truncated - reward: "  + str(self.actual_reward-penalty))
            print('score', self.score)
            self.actual_reward = 0 

            return True, penalty
        
        if self.truncate_early:
            print("Early Truncated - reward: "  + str(self.actual_reward))
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
        
        if (abs(state[7]) > 1.1 or abs(state[8]) > 1.1 # Truncate when the drone is too tilted
        ):
            #print(  f"Truncated tilted: pos {state[0:3]}, angles {state[7:10]}")
            print(" tilted - reward: "  + str(self.actual_reward - penalty))
            print('score', self.score)
            self.actual_reward = 0
            return True, penalty
        
        if state[2] < 0.02:
            #print(  f"Truncated height: pos {state[0:3]}, angles {state[7:10]}")
            print("height limit - reward: "  + str(self.actual_reward - penalty))
            print('score', self.score)
            self.actual_reward = 0
            return True, penalty
        

        
        # if self.lidar is not None and np.max(self.lidar) > 0.95: # Truncate if the drone is about to collide with something
        #     # print(f"Truncated: obstacle detected at distance {self.lidar}")
        #     print("collision special!!!! - reward: "  + str(self.actual_reward - (penalty+700)))
        #     print('score', self.score)
        #     self.actual_reward = 0
        #     return True, penalty 

        if self.obstacle_collision:
            print("obstacle collision - reward: "  + str(self.actual_reward - penalty))
            self.actual_reward = 0
            return True, penalty
        
        if np.linalg.norm(self.TARGET_POS-state[0:3]) < .001:
            print("target reached - reward: "  + str(self.actual_reward - penalty))
            print('score', self.score)
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
        


            
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC*10:
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
