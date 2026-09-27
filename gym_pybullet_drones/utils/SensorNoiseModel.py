import numpy as np
from dataclasses import dataclass, field


@dataclass
class IMUNoiseConfig:
    """Configuración de ruido gaussiano y bias para la IMU."""
    acc_std: np.ndarray = field(default_factory=lambda: np.array([0.8, 0.8, 0.8]))  # m/s^2
    acc_bias: np.ndarray = field(default_factory=lambda: np.array([0.005, -0.005, 0.005]))
    
    gyro_std: np.ndarray = field(default_factory=lambda: np.array([0.8, 0.8, 0.8]))  # rad/s
    gyro_bias: np.ndarray = field(default_factory=lambda: np.array([0.005, -0.005, 0.001]))
    
    orientation_std: float = 0.01  # rad (ruido en estimación Roll/Pitch/Yaw)


@dataclass
class LaserNoiseConfig:
    """Configuración de ruido gaussiano para sensor de distancia / LiDAR."""
    std: float = 0.1            # Desviación estándar en metros
    bias: float = 0.005          # Desplazamiento sistemático
    max_range: float = 2.0      # Rango máximo del sensor (m)
    min_range: float = 0.05      # Rango mínimo del sensor (m)


@dataclass
class OdometryNoiseConfig:
    """Configuración de ruido para odometría / GPS / Sistema de captura de movimiento."""
    pos_std: np.ndarray = field(default_factory=lambda: np.array([0.4, 0.4, 0.4]))  # x, y, z (m)
    vel_std: np.ndarray = field(default_factory=lambda: np.array([0.8, 0.8, 0.8]))  # vx, vy, vz (m/s)


class SensorNoiseModel:
    """
    Orquestador principal para aplicar ruido gaussiano y sesgo
    a las observaciones del dron.
    """
    def __init__(
        self,
        imu_config: IMUNoiseConfig = None,
        laser_config: LaserNoiseConfig = None,
        odom_config: OdometryNoiseConfig = None,
        seed: int = None
    ):
        self.imu_cfg = imu_config or IMUNoiseConfig()
        self.laser_cfg = laser_config or LaserNoiseConfig()
        self.odom_cfg = odom_config or OdometryNoiseConfig()
        
        self.rng = np.random.RandomState(seed)

    def apply_imu_noise(self, acc: np.ndarray, gyro: np.ndarray, angles: np.ndarray = None):
        """
        Aplica ruido y bias a acelerómetro, giroscopio y ángulos de orientación.
        """
        noisy_acc = acc + self.imu_cfg.acc_bias + self.rng.normal(0.0, self.imu_cfg.acc_std, size=acc.shape)
        noisy_gyro = gyro + self.imu_cfg.gyro_bias + self.rng.normal(0.0, self.imu_cfg.gyro_std, size=gyro.shape)
        
        noisy_angles = None
        if angles is not None:
            noisy_angles = angles + self.rng.normal(0.0, self.imu_cfg.orientation_std, size=angles.shape)
            # Normalizar ángulos al rango [-pi, pi]
            noisy_angles = np.arctan2(np.sin(noisy_angles), np.cos(noisy_angles))

        return noisy_acc, noisy_gyro, noisy_angles

    def apply_laser_noise(self, laser_distances: np.ndarray) -> np.ndarray:
        """
        Aplica ruido gaussiano a los lecturas LiDAR/láser con límites físicos de saturación.
        """
        noise = self.rng.normal(self.laser_cfg.bias, self.laser_cfg.std, size=laser_distances.shape)
        noisy_distances = laser_distances + noise
        
        # Recortar lecturas a los límites físicos del sensor
        return np.clip(noisy_distances, self.laser_cfg.min_range, self.laser_cfg.max_range)

    def apply_odometry_noise(self, pos: np.ndarray, vel: np.ndarray):
        """
        Aplica ruido a la estimación de posición y velocidad lineal del vehículo.
        """
        noisy_pos = pos + self.rng.normal(0.0, self.odom_cfg.pos_std, size=pos.shape)
        noisy_vel = vel + self.rng.normal(0.0, self.odom_cfg.vel_std, size=vel.shape)
        
        # Mantener altura mínima en 0 si toca suelo
        # noisy_pos[2] = max(0.0, noisy_pos[2])
        
        return noisy_pos, noisy_vel

    def reset_seed(self, seed: int):
        """Permite sincronizar la semilla en cada reset del entorno."""
        self.rng = np.random.RandomState(seed)