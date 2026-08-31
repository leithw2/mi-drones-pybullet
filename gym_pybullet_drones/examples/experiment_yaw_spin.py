"""Open-loop experiment to verify yaw authority independently of PPO."""

import argparse
import time

import numpy as np

from gym_pybullet_drones.envs.HoverAviary import HoverAviary
from gym_pybullet_drones.utils.enums import ActionType, ObservationType, Physics


def run(seconds=15.0, strength=1.0, direction=1.0, gui=True):
    """Apply a constant differential RPM command and report unwrapped yaw."""
    environment = HoverAviary(
        gui=gui,
        obs=ObservationType.KIN,
        act=ActionType.RPM,
        physics=Physics.PYB,
        ctrl_freq=60,
        random_targets=True,
    )

    try:
        environment.reset(seed=0)
        steps = int(seconds * environment.CTRL_FREQ)
        strength = float(np.clip(strength, 0.0, 1.0))
        direction = 1.0 if direction >= 0 else -1.0

        # BaseAviary computes z torque as -t0 + t1 - t2 + t3.
        action = direction * strength * np.array([[-1.0, 1.0, -1.0, 1.0]], dtype=np.float32)
        previous_yaw = float(environment._getDroneStateVector(0)[9])
        unwrapped_yaw = previous_yaw
        start = time.perf_counter()

        for step in range(steps):
            _, _, terminated, truncated, _ = environment.step(action)
            yaw = float(environment._getDroneStateVector(0)[9])
            yaw_delta = np.arctan2(np.sin(yaw - previous_yaw), np.cos(yaw - previous_yaw))
            unwrapped_yaw += yaw_delta
            previous_yaw = yaw

            if step % environment.CTRL_FREQ == 0:
                print(
                    f"t={step / environment.CTRL_FREQ:5.1f}s "
                    f"yaw={np.degrees(unwrapped_yaw):8.1f} deg "
                    f"yaw_rate={environment._getDroneStateVector(0)[15]:7.3f} rad/s"
                )

            if terminated or truncated:
                print(f"El entorno terminó en t={step / environment.CTRL_FREQ:.2f}s")
                break

        elapsed = time.perf_counter() - start
        turns = unwrapped_yaw / (2.0 * np.pi)
        print(f"Resultado: {np.degrees(unwrapped_yaw):.1f} grados ({turns:.2f} vueltas)")
        print(f"Tiempo de ejecución: {elapsed:.2f}s")
    finally:
        environment.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experimento de autoridad de yaw sin PPO")
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--strength", type=float, default=1.0, help="Amplitud del diferencial RPM, entre 0 y 1")
    parser.add_argument("--direction", type=float, default=1.0, help="1 o -1 para invertir el giro")
    parser.add_argument("--gui", action=argparse.BooleanOptionalAction, default=True)
    run(**vars(parser.parse_args()))