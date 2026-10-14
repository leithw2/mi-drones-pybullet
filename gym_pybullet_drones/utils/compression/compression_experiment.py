
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from stable_baselines3 import PPO

from gym_pybullet_drones.utils.enums import (
    DroneModel,
    ObservationType,
    ActionType,
    Physics,
)
from gym_pybullet_drones.utils.evaluation import (
    EVAL_DEVICE,
    EVAL_SEEDS,
    POWERLOOP_ORIENTATION_VARIATION,
    POWERLOOP_POSITION_VARIATION,
    POWERLOOP_TRACK_SCALE,
    make_eval_env,
)


def linear_layers(module):
    """Obtiene las capas Linear de una rama de la política."""
    return [
        layer for layer in module.modules()
        if isinstance(layer, nn.Linear)
    ]


def select_neurons(importance, count):
    """Selecciona las neuronas con mayor importancia."""
    return torch.topk(
        importance,
        k=count,
    ).indices.sort().values


def copy_reduced_actor(original_policy, reduced_policy, hidden_size):
    """
    Reduce únicamente el actor.

    Conserva las neuronas más importantes según la magnitud
    de sus conexiones salientes.

    No modifica el crítico.
    """
    old_layers = linear_layers(
        original_policy.mlp_extractor.policy_net
    )
    new_layers = linear_layers(
        reduced_policy.mlp_extractor.policy_net
    )

    old_action = original_policy.action_net
    new_action = reduced_policy.action_net

    if len(old_layers) != 2 or len(new_layers) != 2:
        raise ValueError(
            "El actor debe tener exactamente dos capas ocultas."
        )

    old_first, old_second = old_layers
    new_first, new_second = new_layers

    if old_first.out_features != 64 or old_second.out_features != 64:
        raise ValueError(
            "Este código espera que el actor original sea 64-64."
        )

    if not 0 < hidden_size < 64:
        raise ValueError("hidden_size debe estar entre 1 y 63.")

    if old_action.out_features != new_action.out_features:
        raise ValueError("Las dimensiones de salida del actor no coinciden.")

    # Seleccionar primero la segunda capa oculta por su influencia
    # sobre las acciones finales.
    importance_layer2 = old_action.weight.detach().abs().sum(dim=0)
    idx2 = select_neurons(importance_layer2, hidden_size)

    # Seleccionar la primera capa según sus conexiones hacia las
    # neuronas seleccionadas de la segunda capa.
    importance_layer1 = (
        old_second.weight.detach()[idx2, :].abs().sum(dim=0)
    )
    idx1 = select_neurons(importance_layer1, hidden_size)

    with torch.no_grad():
        # Primera capa: entradas originales, menos neuronas de salida.
        new_first.weight.copy_(old_first.weight[idx1, :])
        new_first.bias.copy_(old_first.bias[idx1])

        # Segunda capa: menos entradas y menos neuronas de salida.
        new_second.weight.copy_(
            old_second.weight[idx2, :][:, idx1]
        )
        new_second.bias.copy_(old_second.bias[idx2])

        # Capa de acciones: conservar las conexiones seleccionadas.
        new_action.weight.copy_(old_action.weight[:, idx2])
        if old_action.bias is not None:
            new_action.bias.copy_(old_action.bias)

        # Conservar la desviación estándar aprendida de la política.
        if (
            hasattr(original_policy, "log_std")
            and hasattr(reduced_policy, "log_std")
        ):
            reduced_policy.log_std.copy_(
                original_policy.log_std
            )


def structured_prune_policy(model, hidden_size, env):
    """
    Construye un PPO nuevo con actor reducido y crítico intacto.

    Actor original: 64-64.
    Crítico original: 256-256-128.
    """
    original_policy = model.policy

    # Comprobar la arquitectura real del crítico.
    critic_layers = linear_layers(
        original_policy.mlp_extractor.value_net
    )

    critic_sizes = [
        layer.out_features for layer in critic_layers
    ]

    if critic_sizes != [256, 256, 128]:
        raise ValueError(
            "La arquitectura del crítico no coincide con "
            f"256-256-128. Encontrada: {critic_sizes}"
        )

    # Reutilizar la configuración original y cambiar solamente
    # el tamaño de las capas del actor.
    policy_kwargs = copy.deepcopy(model.policy_kwargs)
    policy_kwargs["net_arch"] = {
        "pi": [hidden_size, hidden_size],
        "vf": [256, 256, 128],
    }

    smaller = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        device=model.device,
        verbose=0,
    )

    # Copiar el actor reducido.
    copy_reduced_actor(
        original_policy,
        smaller.policy,
        hidden_size,
    )

    # Copiar el crítico completo: arquitectura y pesos idénticos.
    smaller.policy.mlp_extractor.value_net.load_state_dict(
        original_policy.mlp_extractor.value_net.state_dict()
    )

    smaller.policy.value_net.load_state_dict(
        original_policy.value_net.state_dict()
    )

    # Copiar el extractor de características si tiene parámetros.
    if (
        original_policy.features_extractor is not None
        and smaller.policy.features_extractor is not None
    ):
        smaller.policy.features_extractor.load_state_dict(
            original_policy.features_extractor.state_dict()
        )

    return smaller


def count_parameters(module):
    return sum(
        parameter.numel()
        for parameter in module.parameters()
    )


def build_env(args):
    return make_eval_env(
        multiagent=args.multiagent,
        gui=args.gui,
        record=False,
        drone_model=DroneModel.CF2X250,
        obs=ObservationType("kin"),
        act=ActionType.BRUSHLESS_THRUST,
        physics=Physics.PYB_WIND,
        random_targets=False,
        num_drones=1,
        track_scale=args.track_scale,
        track_position_variation=args.track_position_variation,
        track_orientation_variation=args.track_orientation_variation,
    )


def evaluate_candidate(model, name, hidden_size, args, output_dir):
    """Evalúa un candidato con el mismo conjunto de semillas."""

    env = build_env(args)
    episode_rows = []
    latencies_ms = []

    try:
        ctrl_freq = env.unwrapped.CTRL_FREQ

        for ep in range(args.episodes):
            seed = EVAL_SEEDS[ep % len(EVAL_SEEDS)]

            obs, info = env.reset()
            terminated = False
            truncated = False
            total_reward = 0.0
            steps = 0

            while not (terminated or truncated):
                start = time.perf_counter()

                action, _ = model.predict(
                    obs,
                    deterministic=True,
                )

                latencies_ms.append(
                    (time.perf_counter() - start) * 1000
                )

                obs, reward, terminated, truncated, info = (
                    env.step(action)
                )

                total_reward += float(reward)
                steps += 1

                if args.sim_speed > 0:
                    time.sleep(
                        1.0 / (ctrl_freq * args.sim_speed)
                    )

            episode_rows.append({
                "experiment": name,
                "hidden_size": hidden_size,
                "episode": ep + 1,
                "seed": seed,
                "reward": total_reward,
                "steps": steps,
                "is_success": info.get("is_success", np.nan),
            })

            print(
                f"[{name}] Episodio {ep + 1}/{args.episodes} "
                f"| seed={seed} | reward={total_reward:.2f} "
                f"| steps={steps}"
            )

    finally:
        env.close()

    artifact_path = output_dir / name
    model.save(str(artifact_path))
    zip_path = artifact_path.with_suffix(".zip")

    rewards = [row["reward"] for row in episode_rows]
    success_values = [
        row["is_success"] for row in episode_rows
        if not pd.isna(row["is_success"])
    ]

    summary = {
        "experiment": name,
        "hidden_size": hidden_size,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_latency_ms": float(np.mean(latencies_ms)),
        "p95_latency_ms": float(np.percentile(latencies_ms, 95)),
        "model_size_mb": zip_path.stat().st_size / (1024 ** 2),
        "total_parameters": count_parameters(model.policy),
        "success_rate": (
            float(np.mean(success_values))
            if success_values else np.nan
        ),
        "episodes": len(episode_rows),
    }

    print(
        f"\n[{name}] Reward: "
        f"{summary['mean_reward']:.2f} +/- "
        f"{summary['std_reward']:.2f}"
    )

    return summary, episode_rows


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Ruta al modelo PPO original",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--sim_speed", type=float, default=0)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--multiagent", action="store_true")

    parser.add_argument(
        "--track_scale",
        type=float,
        default=POWERLOOP_TRACK_SCALE,
    )
    parser.add_argument(
        "--track_position_variation",
        type=float,
        default=POWERLOOP_POSITION_VARIATION,
    )
    parser.add_argument(
        "--track_orientation_variation",
        type=float,
        default=POWERLOOP_ORIENTATION_VARIATION,
    )

    args = parser.parse_args()

    if args.episodes < 1:
        parser.error("--episodes debe ser al menos 1")
    if args.sim_speed < 0:
        parser.error("--sim_speed no puede ser negativo")

    output_dir = Path("results/compression")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Cargando modelo original...")
    baseline = PPO.load(
        args.model_path,
        device=EVAL_DEVICE,
    )

    experiments = [
        ("baseline", 64),
        ("structured_48", 48),
        ("structured_32", 32),
        ("structured_16", 16),
    ]

    summaries = []
    all_episodes = []

    for name, hidden_size in experiments:
        print(f"\n{'=' * 50}")
        print(f"Experimento: {name}")
        print(f"{'=' * 50}")

        if name == "baseline":
            candidate = baseline
        else:
            env_for_build = build_env(args)
            try:
                candidate = structured_prune_policy(
                    baseline,
                    hidden_size,
                    env_for_build,
                )
            finally:
                env_for_build.close()

        summary, episodes = evaluate_candidate(
            candidate,
            name,
            hidden_size,
            args,
            output_dir,
        )

        summaries.append(summary)
        all_episodes.extend(episodes)

    pd.DataFrame(summaries).to_csv(
        output_dir / "summary.csv",
        index=False,
    )

    pd.DataFrame(all_episodes).to_csv(
        output_dir / "episodes.csv",
        index=False,
    )

    print("\n=== COMPARACIÓN FINAL ===")
    print(pd.DataFrame(summaries).to_string(index=False))
    print(f"\nResultados guardados en: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
