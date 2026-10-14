
"""Destilación por imitación de políticas PPO."""

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO


def collect_teacher_data(teacher, env, n_steps=20_000):
    """Recolecta observaciones y acciones deterministas del profesor."""
    if n_steps <= 0:
        raise ValueError("n_steps debe ser mayor que cero.")

    observations = []
    actions = []

    obs, _ = env.reset()

    for _ in range(n_steps):
        action, _ = teacher.predict(obs, deterministic=True)

        observations.append(
            np.asarray(obs, dtype=np.float32).copy()
        )
        actions.append(
            np.asarray(action, dtype=np.float32).copy()
        )

        obs, _, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            obs, _ = env.reset()

    return (
        np.asarray(observations, dtype=np.float32),
        np.asarray(actions, dtype=np.float32),
    )


def distill_student(
    teacher,
    env,
    observations,
    actions,
    hidden_size,
    epochs=10,
    batch_size=256,
    learning_rate=3e-4,
    seed=123,
):
    """Entrena el actor de un PPO estudiante para imitar al profesor."""

    if hidden_size <= 0:
        raise ValueError("hidden_size debe ser mayor que cero.")

    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs y batch_size deben ser mayores que cero.")

    if len(observations) == 0 or len(observations) != len(actions):
        raise ValueError(
            "Observaciones y acciones deben tener igual longitud no vacía."
        )

    if not hasattr(env.action_space, "low"):
        raise TypeError("Se requiere un espacio de acciones Box continuo.")

    torch.manual_seed(seed)
    np.random.seed(seed)

    student = PPO(
        "MlpPolicy",
        env,
        policy_kwargs={
            "net_arch": {
                "pi": [hidden_size, hidden_size],
                "vf": [hidden_size, hidden_size],
            }
        },
        learning_rate=learning_rate,
        n_steps=256,
        batch_size=64,
        seed=seed,
        device=teacher.device,
        verbose=0,
    )

    policy = student.policy

    obs_tensor = torch.as_tensor(
        observations,
        dtype=torch.float32,
        device=policy.device,
    )
    action_tensor = torch.as_tensor(
        actions,
        dtype=torch.float32,
        device=policy.device,
    )

    if action_tensor.ndim == 1:
        action_tensor = action_tensor.unsqueeze(-1)

    expected_dim = int(np.prod(env.action_space.shape))

    if action_tensor.shape[-1] != expected_dim:
        raise ValueError(
            f"Dimensión de acción incorrecta: "
            f"datos={action_tensor.shape[-1]}, entorno={expected_dim}."
        )

    low = torch.as_tensor(
        env.action_space.low,
        dtype=torch.float32,
        device=policy.device,
    ).reshape(1, -1)

    high = torch.as_tensor(
        env.action_space.high,
        dtype=torch.float32,
        device=policy.device,
    ).reshape(1, -1)

    # Solo entrenamos las capas del actor, no el crítico.
    actor_parameters = (
        list(policy.mlp_extractor.policy_net.parameters())
        + list(policy.action_net.parameters())
    )

    if not actor_parameters:
        raise RuntimeError("No se encontraron parámetros entrenables del actor.")

    optimizer = torch.optim.Adam(
        actor_parameters,
        lr=learning_rate,
    )

    n_samples = len(obs_tensor)

    for epoch in range(epochs):
        permutation = torch.randperm(
            n_samples,
            device=policy.device,
        )
        batch_losses = []

        for start in range(0, n_samples, batch_size):
            indices = permutation[start:start + batch_size]

            raw_actions = policy(
                obs_tensor[indices],
                deterministic=True,
            )[0]

            # Replica el recorte de acciones Box que realiza predict()
            # en la política PPO estándar, sin squash_output.
            predicted_actions = torch.maximum(
                torch.minimum(raw_actions, high),
                low,
            )

            loss = F.mse_loss(
                predicted_actions,
                action_tensor[indices],
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                actor_parameters,
                max_norm=1.0,
            )

            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))

        print(
            f"[Distill {hidden_size}-{hidden_size}] "
            f"Época {epoch + 1}/{epochs} "
            f"| MSE={np.mean(batch_losses):.6f}"
        )

    return student
