"""
Natron reinforcement learning training script (PPO).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from natron.dataset_loader import NatronDatasetConfig, NatronFeatureEngineer, load_ohlcv
from natron.rl import MarketEnvConfig, MarketEnvironment, PPOAgent, PPOConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Natron PPO agent.")
    parser.add_argument("--config", type=str, required=True, help="Path to natron_config.yaml")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--total-steps", type=int, default=200_000)
    return parser.parse_args()


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "rl_training.log"),
            logging.StreamHandler(),
        ],
    )


def load_configuration(path: Path) -> Dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as fp:
        return yaml.safe_load(fp)


def load_features_for_rl(config: Dict[str, Any]) -> Dict[str, np.ndarray]:
    dataset_cfg = NatronDatasetConfig(
        sequence_length=config["data"]["sequence_length"],
        normalization=config["data"]["normalize"],
        feature_cache_path=None,
        cache_features=False,
        min_samples=500,
    )
    raw_df = load_ohlcv(Path(config["data"]["csv_path"]))
    feature_engineer = NatronFeatureEngineer(dataset_cfg)
    feature_df = feature_engineer.fit_transform(raw_df)
    aligned_prices = raw_df.loc[feature_df.index]["close"].to_numpy(dtype=np.float32)
    features = feature_df.to_numpy(dtype=np.float32)
    if dataset_cfg.normalization:
        mean = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0, keepdims=True) + 1e-6
        features = (features - mean) / std
    return {"features": features, "prices": aligned_prices}


def evaluate_policy(env: MarketEnvironment, agent: PPOAgent, episodes: int = 5) -> Dict[str, float]:
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        cumulative_reward = 0.0
        while not (done or truncated):
            action, _, _ = agent.act(obs)
            obs, reward, done, truncated, info = env.step(action)
            cumulative_reward += reward
        returns.append(cumulative_reward)
    return {"avg_return": float(np.mean(returns)), "std_return": float(np.std(returns))}


def main() -> None:
    args = parse_args()
    config = load_configuration(Path(args.config))
    output_dir = Path(config["experiment"]["output_dir"])
    setup_logging(output_dir)

    device = torch.device(args.device)
    torch.manual_seed(config["experiment"]["seed"])
    np.random.seed(config["experiment"]["seed"])

    data = load_features_for_rl(config)
    features = data["features"]
    prices = data["prices"]

    env_cfg = MarketEnvConfig(
        sequence_length=config["data"]["sequence_length"],
        feature_dim=features.shape[1],
        episode_length=config["reinforcement"]["steps"] // config["data"]["batch_size"],
        transaction_cost=config["reinforcement"]["reward"]["turnover_penalty"] * 0.001,
        turnover_penalty=config["reinforcement"]["reward"]["turnover_penalty"],
        drawdown_penalty=config["reinforcement"]["reward"]["drawdown_penalty"],
    )
    env = MarketEnvironment(features, prices, env_cfg)
    agent_config = PPOConfig(
        learning_rate=3e-4,
        batch_size=config["data"]["batch_size"] * 4,
        mini_batch_size=config["data"]["batch_size"],
    )
    agent = PPOAgent(env.observation_space, env.action_space, agent_config, device)

    total_steps = args.total_steps
    rollout_steps = agent_config.batch_size

    observation, _ = env.reset()
    memory = {
        "observations": [],
        "actions": [],
        "log_probs": [],
        "values": [],
        "rewards": [],
        "dones": [],
    }

    step_count = 0
    iteration = 0
    while step_count < total_steps:
        iteration += 1
        for _ in range(rollout_steps):
            action, log_prob, value = agent.act(observation)
            next_obs, reward, done, truncated, info = env.step(action)

            memory["observations"].append(observation.copy())
            memory["actions"].append(action)
            memory["log_probs"].append(log_prob)
            memory["values"].append(value)
            memory["rewards"].append(reward)
            memory["dones"].append(float(done or truncated))

            observation = next_obs
            step_count += 1

            if done or truncated:
                observation, _ = env.reset()

            if step_count >= total_steps:
                break

        observations_tensor = torch.from_numpy(np.stack(memory["observations"])).to(agent.device, dtype=torch.float32)
        observations_tensor = observations_tensor.view(-1, agent.obs_dim)
        actions_tensor = torch.tensor(memory["actions"], dtype=torch.int64, device=agent.device)
        log_probs_tensor = torch.tensor(memory["log_probs"], dtype=torch.float32, device=agent.device)
        values_tensor = torch.tensor(memory["values"], dtype=torch.float32, device=agent.device)
        rewards_tensor = torch.tensor(memory["rewards"], dtype=torch.float32, device=agent.device)
        dones_tensor = torch.tensor(memory["dones"], dtype=torch.float32, device=agent.device)

        advantages, returns = agent.compute_advantages(rewards_tensor, values_tensor, dones_tensor)

        buffer = {
            "observations": observations_tensor,
            "actions": actions_tensor,
            "log_probs": log_probs_tensor,
            "advantages": advantages,
            "returns": returns,
        }
        metrics = agent.update(buffer)

        eval_stats = evaluate_policy(env, agent, episodes=3)
        logging.info(
            "Iter %d | Steps %d/%d | Policy %.4f | Value %.4f | Entropy %.4f | Eval %.4f ± %.4f",
            iteration,
            step_count,
            total_steps,
            metrics["policy_loss"],
            metrics["value_loss"],
            metrics["entropy"],
            eval_stats["avg_return"],
            eval_stats["std_return"],
        )

        memory = {key: [] for key in memory}

    policy_path = output_dir / "ppo_policy.pt"
    torch.save({"policy_state_dict": agent.policy.state_dict(), "config": agent_config}, policy_path)
    logging.info("Saved PPO policy to %s", policy_path)


if __name__ == "__main__":
    main()
