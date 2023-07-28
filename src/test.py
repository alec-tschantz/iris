from datetime import datetime
from pathlib import Path

import gym
import hydra
import torch
import torchvision
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig

from mcts import MCTSAgent
from models.actor_critic import ActorCritic
from models.world_model import WorldModel
from utils import make_video


def save_recording(frames: np.ndarray):
    record_dir = Path("media") / "recordings"
    record_dir.mkdir(exist_ok=True, parents=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    np.save(record_dir / timestamp, frames)
    make_video(record_dir / f"{timestamp}.mp4", fps=15, frames=frames)
    print(f"Saved recording {timestamp}.")


def render(obs: torch.FloatTensor) -> np.ndarray:
    return obs[0].mul(255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)


def init_obs(env: gym.Env) -> torch.FloatTensor:
    return torchvision.transforms.functional.to_tensor(env.reset()).unsqueeze(0)


@hydra.main(config_path="../config", config_name="trainer")
def main(cfg: DictConfig):
    device = torch.device("cpu")

    env = instantiate(cfg.env.test)
    tokenizer = instantiate(cfg.tokenizer)
    wm_config = instantiate(cfg.world_model)
    vocab_size = tokenizer.vocab_size
    num_actions = env.action_space.n

    world_model = WorldModel(obs_vocab_size=vocab_size, act_vocab_size=num_actions, config=wm_config)
    actor_critic = ActorCritic(**cfg.actor_critic, act_vocab_size=num_actions)
    agent = MCTSAgent(tokenizer, world_model, actor_critic).to(device)
    agent.load(Path("checkpoints/last.pt"), device)

    obs = init_obs(env)
    agent.actor_critic.reset(1)

    episode_buffer = [render(obs)]
    
    for t in range(100):
        print(f"step: {t}")
        with torch.no_grad():
            _, obs = agent.step(obs)
        episode_buffer.append(render(obs))

    save_recording(np.stack(episode_buffer))


if __name__ == "__main__":
    main()
