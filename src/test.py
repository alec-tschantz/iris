from functools import partial
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import torch

from mcts import MCTSAgent
from envs import SingleProcessEnv, WorldModelEnv
from game import AgentEnv, EpisodeReplayEnv, Game
from models.actor_critic import ActorCritic
from models.world_model import WorldModel


@hydra.main(config_path="../config", config_name="trainer")
def main(cfg: DictConfig):
    device = torch.device("cpu")

    env_fn = partial(instantiate, config=cfg.env.test)
    test_env = SingleProcessEnv(env_fn)

    h, w, _ = test_env.env.unwrapped.observation_space.shape
    multiplier = 800 // h
    size = [h * multiplier, w * multiplier]

    tokenizer = instantiate(cfg.tokenizer)
    world_model = WorldModel(
        obs_vocab_size=tokenizer.vocab_size, act_vocab_size=test_env.num_actions, config=instantiate(cfg.world_model)
    )
    actor_critic = ActorCritic(**cfg.actor_critic, act_vocab_size=test_env.num_actions)
    agent = MCTSAgent(tokenizer, world_model, actor_critic).to(device)
    agent.load(Path("checkpoints/last.pt"), device)

    env = AgentEnv(agent, test_env, cfg.env.keymap, do_reconstruction=False)
    game = Game(env, keymap_name="empty", size=size, fps=15, verbose=False, record_mode=False)
    game.run()


if __name__ == "__main__":
    main()
