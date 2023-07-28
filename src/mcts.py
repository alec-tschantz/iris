from pathlib import Path
from typing import Tuple

import numpy as np
from einops import rearrange

import torch
import torch.nn as nn

from models.actor_critic import ActorCritic
from models.tokenizer import Tokenizer
from models.world_model import WorldModel
from utils import extract_state_dict


"""
B - batch size (1) 
K - number of tokens per frame (16)
C - number of pixel channels (3)
H - height of pixels (64)
W - width of pixels (64)
N - vocabulary size (512)
A - number of actions (4)
"""


class MCTSAgent(nn.Module):
    def __init__(self, tokenizer: Tokenizer, world_model: WorldModel, actor_critic: ActorCritic):
        super().__init__()
        self.tokenizer = tokenizer
        self.world_model = world_model
        self.actor_critic = actor_critic
        self.tokens = []

    @property
    def device(self):
        return self.actor_critic.conv1.weight.device

    def load(
        self,
        path_to_checkpoint: Path,
        device: torch.device,
        load_tokenizer: bool = True,
        load_world_model: bool = True,
        load_actor_critic: bool = True,
    ) -> None:
        agent_state_dict = torch.load(path_to_checkpoint, map_location=device)
        if load_tokenizer:
            self.tokenizer.load_state_dict(extract_state_dict(agent_state_dict, "tokenizer"))
        if load_world_model:
            self.world_model.load_state_dict(extract_state_dict(agent_state_dict, "world_model"))
        if load_actor_critic:
            self.actor_critic.load_state_dict(extract_state_dict(agent_state_dict, "actor_critic"))

    def decode_obs_tokens(self, obs_tokens: torch.LongTensor) -> torch.FloatTensor:
        num_obs_tokens = obs_tokens.shape[1]
        # (B, K) -> (B, K, E)
        embedded_tokens = self.tokenizer.embedding(obs_tokens)
        z = rearrange(embedded_tokens, "b (h w) e -> b e h w", h=int(np.sqrt(num_obs_tokens)))
        # (B, C, H, W)
        obs = self.tokenizer.decode(z, should_postprocess=True)
        return torch.clamp(obs, 0, 1)

    def predict_next_obs_tokens(
        self, tokens: torch.LongTensor, num_tokens: int, temperature: float = 1.0
    ) -> torch.LongTensor:
        for _ in range(num_tokens):
            # (B, T, N)
            logits = self.world_model(tokens).logits_observations
            # (B, N)
            logits = logits[:, -1, :] / temperature
            # (B, N)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            # (B, )
            _, idx_next = torch.topk(probs, k=1, dim=-1)
            # (B, T + 1)
            tokens = torch.cat((tokens, idx_next), dim=1)

        # (B, K)
        return tokens[:, -num_tokens:]

    def step(self, obs: torch.FloatTensor, temperature: float = 1.0) -> Tuple[torch.LongTensor, torch.FloatTensor]:
        # (B, C, H, W) -> (B, K)
        obs_tokens = self.tokenizer.encode(obs, should_preprocess=True).tokens
        num_obs_tokens = obs_tokens.shape[1]
        max_blocks = self.world_model.config.max_blocks

        # (B, A)
        logits_actions = self.actor_critic(obs).logits_actions[:, -1] / temperature
        # (B, )
        act_tokens = logits_actions.argmax(dim=-1)

        # (B, K + 1)
        tokens = torch.cat((obs_tokens, act_tokens.reshape(-1, 1)), dim=1)

        # [(B, T)]
        self.tokens.append(tokens)

        if len(self.tokens) >= max_blocks - 1:
            self.tokens = self.tokens[-max_blocks + 1 :]

        # (B, T)
        tokens = torch.cat(self.tokens, dim=1)

        # (B, K)
        next_obs_tokens = self.predict_next_obs_tokens(tokens, num_obs_tokens)

        # (B, C, H, W)
        next_obs = self.decode_obs_tokens(next_obs_tokens)

        return act_tokens, next_obs
