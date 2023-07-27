from pathlib import Path

import numpy as np
from einops import rearrange

import torch
from torch.distributions.categorical import Categorical
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
        self.keys_values_wm = None
        self.num_observations_tokens = None
        self.obs_token = None

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

    @torch.no_grad()
    def refresh_keys_values_with_initial_obs_tokens(self, obs_tokens: torch.LongTensor) -> torch.FloatTensor:
        n, num_observations_tokens = obs_tokens.shape
        assert num_observations_tokens == self.num_observations_tokens
        self.keys_values_wm = self.world_model.transformer.generate_empty_keys_values(
            n=n, max_tokens=self.world_model.config.max_tokens
        )
        outputs_wm = self.world_model(obs_tokens, past_keys_values=self.keys_values_wm)
         # (B, K, E)
        return outputs_wm.output_sequence 

    @torch.no_grad()
    def reset_from_initial_observations(self, observations: torch.FloatTensor) -> None:
         # (B, C, H, W) -> (B, K)
        obs_tokens = self.tokenizer.encode(observations, should_preprocess=True).tokens 
        _, num_observations_tokens = obs_tokens.shape
        if self.num_observations_tokens is None:
            self.num_observations_tokens = num_observations_tokens

        _ = self.refresh_keys_values_with_initial_obs_tokens(obs_tokens)
        self.obs_tokens = obs_tokens

    def act(self, obs: torch.FloatTensor, should_sample: bool = True, temperature: float = 1.0, depth: int = 0) -> torch.LongTensor:
        # obs (B, C, H, W) in [0, 1]

        # ==== get action from actor critic ====
        # logits_actions (B, A)
        logits_actions = self.actor_critic(obs).logits_actions[:, -1] / temperature
        # act_token (1)
        act_token = Categorical(logits=logits_actions).sample() if should_sample else logits_actions.argmax(dim=-1)

        # ==== check if this is initial observation ====
        if self.keys_values_wm is None:
            self.reset_from_initial_observations(obs)

        # ==== predict next tokens ====
        output_sequence, obs_tokens = [], []
        num_passes = 1 + self.num_observations_tokens
        # act_token_wm (B, 1)
        act_token_wm = act_token.reshape(-1, 1)

        if self.keys_values_wm.size + num_passes > self.world_model.config.max_tokens:
            _ = self.refresh_keys_values_with_initial_obs_tokens(self.obs_tokens)

        for k in range(num_passes):
            outputs_wm = self.world_model(act_token_wm, past_keys_values=self.keys_values_wm)
            output_sequence.append(outputs_wm.output_sequence)

            if k < self.num_observations_tokens:
                token = Categorical(logits=outputs_wm.logits_observations).sample()
                obs_tokens.append(token)

        # output sequence (B, 1 + K, E)
        output_sequence = torch.cat(output_sequence, dim=1)
        # obs_tokens (B, K)
        self.obs_tokens = torch.cat(obs_tokens, dim=1)

        # ==== decode observation ====
        embedded_tokens = self.tokenizer.embedding(self.obs_tokens)     # (B, K, E)
        z = rearrange(embedded_tokens, 'b (h w) e -> b e h w', h=int(np.sqrt(self.num_observations_tokens)))
        rec = self.tokenizer.decode(z, should_postprocess=True)         # (B, C, H, W)
        obs = torch.clamp(rec, 0, 1)

        # if depth < 10:
        #     _ = self.act(obs, should_sample=should_sample, temperature=temperature, depth=depth + 1)

        return act_token

