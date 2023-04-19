import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from agent.memory.DreamerMemory import DreamerMemory
from agent.models.DreamerModel import DreamerModel
from agent.models.ActionModel import ActionModel
from agent.optim.loss import model_loss, actor_loss, value_loss, actor_rollout
from agent.optim.action_loss import action_model_loss
from agent.optim.utils import advantage
from environments import Env
from networks.dreamer.action import Actor
from networks.dreamer.critic import MADDPGCritic
from networks.dreamer.action_model_rnns import RSSMRepresentation, RSSMTransition, DiscreteLatentDist, NormalLatentDist

def orthogonal_init(tensor, gain=1):
    if tensor.ndimension() < 2:
        raise ValueError("Only tensors with 2 or more dimensions are supported")

    rows = tensor.size(0)
    cols = tensor[0].numel()
    flattened = tensor.new(rows, cols).normal_(0, 1)

    if rows < cols:
        flattened.t_()

    # Compute the qr factorization
    u, s, v = torch.svd(flattened, some=True)
    if rows < cols:
        u.t_()
    q = u if tuple(u.shape) == (rows, cols) else v
    with torch.no_grad():
        tensor.view_as(q).copy_(q)
        tensor.mul_(gain)
    return tensor


def initialize_weights(mod, scale=1.0, mode='ortho'):
    for p in mod.parameters():
        if mode == 'ortho':
            if len(p.data.shape) >= 2:
                orthogonal_init(p.data, gain=scale)
        elif mode == 'xavier':
            if len(p.data.shape) >= 2:
                torch.nn.init.xavier_uniform_(p.data)


class DreamerLearner:

    def __init__(self, config):
        self.config = config
        self.model = DreamerModel(config).to(config.DEVICE).eval()

        self.action_model = ActionModel(config).to(config.DEVICE).eval()
        if(not self.config.USE_LATENT_ACTIONS):
          # self.actor = Actor(config.A_FEAT, config.ACTION_SIZE, config.ACTION_HIDDEN, config.ACTION_LAYERS).to( 
          #     config.DEVICE)
          self.actor = Actor(config.IN_DIM, config.ACTION_SIZE, config.ACTION_HIDDEN, config.ACTION_LAYERS).to( 
              config.DEVICE)
        else:
          self.actor =  RSSMTransition(config, config.A_MODEL_HIDDEN).to(config.DEVICE)

          # if(config.A_DISCRETE_LATENTS):
          #     self.actor = DiscreteLatentDist(config.A_DETERMINISTIC, config.A_N_CATEGORICALS, config.A_N_CLASSES,
          #                                                   config.A_MODEL_HIDDEN)
          # else:
          #     self.actor = NormalLatentDist(config.A_DETERMINISTIC, config.A_N_LATENTS,
          #                                         config.A_MODEL_HIDDEN)


        self.critic = MADDPGCritic(config.FEAT, config.HIDDEN).to(config.DEVICE)
        initialize_weights(self.model, mode='xavier')
        initialize_weights(self.action_model, mode='xavier')
        initialize_weights(self.actor) #mode='xavier'
        initialize_weights(self.critic, mode='xavier')
        self.old_critic = deepcopy(self.critic)
        self.replay_buffer = DreamerMemory(config.CAPACITY, config.SEQ_LENGTH, config.ACTION_SIZE, config.IN_DIM, 2,
                                           config.DEVICE, config.ENV_TYPE)
        self.entropy = config.ENTROPY
        self.step_count = -1
        self.cur_update = 1
        self.accum_samples = 0
        self.total_samples = 0
        self.init_optimizers()
        self.n_agents = 2
        Path(config.LOG_FOLDER).mkdir(parents=True, exist_ok=True)
        global wandb
        import wandb
        wandb.init(project="10708_pro", dir=config.LOG_FOLDER)

    def init_optimizers(self):
        self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.MODEL_LR)
        self.action_model_optimizer = torch.optim.Adam(self.action_model.parameters(), lr=self.config.MODEL_LR)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.ACTOR_LR, weight_decay=0.00001)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.VALUE_LR)

    def params(self):
        return {'model': {k: v.cpu() for k, v in self.model.state_dict().items()},
                'action_model': {k: v.cpu() for k, v in self.action_model.state_dict().items()},
                'actor': {k: v.cpu() for k, v in self.actor.state_dict().items()},
                'critic': {k: v.cpu() for k, v in self.critic.state_dict().items()}}

    def step(self, rollout, steps_done):
        if self.n_agents != rollout['action'].shape[-2]:
            self.n_agents = rollout['action'].shape[-2]

        self.accum_samples += len(rollout['action'])
        self.total_samples += len(rollout['action'])
        self.replay_buffer.append(rollout['observation'], rollout['action'], rollout['reward'], rollout['done'],
                                  rollout['fake'], rollout['last'], rollout.get('avail_action'))
        self.step_count += 1
        if self.accum_samples < self.config.N_SAMPLES:
            return

        if len(self.replay_buffer) < self.config.MIN_BUFFER_SIZE:
            return

        self.accum_samples = 0
        sys.stdout.flush()

        for i in range(self.config.MODEL_EPOCHS):
            samples1 = self.replay_buffer.sample1(self.config.MODEL_BATCH_SIZE)
            self.train_model(samples1)

            # NOTE: commented so that we only run the action model training    
            # if(steps_done < 2000):
              # print(steps_done)
            samples2 = self.replay_buffer.sample2(self.config.MODEL_BATCH_SIZE)
            self.train_action_model(samples2)           

        # NOTE: commented so that we only run the action model training
        # if(steps_done > 2000):
        #   for i in range(self.config.EPOCHS):
        #       samples = self.replay_buffer.sample1(self.config.BATCH_SIZE)
        #       self.train_agent(samples)

    def train_model(self, samples):
        self.model.train()
        loss = model_loss(self.config, self.model, samples['observation'], samples['action'], samples['av_action'],
                          samples['reward'], samples['done'], samples['fake'], samples['last'])
        self.apply_optimizer(self.model_optimizer, self.model, loss, self.config.GRAD_CLIP)
        self.model.eval()


    def index_latents(self, y, idxes):
       return y.map(lambda x: x[idxes])

    def train_action_model(self, samples):
        self.action_model.train()
        loss = action_model_loss(self.config, self.action_model, samples['observation'], samples['action'], samples['av_action'],
                          samples['reward'], samples['done'], samples['fake'], samples['last'])
        self.apply_optimizer(self.action_model_optimizer, self.action_model, loss, self.config.GRAD_CLIP)
        self.action_model.eval()
  

    def train_agent(self, samples): #TODO
        #Freeze action model
        for par in self.action_model.parameters():
            par.requires_grad = False

        list_vals, am_priors, am_posteriors, prev_policy_latents = actor_rollout(samples['observation'],
                                                                            samples['action'],
                                                                            samples['last'], self.model, self.action_model,
                                                                            self.actor,
                                                                            self.critic if self.config.ENV_TYPE == Env.STARCRAFT
                                                                            else self.old_critic,
                                                                            self.config)
        actions, av_actions, old_policy, imag_feat, imag_obs, returns = list_vals

        adv = returns.detach() - self.critic(imag_feat, actions).detach()
        if self.config.ENV_TYPE == Env.STARCRAFT:
            adv = advantage(adv)
        wandb.log({'Agent/Returns': returns.mean()})
        #print(actions.shape, imag_feat.shape)
        for epoch in range(self.config.PPO_EPOCHS):
            inds = np.random.permutation(actions.shape[0])
            step = 2000
            for i in range(0, len(inds), step):
                self.cur_update += 1
                idx = inds[i:i + step]

                loss = actor_loss(self.config, self.index_latents(am_priors, idx), self.index_latents(am_posteriors, idx), self.index_latents(prev_policy_latents, idx),\
                                               imag_feat[idx], imag_obs[idx], actions[idx], av_actions[idx] if av_actions is not None else None,
                                  old_policy[idx], adv[idx], self.actor, self.action_model, self.entropy)
                self.apply_optimizer(self.actor_optimizer, self.actor, loss, self.config.GRAD_CLIP_POLICY)
                self.entropy *= self.config.ENTROPY_ANNEALING

                val_loss = value_loss(self.config, self.index_latents(am_priors, idx), self.index_latents(am_posteriors, idx), self.index_latents(prev_policy_latents, idx),\
                                                           imag_obs[idx], self.critic, actions[idx], imag_feat[idx], returns[idx])
                if np.random.randint(20) == 9:
                    wandb.log({'Agent/val_loss': val_loss, 'Agent/actor_loss': loss})
                self.apply_optimizer(self.critic_optimizer, self.critic, val_loss, self.config.GRAD_CLIP_POLICY)
                if self.config.ENV_TYPE == Env.FLATLAND and self.cur_update % self.config.TARGET_UPDATE == 0:
                    self.old_critic = deepcopy(self.critic)
        #Unfreeze action model
        for par in self.action_model.parameters():
            par.requires_grad = True

    def apply_optimizer(self, opt, model, loss, grad_clip):
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
