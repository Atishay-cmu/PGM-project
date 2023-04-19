from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
from torch.distributions import OneHotCategorical
from configs.dreamer.DreamerAgentConfig import A_RSSMStateDiscrete as RSSMStateDiscrete, A_RSSMStateCont as RSSMStateCont
from agent.models.DreamerModel import DreamerModel
from networks.dreamer.action import Actor


class DreamerController:

    def __init__(self, config):

        self.config = config
        self.expl_decay = config.EXPL_DECAY
        self.expl_noise = config.EXPL_NOISE
        self.expl_min = config.EXPL_MIN
        self.device = config.DEVICE
        self.init_rnns()
        self.init_buffer()

    def receive_params(self, params):
        self.model.load_state_dict(params['model'])
        self.actor.load_state_dict(params['actor'])

    def init_buffer(self):
        self.buffer = defaultdict(list)

    def init_rnns(self):
        self.prev_rnn_state = None
        self.prev_actions = None
        self.prev_a_latent_state = None
        self.prev_policy_latent_state = None

    def dispatch_buffer(self):
        total_buffer = {k: np.asarray(v, dtype=np.float32) for k, v in self.buffer.items()}
        last = np.zeros_like(total_buffer['done'])
        last[-1] = 1.0
        total_buffer['last'] = last
        self.init_rnns()
        self.init_buffer()
        return total_buffer

    def update_buffer(self, items):
        for k, v in items.items():
            if v is not None:
                self.buffer[k].append(v.squeeze(0).detach().clone().numpy())

    
    
    def get_latent_states(self, ac_model, obs, nn_mask):

      if(self.prev_a_latent_state is None):
        self.prev_a_latent_state = ac_model.transition.initial_state(obs.size(0), obs.size(1), device=self.device)
      action_model_state = ac_model.transition(obs.to(self.device), self.prev_a_latent_state, nn_mask)

      return action_model_state

    def select_latent_action(self, actor, obs, nn_mask):

      if(self.prev_policy_latent_state is None):
        self.prev_policy_latent_state = actor.initial_state(obs.size(0), obs.size(1), device=self.device)
      actor_state = actor(obs.to(self.device), self.prev_policy_latent_state, nn_mask)

      return actor_state

    
    def step(self, observations, avail_actions, nn_mask, model, action_model, actor):
        """"
        Compute policy's action distribution from inputs, and sample an
        action. Calls the model to produce mean, log_std, value estimate, and
        next recurrent state.  Moves inputs to device and returns outputs back
        to CPU, for the sampler.  Advances the recurrent state of the agent.
        (no grad)
        """
        model.eval()
        actor.eval()
        action_model.eval()
        with torch.no_grad():

          action_model_state = self.get_latent_states(action_model, observations, nn_mask)
          action_model_state.map(lambda x:x.detach())
          policy_latent_state = None
          if(self.config.USE_LATENT_ACTIONS):

            policy_latent_state = self.select_latent_action(actor, observations, nn_mask) #method 1(policy consists of full transition model)
            policy_latent_state.map(lambda x:x.detach())
            latent_action = policy_latent_state.stoch

            # if(self.config.A_DISCRETE_LATENTS):
            #   _, latent_action = actor(action_model_state.deter)
            # else:
            #   _, _, latent_action = actor(action_model_state.deter)


            action, pi = action_model.decode(latent_action)        
            action.detach()
            pi.detach() #logits

          else:
            action, pi = actor(observations.to(self.device))
            action.detach()
            pi.detach() #logits

          if avail_actions is not None:
              pi[avail_actions == 0] = -1e10
              action_dist = OneHotCategorical(logits=pi)
              action = action_dist.sample()

          self.advance_rnns(action_model, action, action_model_state, policy_latent_state)
          self.prev_actions = action.detach().clone()
          observations.cpu()
        model.train()
        actor.train()
        action_model.train()
        return action.squeeze(0).clone().cpu()

    def advance_rnns(self, action_model, action, action_model_state, policy_latent):

        action_embed = action_model.action_encoder(action)
        x = torch.cat([action_model_state.deter, action_embed], dim=-1)
        if(self.config.A_DISCRETE_LATENTS):
          logits, stoch_state = action_model.representation._stochastic_posterior_model(x)
          posterior_state = RSSMStateDiscrete(logits=logits, stoch=stoch_state, deter=action_model_state.deter)
        else:
          mean, std, stoch_state = action_model.representation._stochastic_posterior_model(x)
          posterior_state = RSSMStateCont(mean = mean, std = std, stoch=stoch_state, deter=action_model_state.deter)        
        
        self.prev_a_latent_state = deepcopy(posterior_state)
        self.prev_policy_latent_state = deepcopy(policy_latent)

    def exploration(self, action):
        """
        :param action: action to take, shape (1,)
        :return: action of the same shape passed in, augmented with some noise
        """
        for i in range(action.shape[0]):
            if np.random.uniform(0, 1) < self.expl_noise:
                index = torch.randint(0, action.shape[-1], (1, ), device=action.device)
                transformed = torch.zeros(action.shape[-1])
                transformed[index] = 1.
                action[i] = transformed
        self.expl_noise *= self.expl_decay
        self.expl_noise = max(self.expl_noise, self.expl_min)
        return action
