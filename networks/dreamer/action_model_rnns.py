import torch
import torch.nn as nn
from torch.distributions import OneHotCategorical, MultivariateNormal
from torch.nn import functional as F
from configs.dreamer.DreamerAgentConfig import A_RSSMStateDiscrete as RSSMStateDiscrete, A_RSSMStateCont as RSSMStateCont
from networks.transformer.layers import AttentionEncoder
from networks.dreamer.vae import Encoder, Decoder
# from agent.optim.utils import OneHotDist

def stack_states(rssm_states: list, config, dim): 
    return reduce_states(rssm_states, dim, config, torch.stack)

  
def cat_states(rssm_states: list, config, dim):
    return reduce_states(rssm_states, dim, config, torch.cat)


def reduce_states(rssm_states: list, dim, config, func):
    if(config.A_DISCRETE_LATENTS):
      return RSSMStateDiscrete(*[func([getattr(state, key) for state in rssm_states], dim=dim)
                        for key in rssm_states[0].__dict__.keys()])
    else:
      return RSSMStateCont(*[func([getattr(state, key) for state in rssm_states], dim=dim)
                  for key in rssm_states[0].__dict__.keys()])

class DiscreteLatentDist(nn.Module):
    def __init__(self, in_dim, n_categoricals, n_classes, hidden_size, use_mix = True):
        super().__init__()
        self.n_categoricals = n_categoricals
        self.n_classes = n_classes
        self.use_mix = use_mix
        self.dists = nn.Sequential(nn.Linear(in_dim, hidden_size),
                                   nn.ReLU(),
                                   nn.Linear(hidden_size, n_classes * n_categoricals))

    def forward(self, x):
        logits = self.dists(x).view(x.shape[:-1] + (self.n_categoricals, self.n_classes))
        class_dist = OneHotCategorical(logits=logits)
        #class_dist = OneHotDist(logits=logits, unimix_ratio=0., use_mix = self.use_mix)
        one_hot = class_dist.sample()
        latents = one_hot #+ class_dist.probs - class_dist.probs.detach()
 
        
        return  logits.view(x.shape[:-1] + (-1,)), latents.view(x.shape[:-1] + (-1,)), class_dist.log_prob(one_hot)

class NormalLatentDist(nn.Module):
    def __init__(self, in_dim, n_latents, hidden_size, use_mix = True):
        super().__init__()
        self.n_latents = n_latents
        self.dists = nn.Sequential(nn.Linear(in_dim, hidden_size),
                                   nn.ReLU(),
                                   nn.Linear(hidden_size, n_latents*2))
        if(use_mix):
          self.unimix_ratio = 0.0
        else:
          self.unimix_ratio = 0.0

    def forward(self, x):

        x_latents = self.dists(x).view(x.shape[0], x.shape[1], self.n_latents*2)
        mean, std = torch.chunk(x_latents, 2, dim=-1)
        std = F.softplus(std) + 1e-5
        latents = mean + torch.randn_like(std) * std #+ self.unimix_ratio*(torch.zeros_like(mean) + torch.randn_like(std)*torch.ones_like(std))
        return mean, std, latents


class RSSMTransition(nn.Module):
    def __init__(self, config, hidden_size=200, activation=nn.ReLU, use_mix = True):
        super().__init__()
        self.config = config
        self._stoch_size = config.A_STOCHASTIC
        self._deter_size = config.A_DETERMINISTIC
        self._hidden_size = hidden_size
        self._activation = activation
        if(self.config.A_USE_RNN):
          self._cell = nn.GRU(hidden_size, self._deter_size)
        # if(config.OPP):
        #   self.obs_encoder = Encoder(in_dim=config.IN_DIM + (config.NUM_AGENTS-1)*(config.ACTION_SIZE), hidden=config.HIDDEN, embed=config.OBS_EMBED)
        # else:
        self.obs_encoder = Encoder(in_dim=config.IN_DIM, hidden=config.A_HIDDEN, embed=config.OBS_EMBED)

        #self._attention_stack = AttentionEncoder(3, hidden_size, hidden_size, dropout=0.1)
        self._rnn_input_model = self._build_rnn_input_model(config.OBS_EMBED + self._stoch_size)
        if(config.A_DISCRETE_LATENTS):
          self._stochastic_prior_model = DiscreteLatentDist(self._deter_size, config.A_N_CATEGORICALS, config.A_N_CLASSES,
                                                            self._hidden_size, use_mix)
        else:
          self._stochastic_prior_model = NormalLatentDist(self._deter_size, config.A_N_LATENTS,
                                                  self._hidden_size, use_mix)

    def _build_rnn_input_model(self, in_dim):
        rnn_input_model = [nn.Linear(in_dim, self._hidden_size)]
        rnn_input_model += [self._activation()]
        return nn.Sequential(*rnn_input_model)

    def initial_state(self, batch_size, n_agents, **kwargs):
        if(self.config.A_DISCRETE_LATENTS):
          return RSSMStateDiscrete(stoch=torch.zeros(batch_size, n_agents, self._stoch_size, **kwargs),
                         logits=torch.zeros(batch_size, n_agents, self._stoch_size, **kwargs),
                         deter=torch.zeros(batch_size, n_agents, self._deter_size, **kwargs),
                         log_probs=torch.zeros(batch_size, n_agents, self.config.A_N_CATEGORICALS, **kwargs))
        else:
          return RSSMStateCont(stoch=torch.zeros(batch_size, n_agents, self._stoch_size, **kwargs),
                mean=torch.zeros(batch_size, n_agents, self.config.A_N_LATENTS, **kwargs),
                std = torch.zeros(batch_size, n_agents, self.config.A_N_LATENTS, **kwargs),
                deter=torch.zeros(batch_size, n_agents, self._deter_size, **kwargs))
                
    def forward(self, obs, prev_states, mask=None):
        batch_size = obs.shape[0]
        n_agents = obs.shape[1]
        obs_embed = self.obs_encoder(obs)
        stoch_input = self._rnn_input_model(torch.cat([obs_embed, prev_states.stoch], dim=-1))
        attn = stoch_input#self._attention_stack(stoch_input, mask=mask)
        if(self.config.A_USE_RNN):
          deter_state = self._cell(attn.reshape(1, batch_size * n_agents, -1),
                                  prev_states.deter.reshape(1, batch_size * n_agents, -1))[0].reshape(batch_size, n_agents, -1)
        else:
          deter_state = attn
        if(self.config.A_DISCRETE_LATENTS):
          logits, stoch_state, log_probs = self._stochastic_prior_model(deter_state)
          rssm_state = RSSMStateDiscrete(logits=logits, stoch=stoch_state, deter=deter_state, log_probs = log_probs)
        else:
          mean, std, stoch_state = self._stochastic_prior_model(deter_state)
          rssm_state = RSSMStateCont(mean = mean, std = std, stoch=stoch_state, deter=deter_state)
        return rssm_state




class RSSMRepresentation(nn.Module):
    def __init__(self, config, transition_model: RSSMTransition):
        super().__init__()
        self.config = config
        self._transition_model = transition_model
        self._stoch_size = config.A_STOCHASTIC
        self._deter_size = config.A_DETERMINISTIC
        if(config.A_DISCRETE_LATENTS):
          self._stochastic_posterior_model = DiscreteLatentDist(self._deter_size + config.ACTION_EMBED, config.A_N_CATEGORICALS,
                                                                config.A_N_CLASSES, config.A_HIDDEN)
        else:
          self._stochastic_posterior_model = NormalLatentDist(self._deter_size + config.ACTION_EMBED, config.A_N_LATENTS,
                                                  config.A_HIDDEN)

    def initial_state(self, batch_size, n_agents, **kwargs):
        if(self.config.A_DISCRETE_LATENTS):
          return RSSMStateDiscrete(stoch=torch.zeros(batch_size, n_agents, self._stoch_size, **kwargs),
                         logits=torch.zeros(batch_size, n_agents, self._stoch_size, **kwargs),
                         deter=torch.zeros(batch_size, n_agents, self._deter_size, **kwargs),
                         log_probs=torch.zeros(batch_size, n_agents, self.config.A_N_CATEGORICALS, **kwargs))
        else:
          return RSSMStateCont(stoch=torch.zeros(batch_size, n_agents, self._stoch_size, **kwargs),
                mean=torch.zeros(batch_size, n_agents, self.config.A_N_LATENTS, **kwargs),
                std = torch.zeros(batch_size, n_agents, self.config.A_N_LATENTS, **kwargs),
                deter=torch.zeros(batch_size, n_agents, self._deter_size, **kwargs))

    def forward(self, action_embed, obs, prev_states, mask=None):
        """
        :param obs_embed: size(batch, n_agents, obs_size)
        :param prev_actions: size(batch, n_agents, action_size)
        :param prev_states: size(batch, n_agents, state_size)
        :return: RSSMState, global_state: size(batch, 1, global_state_size)
        """
        prior_states = self._transition_model(obs, prev_states, mask)
        x = torch.cat([prior_states.deter, action_embed], dim=-1)
        if(self.config.A_DISCRETE_LATENTS):
          logits, stoch_state, log_probs = self._stochastic_posterior_model(x)
          posterior_states = RSSMStateDiscrete(logits=logits, stoch=stoch_state, deter=prior_states.deter, log_probs = log_probs)
        else:
          mean, std, stoch_state = self._stochastic_posterior_model(x)
          posterior_states = RSSMStateCont(mean = mean, std = std, stoch=stoch_state, deter=prior_states.deter)

        return prior_states, posterior_states

def rollout_representation(config, representation_model, steps, action_embed, obs, prev_states, done): #should it be last or fake? should I mask fake actions?
    """
        Roll out the model with actions and observations from data.
        :param steps: number of steps to roll out
        :param obs_embed: size(time_steps, batch_size, n_agents, embedding_size)
        :param action: size(time_steps, batch_size, n_agents, action_size)
        :param prev_states: RSSM state, size(batch_size, n_agents, state_size)
        :return: prior, posterior states. size(time_steps, batch_size, n_agents, state_size)
        """
    priors = []
    posteriors = []
    for t in range(steps):

        prior_states, posterior_states = representation_model(action_embed[t], obs[t], prev_states)
        prev_states = posterior_states.map(lambda x: x * (1.0 - done[t])) #no need to do t-1 here. t is right.
        priors.append(prior_states)
        posteriors.append(posterior_states)

    prior = stack_states(priors, config, dim=0)
    post = stack_states(posteriors, config, dim=0)
    return prior.map(lambda x: x[:-1]), post.map(lambda x: x[:-1]), post.deter[1:] if config.A_USE_RNN else post.stoch[1:] #necessary for reward prediction etc. 
