import torch
import torch.nn as nn
from torch.distributions import OneHotCategorical
from copy import deepcopy
from configs.dreamer.DreamerAgentConfig import RSSMState
from networks.transformer.layers import AttentionEncoder
from configs.dreamer.DreamerAgentConfig import A_RSSMStateDiscrete, A_RSSMStateCont
from networks.dreamer.action_model_rnns import stack_states as a_stack_states
def stack_states(rssm_states: list, dim):
    return reduce_states(rssm_states, dim, torch.stack)


def cat_states(rssm_states: list, dim):
    return reduce_states(rssm_states, dim, torch.cat)


def reduce_states(rssm_states: list, dim, func):
    return RSSMState(*[func([getattr(state, key) for state in rssm_states], dim=dim)
                       for key in rssm_states[0].__dict__.keys()])


class DiscreteLatentDist(nn.Module):
    def __init__(self, in_dim, n_categoricals, n_classes, hidden_size):
        super().__init__()
        self.n_categoricals = n_categoricals
        self.n_classes = n_classes
        self.dists = nn.Sequential(nn.Linear(in_dim, hidden_size),
                                   nn.ReLU(),
                                   nn.Linear(hidden_size, n_classes * n_categoricals))

    def forward(self, x):
        logits = self.dists(x).view(x.shape[:-1] + (self.n_categoricals, self.n_classes))
        class_dist = OneHotCategorical(logits=logits)
        one_hot = class_dist.sample()
        latents = one_hot + class_dist.probs - class_dist.probs.detach()
        return logits.view(x.shape[:-1] + (-1,)), latents.view(x.shape[:-1] + (-1,))


class RSSMTransition(nn.Module):
    def __init__(self, config, hidden_size=200, activation=nn.ReLU):
        super().__init__()
        self._stoch_size = config.STOCHASTIC
        self._deter_size = config.DETERMINISTIC
        self._hidden_size = hidden_size
        self._activation = activation
        self._cell = nn.GRU(hidden_size, self._deter_size)
        self._attention_stack = AttentionEncoder(3, hidden_size, hidden_size, dropout=0.1)
        self._rnn_input_model = self._build_rnn_input_model(config.ACTION_SIZE + self._stoch_size)
        self._stochastic_prior_model = DiscreteLatentDist(self._deter_size, config.N_CATEGORICALS, config.N_CLASSES,
                                                          self._hidden_size)

    def _build_rnn_input_model(self, in_dim):
        rnn_input_model = [nn.Linear(in_dim, self._hidden_size)]
        rnn_input_model += [self._activation()]
        return nn.Sequential(*rnn_input_model)

    def forward(self, prev_actions, prev_states, mask=None):
        batch_size = prev_actions.shape[0]
        n_agents = prev_actions.shape[1]
        stoch_input = self._rnn_input_model(torch.cat([prev_actions, prev_states.stoch], dim=-1))
        attn = self._attention_stack(stoch_input, mask=mask)
        deter_state = self._cell(attn.reshape(1, batch_size * n_agents, -1),
                                 prev_states.deter.reshape(1, batch_size * n_agents, -1))[0].reshape(batch_size, n_agents, -1)
        logits, stoch_state = self._stochastic_prior_model(deter_state)
        return RSSMState(logits=logits, stoch=stoch_state, deter=deter_state)


class RSSMRepresentation(nn.Module):
    def __init__(self, config, transition_model: RSSMTransition):
        super().__init__()
        self._transition_model = transition_model
        self._stoch_size = config.STOCHASTIC
        self._deter_size = config.DETERMINISTIC
        self._stochastic_posterior_model = DiscreteLatentDist(self._deter_size + config.EMBED, config.N_CATEGORICALS,
                                                              config.N_CLASSES, config.HIDDEN)

    def initial_state(self, batch_size, n_agents, **kwargs):
        return RSSMState(stoch=torch.zeros(batch_size, n_agents, self._stoch_size, **kwargs),
                         logits=torch.zeros(batch_size, n_agents, self._stoch_size, **kwargs),
                         deter=torch.zeros(batch_size, n_agents, self._deter_size, **kwargs))

    def forward(self, obs_embed, prev_actions, prev_states, mask=None):
        """
        :param obs_embed: size(batch, n_agents, obs_size)
        :param prev_actions: size(batch, n_agents, action_size)
        :param prev_states: size(batch, n_agents, state_size)
        :return: RSSMState, global_state: size(batch, 1, global_state_size)
        """
        prior_states = self._transition_model(prev_actions, prev_states, mask)
        x = torch.cat([prior_states.deter, obs_embed], dim=-1)
        logits, stoch_state = self._stochastic_posterior_model(x)
        posterior_states = RSSMState(logits=logits, stoch=stoch_state, deter=prior_states.deter)
        return prior_states, posterior_states


def rollout_representation(representation_model, steps, obs_embed, action, prev_states, done):
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
        prior_states, posterior_states = representation_model(obs_embed[t], action[t], prev_states)
        prev_states = posterior_states.map(lambda x: x * (1.0 - done[t]))
        priors.append(prior_states)
        posteriors.append(posterior_states)

    prior = stack_states(priors, dim=0)
    post = stack_states(posteriors, dim=0)
    return prior.map(lambda x: x[:-1]), post.map(lambda x: x[:-1]), post.deter[1:]



def select_latent_action(actor, prev_latent_state, observations):

    actor_state = actor(observations, prev_latent_state)
    return actor_state

def get_latent_action_states(ac_model, obs, prev_latent_state):
    
    action_model_state = ac_model.transition(obs, prev_latent_state)
    return action_model_state



def update_action_model_state(config, action_model, action_model_state, action):


    if(config.OPP):
      joint_action = action.reshape(-1, config.NUM_AGENTS*action.shape[-1]).unsqueeze(dim = 1).expand(-1, config.NUM_AGENTS, -1)
      action_embed = action_model.action_encoder(joint_action)     
    else:
      action_embed = action_model.action_encoder(action)
    x = torch.cat([action_model_state.deter, action_embed], dim=-1)
    if(config.A_DISCRETE_LATENTS):
      logits, stoch_state, log_probs = action_model.representation._stochastic_posterior_model(x)
      posterior_state = A_RSSMStateDiscrete(logits=logits, stoch=stoch_state, deter=action_model_state.deter, log_probs = log_probs)
    else:
      mean, std, stoch_state = action_model.representation._stochastic_posterior_model(x)
      posterior_state = A_RSSMStateCont(mean = mean, std = std, stoch=stoch_state, deter=action_model_state.deter)  
    
    return posterior_state

def rollout_policy(config, action_model, model, transition_model, av_action, steps, policy, prev_state, prev_action_model_state, prev_policy_latent_state):
    """
        Roll out the model with a policy function.
        :param steps: number of steps to roll out
        :param policy: RSSMState -> action
        :param prev_state: RSSM state, size(batch_size, state_size)
        :return: next states size(time_steps, batch_size, state_size),
                 actions size(time_steps, batch_size, action_size)
        """
    state = prev_state
    next_states = []
    actions = []
    av_actions = []
    policies = []
    imag_observations = []
    policy_latent_states = []
    action_model_priors, action_model_posteriors, prev_policy_latent_states = [], [], []
    for t in range(steps):
        feat = state.get_features().detach()
        imag_obs, _ = model.observation_decoder(feat)
        action_model_state = get_latent_action_states(action_model, imag_obs, prev_action_model_state)
 
        if(config.USE_LATENT_ACTIONS):
          #method 1
          policy_latent_state = select_latent_action(policy, prev_policy_latent_state, imag_obs)

          #method 2

          # if(config.A_DISCRETE_LATENTS):
          #   logits, stoch_state = policy(action_model_state.deter)
          #   policy_latent_state = A_RSSMStateDiscrete(logits=logits, stoch=stoch_state, deter=action_model_state.deter)

          # else:
          #   mean, std, latent_action = policy(action_model_state.deter) 
          #   policy_latent_state = A_RSSMStateCont(mean = mean, std = std, stoch=stoch_state, deter=action_model_state.deter)  

          latent_action  = policy_latent_state.stoch   

        
          action, pi = action_model.decode(latent_action, policy_latent_state)
        else:
          action, pi = policy(imag_obs)

        if av_action is not None:
            avail_actions = av_action(feat).sample()
            pi[avail_actions == 0] = -1e10
            action_dist = OneHotCategorical(logits=pi)
            action = action_dist.sample().squeeze(0)
            av_actions.append(avail_actions.squeeze(0))
        next_states.append(state)
        action_model_priors.append(action_model_state)
        prev_policy_latent_states.append(prev_policy_latent_state)
        policy_latent_states.append(policy_latent_state)
        policies.append(pi)
        imag_observations.append(imag_obs)
        actions.append(action)
        state = transition_model(action, state)



        if(config.USE_LATENT_ACTIONS):
          if(config.A_DISCRETE_LATENTS):
            prev_policy_latent_state = A_RSSMStateDiscrete(logits=policy_latent_state.logits.clone(), stoch=policy_latent_state.stoch.clone(),\
                                              deter=policy_latent_state.deter.clone(), log_probs=policy_latent_state.log_probs.clone()) 
          else:
            prev_policy_latent_state = A_RSSMStateCont(mean=policy_latent_state.mean.clone(), std=policy_latent_state.std.clone(), stoch=policy_latent_state.stoch.clone(),\
                                             deter=policy_latent_state.deter.clone())             

        posterior_state = update_action_model_state(config, action_model, action_model_state, action)
        action_model_posteriors.append(posterior_state)
        prev_action_model_state = deepcopy(posterior_state)
        
    return {"imag_states": stack_states(next_states, dim=0),
            "imag_obs":torch.stack(imag_observations, dim=0),
            "am_priors": a_stack_states(action_model_priors, config, dim=0),
            "am_posteriors": a_stack_states(action_model_posteriors, config, dim=0),
            "policy_latents": a_stack_states(policy_latent_states, config, dim=0),
            "prev_policy_latents": a_stack_states(prev_policy_latent_states, config, dim=0),
            "actions": torch.stack(actions, dim=0),
            "av_actions": torch.stack(av_actions, dim=0) if len(av_actions) > 0 else None,
            "old_policy": torch.stack(policies, dim=0)}

