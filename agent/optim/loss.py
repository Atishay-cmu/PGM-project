import numpy as np
import torch
import wandb
import torch.nn.functional as F
from torch.distributions import OneHotCategorical
from configs.dreamer.DreamerAgentConfig import A_RSSMStateDiscrete, A_RSSMStateCont
from agent.optim.utils import rec_loss, compute_return, state_divergence_loss, calculate_ppo_loss1, calculate_ppo_loss2, calculate_kl_gaussian,\
    batch_multi_agent, log_prob_loss, info_loss, new_kl_loss
from agent.utils.params import FreezeParameters
from networks.dreamer.rnns import rollout_representation, rollout_policy


def model_loss(config, model, obs, action, av_action, reward, done, fake, last):
    time_steps = obs.shape[0]
    batch_size = obs.shape[1]
    n_agents = obs.shape[2]

    embed = model.observation_encoder(obs.reshape(-1, n_agents, obs.shape[-1]))
    embed = embed.reshape(time_steps, batch_size, n_agents, -1)

    prev_state = model.representation.initial_state(batch_size, n_agents, device=obs.device)
    prior, post, deters = rollout_representation(model.representation, time_steps, embed, action, prev_state, last)
    feat = torch.cat([post.stoch, deters], -1)
    feat_dec = post.get_features()

    reconstruction_loss, i_feat = rec_loss(model.observation_decoder,
                                           feat_dec.reshape(-1, n_agents, feat_dec.shape[-1]),
                                           obs[:-1].reshape(-1, n_agents, obs.shape[-1]),
                                           1. - fake[:-1].reshape(-1, n_agents, 1))
    reward_loss = F.smooth_l1_loss(model.reward_model(feat), reward[1:])
    pcont_loss = log_prob_loss(model.pcont, feat, (1. - done[1:]))
    av_action_loss = log_prob_loss(model.av_action, feat_dec, av_action[:-1]) if av_action is not None else 0.
    i_feat = i_feat.reshape(time_steps - 1, batch_size, n_agents, -1)

    dis_loss = info_loss(i_feat[1:], model, action[1:-1], 1. - fake[1:-1].reshape(-1))
    #if(config.DISCRETE_LATENTS):
    div = state_divergence_loss(prior, post, config)
    # else:  
    #   div = calculate_kl_gaussian(post, prior, config)

    model_loss = div + reward_loss + dis_loss + reconstruction_loss + pcont_loss + av_action_loss
    if np.random.randint(20) == 4:
        wandb.log({'Model/reward_loss': reward_loss, 'Model/div': div, 'Model/av_action_loss': av_action_loss,
                   'Model/reconstruction_loss': reconstruction_loss, 'Model/info_loss': dis_loss,
                   'Model/pcont_loss': pcont_loss})

    return model_loss

def detach(y):
  return y.map(lambda x:x.detach())

def truncate_reshape(y, n_agents):
  return y.map(lambda x: x[:-1].reshape(-1, n_agents, x.shape[-1]))
def actor_rollout(obs, action, last, model, action_model, actor, critic, config):
    n_agents = obs.shape[2]
    with FreezeParameters([model]):
        embed = model.observation_encoder(obs.reshape(-1, n_agents, obs.shape[-1]))
        embed = embed.reshape(obs.shape[0], obs.shape[1], n_agents, -1)
        prev_state = model.representation.initial_state(obs.shape[1], obs.shape[2], device=obs.device)
        prior, post, _ = rollout_representation( model.representation, obs.shape[0], embed, action,
                                                prev_state, last)
        post = post.map(lambda x: x.reshape((obs.shape[0] - 1) * obs.shape[1], n_agents, -1))
        ########################################################################################################
        if(config.USE_LATENT_ACTIONS):
          prev_policy_latent_state = actor.initial_state(post.stoch.shape[0], n_agents, device=obs.device) #method 1
          #prev_policy_latent_state = action_model.transition.initial_state(post.stoch.shape[0], n_agents, device=obs.device) #method 2
        else:
          prev_policy_latent_state = action_model.transition.initial_state(post.stoch.shape[0], n_agents, device=obs.device)
        prev_action_model_state = action_model.transition.initial_state(post.stoch.shape[0], n_agents, device=obs.device) #method 2
        ########################################################################################################

        items = rollout_policy(config, action_model, model, model.transition, model.av_action, config.HORIZON, actor, post, prev_action_model_state, prev_policy_latent_state)
    imag_feat = items["imag_states"].get_features()
    imag_obs = items["imag_obs"]#,_ = model.observation_decoder(imag_feat)
    imag_rew_feat = torch.cat([items["imag_states"].stoch[:-1], items["imag_states"].deter[1:]], -1)
    returns = critic_rollout(model, critic, imag_feat, imag_rew_feat, items["actions"],
                             items["imag_states"].map(lambda x: x.reshape(-1, n_agents, x.shape[-1])), config)
    output = [items["actions"][:-1].detach(),
              items["av_actions"][:-1].detach() if items["av_actions"] is not None else None,
              items["old_policy"][:-1].detach(), imag_feat[:-1].detach(), imag_obs[:-1].detach(), returns.detach()]
    return [batch_multi_agent(v, n_agents) for v in output], detach(truncate_reshape(items["am_priors"], n_agents)),\
           detach(truncate_reshape(items["am_posteriors"], n_agents)),\
           detach(truncate_reshape(items["prev_policy_latents"], n_agents)), detach(truncate_reshape(items["policy_latents"], n_agents)) 


def critic_rollout(model, critic, states, rew_states, actions, raw_states, config):
    with FreezeParameters([model, critic]):
        imag_reward = calculate_next_reward(model, actions, raw_states)
        imag_reward = imag_reward.reshape(actions.shape[:-1]).unsqueeze(-1).mean(-2, keepdim=True)[:-1]
        value = critic(states, actions)
        discount_arr = model.pcont(rew_states).mean
        wandb.log({'Value/Max reward': imag_reward.max(), 'Value/Min reward': imag_reward.min(),
                   'Value/Reward': imag_reward.mean(), 'Value/Discount': discount_arr.mean(),
                   'Value/Value': value.mean()})
    returns = compute_return(imag_reward, value[:-1], discount_arr, bootstrap=value[-1], lmbda=config.DISCOUNT_LAMBDA,
                             gamma=config.GAMMA)
    return returns


def calculate_reward(model, states, mask=None):
    imag_reward = model.reward_model(states)
    if mask is not None:
        imag_reward *= mask
    return imag_reward


def calculate_next_reward(model, actions, states):
    actions = actions.reshape(-1, actions.shape[-2], actions.shape[-1])
    next_state = model.transition(actions, states)
    imag_rew_feat = torch.cat([states.stoch, next_state.deter], -1)
    return calculate_reward(model, imag_rew_feat)


def select_latent_action(actor, prev_latent_state, observations):

    actor_state = actor(observations, prev_latent_state)
    return actor_state


def actor_loss(config, am_priors, am_posts, prev_policy_latents, old_policy_latents, imag_states, imag_obs, actions, av_actions, old_policy, advantage, actor, action_model, ent_weight):


    #_, new_policy = actor(imag_states)
    if(config.USE_LATENT_ACTIONS):
      #method 1
      policy_latent_state = select_latent_action(actor, prev_policy_latents, imag_obs)
      

      # #method 2
      # if(config.A_DISCRETE_LATENTS):
      #   logits, stoch_state = actor(am_priors.deter)
      #   policy_latent_state = A_RSSMStateDiscrete(logits=logits, stoch=stoch_state, deter=am_priors.deter)

      # else:
      #   mean, std, latent_action = actor(am_priors.deter) 
      #   policy_latent_state = A_RSSMStateCont(mean = mean, std = std, stoch=stoch_state, deter=am_priors.deter)     

      latent_action  = policy_latent_state.stoch
      _, new_policy = action_model.decode(latent_action, policy_latent_state) 
    else:
      _, new_policy = actor(imag_obs)
    if av_actions is not None:
        new_policy[av_actions == 0] = -1e10
    actions = actions.argmax(-1, keepdim=True)

    ##############################################################################
    # old_log_probs = old_policy_latents.log_probs
    # new_log_probs = policy_latent_state.log_probs
    # rho_latent = torch.exp(new_log_probs - old_log_probs).mean(dim = -1, keepdim = True)
    rho = (F.log_softmax(new_policy, dim=-1).gather(2, actions) -
           F.log_softmax(old_policy, dim=-1).gather(2, actions)).exp()
    
    rho_all = rho# + #0.0*rho_latent
    if(config.USE_LATENT_ACTIONS):           
      ppo_loss, ent_loss, kl_loss = calculate_ppo_loss2(config, new_policy, rho_all, advantage, am_priors, am_posts, policy_latent_state)
      if np.random.randint(10) == 9:
          wandb.log({'Policy/Entropy': ent_loss.mean(), 'Policy/Imp_weight': rho.mean(), 'Policy/Kl_new': kl_loss.mean(),\
                                                     'Policy/Latent_Imp_weight': 0.0 , 'Policy/Mean action': actions.float().mean()})
      return (ppo_loss + ent_loss.unsqueeze(-1) * ent_weight).mean() + config.ALPHA_WEIGHT*kl_loss
    else:
      ppo_loss, ent_loss = calculate_ppo_loss1(config, new_policy, rho, advantage)     
      if np.random.randint(10) == 9:
          wandb.log({'Policy/Entropy': ent_loss.mean(), 'Policy/Imp_weight': rho.mean(), 'Policy/Mean action': actions.float().mean()})
      return (ppo_loss + ent_loss.unsqueeze(-1) * ent_weight).mean() 



def value_loss(config, am_priors, am_posts, prev_policy_latents, imag_obs, critic, actions, imag_feat, targets):


    #_, new_policy = actor(imag_states)

    # if(config.USE_LATENT_ACTIONS):
    #   with torch.no_grad():
    #     policy_latent_state = select_latent_action(actor, prev_policy_latents, imag_obs)
    #     kl_loss = new_kl_loss(config, am_priors, policy_latent_state)
    #     value_pred = critic(imag_feat, actions)
    #     mse_loss = (targets - value_pred) ** 2 / 2.0
    #     return torch.mean(mse_loss) +  kl_loss
    # else:
    value_pred = critic(imag_feat, actions)
    mse_loss = (targets - value_pred) ** 2 / 2.0
    return torch.mean(mse_loss)     
