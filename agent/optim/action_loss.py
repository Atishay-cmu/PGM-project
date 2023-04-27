import numpy as np
import torch
import wandb
import torch.nn.functional as F

from agent.optim.utils import rec_loss, compute_return, state_divergence_loss, calculate_kl_gaussian,\
    batch_multi_agent, log_prob_loss, info_loss1, info_loss2, action_rec_loss
from agent.utils.params import FreezeParameters
from networks.dreamer.action_model_rnns import rollout_representation


def action_model_loss(config, model, obs, action, av_action, reward, done, fake, last):
    time_steps = obs.shape[0]
    batch_size = obs.shape[1]
    n_agents = obs.shape[2]

    
    # if(config.OPP):
    #   obs_new = torch.zeros((time_steps, batch_size, n_agents, config.IN_DIM + config.ACTION_SIZE*(config.NUM_AGENTS-1))).to(config.DEVICE)
    #   for i in range(n_agents):
    #     if(i == 0):
    #       other_actions = action[:, :, 1:].clone().reshape(time_steps, batch_size, -1)
    #     elif(i == (n_agents -1)):
    #       other_actions = action[:, :, :-1].clone().reshape(time_steps, batch_size, -1)
    #     else:  
    #       other_actions = torch.cat([action[:, :, :i], action[:, :, i+1:]]).clone().reshape(time_steps, batch_size, -1)
    #     obs_new[:, :, i] = torch.cat([obs[:, :, i], other_actions], dim = -1)
    #   obs = obs_new.clone()

    if(config.OPP):
      joint_action = action.reshape(-1, batch_size, n_agents*action.shape[-1]).unsqueeze(dim = 2).expand(-1, -1, n_agents, -1)
      action_embed = model.action_encoder(joint_action.reshape(-1, n_agents, joint_action.shape[-1])) 
      action_embed = action_embed.reshape(time_steps, batch_size, n_agents, -1)
    else:
      action_embed = model.action_encoder(action.reshape(-1, n_agents, action.shape[-1])) 
      action_embed = action_embed.reshape(time_steps, batch_size, n_agents, -1)




    prev_state = model.representation.initial_state(batch_size, n_agents, device=obs.device)
    prior, post, deters = rollout_representation(config, model.representation, time_steps, action_embed, obs, prev_state, last) #TODO
    feat = torch.cat([post.stoch, deters], -1) 
    feat_dec = post.get_features() if config.A_USE_RNN else post.stoch

    #invalid_loss,
    reconstruction_loss, i_feat = action_rec_loss(model,  #use argmax , dim = -1 to recover predicted actions from logits, or fit to OneHotCategorical and sample, alternatively
                                           feat_dec.reshape(-1, n_agents, feat_dec.shape[-1]),
                                           action[:-1].reshape(-1, n_agents, action.shape[-1]),
                                           1. - fake[:-1].detach().clone().reshape(-1, n_agents, 1), av_action[:-1].reshape(-1, n_agents, av_action.shape[-1]))
    reward_loss = F.smooth_l1_loss(model.reward_model(feat), reward[:-1])
    pcont_loss = 0.#log_prob_loss(model.pcont, feat, (1. - done[:-1]))
    av_action_loss = log_prob_loss(model.av_action, feat_dec, av_action[:-1]) if av_action is not None else 0.
    i_feat = i_feat.reshape(time_steps - 1, batch_size, n_agents, -1)

    action_diff = torch.abs(action[1:] - action[:-1])
    # #predicting current obs
    #dis_loss1 = info_loss1(i_feat.reshape(-1, n_agents, i_feat.shape[-1]), model, obs[:-1].reshape(-1, n_agents, obs.shape[-1]), 1. - fake[:-1].reshape(-1, n_agents, 1))
    #predicting diff
    dis_loss2 = info_loss2(i_feat[1:], model, action_diff[:-1], 1. - fake[1:-1].detach().clone())
    dis_loss =  dis_loss2 #+ dis_loss1 * 0.001

    if(config.A_DISCRETE_LATENTS):
      div = state_divergence_loss(prior, post, config, action = True)
    else:  
      div = calculate_kl_gaussian(post, prior, config) #?

    model_loss = div + reward_loss + reconstruction_loss + av_action_loss + dis_loss + pcont_loss #+ invalid_loss

    if np.random.randint(20) == 4:
        wandb.log({'ActionModel/reward_loss': reward_loss, 'ActionModel/div': div, 'ActionModel/av_action_loss': av_action_loss,
                   'ActionModel/reconstruction_loss': reconstruction_loss, 'ActionModel/info_loss': dis_loss, #'ActionModel/invalid_loss': invalid_loss,
                   'ActionModel/pcont_loss': pcont_loss})

    return model_loss