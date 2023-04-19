import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def rec_loss(decoder, z, x, fake):
    x_pred, feat = decoder(z)
    batch_size = np.prod(list(x.shape[:-1]))
    gen_loss1 = (F.smooth_l1_loss(x_pred, x, reduction='none') * fake).sum() / batch_size
    return gen_loss1, feat



def action_rec_loss(model, z, actions, fake):
    action_feat = F.relu(model.action_decoder1(z))
    action_logits = model.action_decoder2(action_feat)
    return (fake * action_information_loss(action_logits, actions)).mean(), action_feat


def info_loss2(feat, model, action_diff, fake): #for predicting action diff
    pred = model.q_features_diff(feat)
    return -torch.mean(fake.squeeze(dim = -1) * pred.log_prob(action_diff))


def info_loss1(feat, model, obs, fake): #for predicting observations
    x_pred, _ = model.q_features_obs(feat)
    batch_size = np.prod(list(obs.shape[:-1]))
    gen_loss1 = (F.smooth_l1_loss(x_pred, obs, reduction='none') * fake).sum() / batch_size
    return gen_loss1


def ppo_loss(A, rho, eps=0.2):
    return -torch.min(rho * A, rho.clamp(1 - eps, 1 + eps) * A)


def mse(model, x, target):
    pred = model(x)
    return ((pred - target) ** 2 / 2).mean()


def entropy_loss(prob, logProb):
    return (prob * logProb).sum(-1)

def new_kl_loss(config, p, q):
  if(config.A_DISCRETE_LATENTS):
    return state_divergence_loss(p, q, config)
  else:
    return calculate_kl_gaussian(q, p, config)


def advantage(A):
    std = 1e-4 + A.std() if len(A) > 0 else 1
    adv = (A - A.mean()) / std
    adv = adv.detach()
    adv[adv != adv] = 0
    return adv


def calculate_ppo_loss2(config, logits, rho, A, am_prior, am_post, policy_latent):
    prob = F.softmax(logits, dim=-1)
    logProb = F.log_softmax(logits, dim=-1)
    polLoss = ppo_loss(A, rho)
    entLoss = entropy_loss(prob, logProb)
    klLoss = new_kl_loss(config, am_prior, policy_latent)
    return polLoss, entLoss, klLoss

def calculate_ppo_loss1(config, logits, rho, A):
    prob = F.softmax(logits, dim=-1)
    logProb = F.log_softmax(logits, dim=-1)
    polLoss = ppo_loss(A, rho)
    entLoss = entropy_loss(prob, logProb)
    return polLoss, entLoss

def batch_multi_agent(tensor, n_agents):
    return tensor.view(-1, n_agents, tensor.shape[-1]) if tensor is not None else None


def compute_return(reward, value, discount, bootstrap, lmbda, gamma):
    next_values = torch.cat([value[1:], bootstrap[None]], 0)
    target = reward + gamma * discount * next_values * (1 - lmbda)
    outputs = []
    accumulated_reward = bootstrap
    for t in reversed(range(reward.shape[0])):
        discount_factor = discount[t]
        accumulated_reward = target[t] + gamma * discount_factor * accumulated_reward * lmbda
        outputs.append(accumulated_reward)
    returns = torch.flip(torch.stack(outputs), [0])
    return returns


def info_loss(feat, model, actions, fake):
    q_feat = F.relu(model.q_features(feat))
    action_logits = model.q_action(q_feat)
    return (fake * action_information_loss(action_logits, actions)).mean()


def action_information_loss(logits, target):
    criterion = nn.CrossEntropyLoss(reduction='none')
    return criterion(logits.view(-1, logits.shape[-1]), target.argmax(-1).view(-1))


def log_prob_loss(model, x, target):
    pred = model(x)
    return -torch.mean(pred.log_prob(target))


def kl_div_categorical(p, q):
    eps = 1e-7
    return (p * (torch.log(p + eps) - torch.log(q + eps))).sum(-1)


def reshape_dist(dist, config):
    return dist.get_dist(dist.deter.shape[:-1], config.N_CATEGORICALS, config.N_CLASSES)

def calculate_kl_gaussian(q, p, config):

    var_ratio = (p.std / q.std).pow_(2)
    t1 = ((p.mean - q.mean) / q.std).pow_(2)
    kl =  0.5 * (var_ratio + t1 - 1 - var_ratio.log())
    return torch.mean(kl)


def state_divergence_loss(prior, posterior, config, reduce=True, balance=0.2):
    prior_dist = reshape_dist(prior, config)
    post_dist = reshape_dist(posterior, config)
    post = kl_div_categorical(post_dist, prior_dist.detach())
    pri = kl_div_categorical(post_dist.detach(), prior_dist)
    kl_div = balance * post.mean(-1) + (1 - balance) * pri.mean(-1)
    if reduce:
        return torch.mean(kl_div)
    else:
        return kl_div
