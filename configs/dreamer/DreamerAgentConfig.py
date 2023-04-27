from dataclasses import dataclass

import torch
import torch.distributions as td
import torch.nn.functional as F

from configs.Config import Config

RSSM_STATE_MODE = 'discrete'


class DreamerConfig(Config):
    def __init__(self):
        super().__init__()

        
        self.TRAINING_START = 1000
        self.A_DISCRETE_LATENTS = False
        self.A_N_LATENTS = 2
        self.DISCRETE_LATENTS = True
        self.N_LATENTS = 256
        self.USE_LATENT_ACTIONS = True

        self.HIDDEN = 256
        self.MODEL_HIDDEN = 256 
        self.EMBED = 256
        self.N_CATEGORICALS = 32
        self.N_CLASSES = 32
        if(self.DISCRETE_LATENTS):
          self.STOCHASTIC = self.N_CATEGORICALS * self.N_CLASSES
        else:
          self.STOCHASTIC = self.N_LATENTS
        self.DETERMINISTIC = 256
        self.FEAT = self.STOCHASTIC + self.DETERMINISTIC
        self.GLOBAL_FEAT = self.FEAT + self.EMBED
        self.VALUE_LAYERS = 2
        self.VALUE_HIDDEN = 256
        self.PCONT_LAYERS = 2
        self.PCONT_HIDDEN = 256
        self.ACTION_SIZE = 9
        self.ACTION_LAYERS = 2
        self.ACTION_HIDDEN = 256
        self.REWARD_LAYERS = 2
        self.REWARD_HIDDEN = 256
        self.GAMMA = 0.99
        self.DISCOUNT = 0.99
        self.DISCOUNT_LAMBDA = 0.95
        self.IN_DIM = 30
        self.LOG_FOLDER = 'wandb/'
        self.NUM_AGENTS = 2
        self.ALPHA_WEIGHT = 0.01 #should be 0.05


        self.OBS_EMBED = 256 
        self.ACTION_EMBED = 256
        self.OPP = False
        self.A_USE_RNN = True

        self.A_HIDDEN = 256
        self.A_MODEL_HIDDEN = 256
        self.A_EMBED = 256
        self.A_N_CATEGORICALS = 16
        self.A_N_CLASSES = 2
        if(self.A_DISCRETE_LATENTS):
          self.A_STOCHASTIC = self.A_N_CATEGORICALS * self.A_N_CLASSES
        else:
          self.A_STOCHASTIC = self.A_N_LATENTS  
        self.A_DETERMINISTIC = 256
        self.A_FEAT = self.A_STOCHASTIC + self.A_DETERMINISTIC 
        self.GLOBAL_FEAT = self.A_FEAT + self.A_EMBED

        self.A_PCONT_LAYERS = 2
        self.A_PCONT_HIDDEN = 256

        self.A_REWARD_LAYERS = 2
        self.A_REWARD_HIDDEN = 256


@dataclass
class RSSMStateBase:
    stoch: torch.Tensor
    deter: torch.Tensor

    def map(self, func):
        return RSSMState(**{key: func(val) for key, val in self.__dict__.items()})

    def get_features(self):
        return torch.cat((self.stoch, self.deter), dim=-1)

    def get_dist(self, *input):
        pass


@dataclass
class RSSMStateDiscrete(RSSMStateBase):
    logits: torch.Tensor

    def get_dist(self, batch_shape, n_categoricals, n_classes):
        return F.softmax(self.logits.reshape(*batch_shape, n_categoricals, n_classes), -1)


@dataclass
class RSSMStateCont(RSSMStateBase):
    mean: torch.Tensor
    std: torch.Tensor

    def get_dist(self, *input):
        return td.independent.Independent(td.Normal(self.mean, self.std), 1)

# @dataclass
# class S_RSSMStateDiscrete:
#     stoch: torch.Tensor
#     deter: torch.Tensor
#     logits: torch.Tensor

#     def map(self, func):
#         return S_RSSMStateDiscrete(**{key: func(val) for key, val in self.__dict__.items()})

#     def get_features(self):
#         return torch.cat((self.stoch, self.deter), dim=-1)

#     def get_dist(self, batch_shape, n_categoricals, n_classes):
#         return F.softmax(self.logits.reshape(*batch_shape, n_categoricals, n_classes), -1)

# @dataclass
# class S_RSSMStateCont:
#     stoch: torch.Tensor
#     deter: torch.Tensor
#     mean: torch.Tensor
#     std: torch.Tensor

#     def map(self, func):
#         return S_RSSMStateCont(**{key: func(val) for key, val in self.__dict__.items()})

#     def get_features(self):
#         return torch.cat((self.stoch, self.deter), dim=-1)

#     def get_dist(self, *input):
#         return td.independent.Independent(td.Normal(self.mean, self.std), 1)

@dataclass
class A_RSSMStateDiscrete:
    stoch: torch.Tensor
    deter: torch.Tensor
    logits: torch.Tensor
    log_probs: torch.Tensor

    def map(self, func):
        return A_RSSMStateDiscrete(**{key: func(val) for key, val in self.__dict__.items()})

    def get_features(self):
        return torch.cat((self.stoch, self.deter), dim=-1)

    def get_dist(self, batch_shape, n_categoricals, n_classes):
        return F.softmax(self.logits.reshape(*batch_shape, n_categoricals, n_classes), -1)


@dataclass
class A_RSSMStateCont:
    stoch: torch.Tensor
    deter: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor

    def map(self, func):
        return A_RSSMStateCont(**{key: func(val) for key, val in self.__dict__.items()})

    def get_features(self):
        return torch.cat((self.stoch, self.deter), dim=-1)

    def get_dist(self, *input):
        return td.independent.Independent(td.Normal(self.mean, self.std), 1)


RSSMState = {'discrete': RSSMStateDiscrete,
             'cont': RSSMStateCont}[RSSM_STATE_MODE]
