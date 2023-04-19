import torch
import torch.nn as nn

from torch.nn import functional as F
from torch.distributions import OneHotCategorical
from networks.dreamer.dense import DenseBinaryModel, DenseModel
from networks.dreamer.vae import Encoder, Decoder
from networks.dreamer.action_model_rnns import RSSMRepresentation, RSSMTransition


class ActionModel(nn.Module):
    def __init__(self, config): #action size, in dim, 
        super().__init__()

        self.action_size = config.ACTION_SIZE
        if(config.OPP):
          config.OBS_EMBED = 16


        self.action_encoder = Encoder(in_dim=config.ACTION_SIZE, hidden=config.HIDDEN, embed=config.ACTION_EMBED)
        action_dec_input_size = config.A_FEAT if config.A_USE_RNN else config.A_STOCHASTIC
        self.action_decoder1 = DenseModel(action_dec_input_size, config.A_PCONT_HIDDEN, 1, config.A_PCONT_HIDDEN) #Decoder(embed=config.A_FEAT, hidden=config.A_HIDDEN, out_dim=config.ACTION_SIZE)
        self.action_decoder2 = nn.Linear(config.A_PCONT_HIDDEN, config.ACTION_SIZE)

        self.transition = RSSMTransition(config, config.A_MODEL_HIDDEN)
        self.representation = RSSMRepresentation(config, self.transition)
        in_size = config.A_FEAT if config.A_USE_RNN else config.A_STOCHASTIC*2
        #self.reward_model = DenseModel(in_size, 1, config.A_REWARD_LAYERS, config.A_REWARD_HIDDEN)
        #self.pcont = DenseBinaryModel(in_size, 1, config.A_PCONT_LAYERS, config.A_PCONT_HIDDEN)
        self.av_action = DenseBinaryModel(action_dec_input_size, config.ACTION_SIZE, config.A_PCONT_LAYERS, config.A_PCONT_HIDDEN)



        #self.q_features_diff = DenseBinaryModel(config.A_PCONT_HIDDEN, config.ACTION_SIZE, config.A_PCONT_LAYERS, config.A_PCONT_HIDDEN)
 
    def decode(self, latent_action):
        action_feat = F.relu(self.action_decoder1(latent_action))
        action_logits = self.action_decoder2(action_feat)
        pi = OneHotCategorical(logits=action_logits)
        action = pi.sample()   
        return action, action_logits