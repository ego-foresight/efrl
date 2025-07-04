# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image

import utils
from models.actor import Actor
from models.critic import Critic
from models.mlp import mlp
from models.encoder import PoolEncoder
from models.random_shifts_aug import RandomShiftsAug
from models.dcgan_unet_64 import decoder_no_skips

        
class DrQV2EfAgent:

    def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
                 hidden_dim, critic_target_tau, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip, use_tb,
                 context_steps, hs_dim, ha_dim, num_babbling_steps,
                 rec_loss_weight, batch_size, **kwargs):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.context_steps = context_steps
        self.hs_dim = hs_dim
        self.ha_dim = ha_dim
        self.num_babbling_steps = num_babbling_steps
        obs_shape = [obs_shape[0]*obs_shape[1], obs_shape[2], obs_shape[3]]
        self.batch_size = batch_size
        self.beta = rec_loss_weight
        self.action_dim = action_shape[0]
        
        # models
        self.encoder       = PoolEncoder(obs_shape, repr_dim=hs_dim).to(device)
        self.actor         = Actor(self.encoder.repr_dim, action_shape, feature_dim, hidden_dim).to(device)
        self.critic        = Critic(self.encoder.repr_dim, action_shape, feature_dim, hidden_dim).to(device)
        self.critic_target = Critic(self.encoder.repr_dim, action_shape, feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # data augmentation
        self.aug = RandomShiftsAug(pad=4)

        # For Ego-Foresight
        self.mlp     = mlp(self.ha_dim+self.action_dim, [512, 512], self.ha_dim).to(device)
        self.decoder = decoder_no_skips(hs_dim).to(device)
        self.reconstruction_loss_fn = nn.MSELoss(reduction="none")

        print("Encoder parameters: ", sum(p.numel() for p in self.encoder.parameters() if p.requires_grad))
        print("Decoder parameters: ", sum(p.numel() for p in self.decoder.parameters() if p.requires_grad))
        print("MLP parameters: ",     sum(p.numel() for p in self.mlp.parameters()     if p.requires_grad))
        print("Critic parameters: ",  sum(p.numel() for p in self.critic.parameters()  if p.requires_grad))
        print("Actor parameters: ",   sum(p.numel() for p in self.actor.parameters()   if p.requires_grad))
            
        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt   = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt  = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.mlp_opt      = torch.optim.Adam(self.mlp.parameters(), lr=lr)
        self.decoder_opt = torch.optim.Adam(self.decoder.parameters(),  lr=lr)
        
        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)
        self.mlp.train(training)
        self.decoder.train(training)
        
    def act(self, obs, step, eval_mode):
        obs = obs["pixels"]
        obs = torch.as_tensor(obs, device=self.device)
        obs = obs.view((3*self.context_steps, 84, 84)).squeeze(1)
        obs = self.encoder(obs.unsqueeze(0))
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        if step < self.num_babbling_steps:
            action = dist.sample(clip=None)
            action.uniform_(-4.0, 4.0)  
            action = torch.clamp(action, -1.0, 1.0) 
        else:
            if eval_mode:
                action = dist.mean
            else:
                action = dist.sample(clip=None)
                if step < self.num_expl_steps:
                    action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_critic(self, encoded_obs, action, reward, discount,
                      encoded_next_obs, step, h_ef, h_scene, h_agent, rgb_target):
        metrics = {}

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(encoded_next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_q1, target_q2 = self.critic_target(encoded_next_obs, next_action)
            target_v = torch.min(target_q1, target_q2)
            target_q = reward + (discount * target_v)

        q1, q2 = self.critic(encoded_obs, action)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        
        # For Ego-Foresight
        reconstructed_obs = self.decoder(h_ef)
        rgb_target = rgb_target / 255.0
        reconstructed_obs = torch.clamp(reconstructed_obs, 0.0, 1.0)
        
        reconstruction_loss = self.reconstruction_loss_fn(reconstructed_obs, rgb_target)
        reconstruction_loss = reconstruction_loss.reshape(rgb_target.shape[0], -1).sum(dim=1).mean()
        
        loss = critic_loss + self.beta * reconstruction_loss
        
        self.encoder_opt.zero_grad(set_to_none=True)
        self.mlp_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        self.decoder_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.decoder_opt.step()
        self.critic_opt.step()
        self.mlp_opt.step()
        self.encoder_opt.step()

        if self.use_tb:
            metrics["critic_target_q"] = target_q.mean().item()
            metrics["critic_q1"] = q1.mean().item()
            metrics["critic_q2"] = q2.mean().item()
            metrics["critic_loss"] = critic_loss.item()
            metrics["reconstruction_loss"] = reconstruction_loss.item()

        return metrics

    def update_actor(self, encoded_obs, step):
        metrics = {}

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(encoded_obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        q1, q2 = self.critic(encoded_obs, action)
        q = torch.min(q1, q2)

        actor_loss = -q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics["actor_loss"] = actor_loss.item()
            metrics["actor_logprob"] = log_prob.mean().item()
            metrics["actor_ent"] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def update(self, replay_iter, step):
        metrics = {}

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        extended_obs, action, reward, discount, next_obs = batch
        
        actions = extended_obs["action"]
        
        n_cntxt = self.context_steps
        t = actions.size()[1]
        j = np.random.randint(n_cntxt, t)
        
        # select inputs
        actions = actions[:, n_cntxt:j+1]
                
        action, reward, discount, next_rgb_obs, \
        extended_rgb, actions = utils.to_torch((action, reward, discount, next_obs["pixels"],
             extended_obs["pixels"], actions), self.device)
        
        context_frames = extended_rgb[:, :n_cntxt]
        rgb_target = extended_rgb[:, j:j+1]
    
        # augment
        context_frames = self.aug(context_frames.float())
        rgb_target     = self.aug(rgb_target.float(), repeat_last=True)
        next_rgb_obs   = self.aug(next_rgb_obs.float())
        
        context_frames = context_frames.view((self.batch_size, 3*n_cntxt, 84, 84))
        next_rgb_obs   = next_rgb_obs.view((self.batch_size, 3*n_cntxt, 84, 84))
             
        # encode
        encoded_obs = self.encoder(context_frames)
        
        h_agent = encoded_obs[:, :self.ha_dim]
        h_scene = encoded_obs[:, self.ha_dim:]
           
        for i in range(actions.size(1)):  
            h_agent  = self.mlp(torch.cat([h_agent, actions[:, i]], -1))
                
        h_ef = torch.cat([h_agent, h_scene], dim=-1).unsqueeze(1)
        
        with torch.no_grad():
            encoded_next_obs = self.encoder(next_rgb_obs)

        if self.use_tb:
            metrics["batch_reward"] = reward.mean().item()

        # update critic and decoder
        metrics.update(
            self.update_critic(encoded_obs, action, reward, discount,
                               encoded_next_obs, step, h_ef, h_scene, h_agent, rgb_target))

        # update actor
        metrics.update(self.update_actor(encoded_obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics

    def get_frames_to_record(self, obs, first, actions):
        
        if first:
            self.i = 0
            self._actions = []
            self._rgb_obs = []
        else:
            self.i += 1

        rec_loss = None

        rgb_obs = obs["pixels"]
        rgb_obs_tensor = torch.as_tensor(rgb_obs, device=self.device, dtype=torch.float).view((3*self.context_steps, 84, 84))
                    
        with torch.no_grad():    
            _step_rec = 250
            _rec_len = _step_rec - self.context_steps
           
            self._actions.append(torch.tensor(actions, device=self.device).unsqueeze(0).clone().detach())
            self._rgb_obs.append(obs['pixels'][-1:].copy())
            
            if self.i == _step_rec-1:
                
                actions_tensor = torch.stack(self._actions, dim=1)[:, self.context_steps:]
                _rgb_obs_tensor = torch.tensor(np.concatenate(self._rgb_obs, axis=0)).cuda()
                rgb_obs_tensor = _rgb_obs_tensor[:self.context_steps]
                rgb_obs_tensor = rgb_obs_tensor.view((1, 3*self.context_steps, 84, 84))
                hs = self.encoder(rgb_obs_tensor)
                
                h_agent = hs[:, :self.ha_dim]
                h_scene = torch.unsqueeze(hs[:, self.ha_dim:], dim=1).repeat((1, _rec_len, 1))
                
                h_agent_list = []
                for i in range(_rec_len):
                    h_agent  = self.mlp(torch.cat([h_agent, actions_tensor[:, i]], -1))
                    h_agent_list.append(h_agent)
                h_agent = torch.stack(h_agent_list, dim=1)
                
                h_pred = torch.cat([h_agent, h_scene], -1) 
                
                reconstructed_pred = self.decoder(h_pred)
                reconstructed_pred = torch.clamp(reconstructed_pred, 0.0, 1.0)
                target = (_rgb_obs_tensor[self.context_steps:]/255.0).unsqueeze(0)

                rec_loss = self.reconstruction_loss_fn(reconstructed_pred, target).reshape(1, _rec_len, -1).sum(dim=2).mean().detach().cpu().numpy()
                reconstructed_pred = (reconstructed_pred[0].cpu().numpy() * 255.0)       
                      
        if self.i == _step_rec-1:
            frame_names = ["rgb", "reconstructed_pred"]
            obs_list = [rgb_obs, reconstructed_pred]
        else:
            frame_names = ["rgb"]
            obs_list = [rgb_obs]
        frames = {}
        for k, v in zip(frame_names, obs_list):
            if k == "reconstructed_pred":
                frames[k] = v.transpose(0, 2, 3, 1).clip(0, 255).astype(np.uint8)
            else:
                if len(v.shape) == 4:
                    frames[k] = v[-1].transpose(1, 2, 0).clip(0, 255).astype(np.uint8)
                elif len(v.shape) == 3:
                    frames[k] = v[-3:].transpose(1, 2, 0).clip(0, 255).astype(np.uint8)

        return frames, rec_loss

    def load_pretrained_weights(self, pretrain_path, just_encoder_decoders):
        if just_encoder_decoders:
            print("Loading pretrained encoder and decoders")
        else:
            print("Loading entire agent")

        payload = torch.load(pretrain_path, map_location="cpu")
        pretrained_agent = payload['agent']

        self.encoder.load_state_dict(pretrained_agent.encoder.state_dict())

        if not just_encoder_decoders:
            self.mlp.load_state_dict(pretrained_agent.mlp.state_dict())
            self.decoder.load_state_dict(pretrained_agent.decoder.state_dict())
            self.actor.load_state_dict(pretrained_agent.actor.state_dict())
            self.critic.load_state_dict(pretrained_agent.critic.state_dict())
            self.critic_target.load_state_dict(self.critic.state_dict())
