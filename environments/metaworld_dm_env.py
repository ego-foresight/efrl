import random

import dm_env
import metaworld
import numpy as np
from dm_control.suite.wrappers import action_scale, pixels
from dm_env import specs

from environments.metaworld_custom_envs import (MT3_Customized,
                                                MT5_Customized,
                                                MT10_Customized,
                                                MT_Door)
from environments.wrappers import (
    ActionDTypeWrapper, ActionRepeatWrapper, ExtendedTimeStepWrapper,
    FrameStackWrapper, NoisyMaskWrapper, Render_Wrapper, SegmentationFilter,
    SegmentationToRobotMaskWrapper, SlimMaskWrapper, StackRGBAndMaskWrapper,
    get_env_action_spec, get_env_observation_spec)


def get_proprioceptive_spec(spec, num_proprioceptive_features):
    return specs.BoundedArray(
        shape=(num_proprioceptive_features,),
        dtype=spec.dtype,
        minimum=spec.minimum[0:num_proprioceptive_features],
        maximum=spec.maximum[0:num_proprioceptive_features],
        name="observation")


class MT_Wrapper(dm_env.Environment):

    def __init__(self,
                 env_name: str,
                 discount=1.0,
                 seed=None,
                 proprioceptive_state=True):
        self.discount = discount
        if env_name in ["MT10", "mt10"]:
            self.mt = metaworld.MT10(seed=seed)
        elif env_name in ["MT50", "mt50"]:
            self.mt = metaworld.MT50(seed=seed)
        elif env_name in ["MT3-Customized", "mt3-customized"]:
            self.mt = MT3_Customized(seed=seed)
        elif env_name in ["MT5-Customized", "mt5-customized"]:
            self.mt = MT5_Customized(seed=seed)
        elif env_name in ["MT10-Customized", "mt10-customized"]:
            self.mt = MT10_Customized(seed=seed)
        elif env_name in ["MT-Door", "mt-door"]:
            self.mt = MT_Door(seed=seed)
        else:
            self.mt = metaworld.MT1(env_name, seed=seed)
        self.all_envs = {
            name: env_cls() for name, env_cls in self.mt.train_classes.items()
        }
        self.env_name, self._env = self.sample_env()
        self.physics = Render_Wrapper(self._env.sim.render)
        self._reset_next_step = True
        self.current_step = 0
        self.proprioceptive_state = proprioceptive_state
        self.NUM_PROPRIOCEPTIVE_STATES = 7
        self._observation_spec = None
        self._action_spec = None
        self.robot_segmentation_ids = list(range(8, 35))

    def sample_env(self):
        return random.choice(list(self.all_envs.items()))

    def sample_task(self):
        return random.choice([
            task for task in self.mt.train_tasks
            if task.env_name == self.env_name
        ])

    def reset(self) -> dm_env.TimeStep:
        self._reset_next_step = False
        self.current_step = 0
        self.env_name, self._env = self.sample_env()
        self.physics = Render_Wrapper(self._env.sim.render)
        task = self.sample_task()
        self._env.set_task(task)
        observation = self._env.reset()
        if self.proprioceptive_state:
            observation = self.get_proprioceptive_observation(observation)
        observation = observation.astype(self._env.observation_space.dtype)
        self.current_step += 1
        return dm_env.restart(observation)

    def step(self, action: int) -> dm_env.TimeStep:
        if self._reset_next_step:
            return self.reset()

        observation, reward, _, info = self._env.step(action)
        self.current_step += 1

        if self.proprioceptive_state:
            observation = self.get_proprioceptive_observation(observation)

        observation = observation.astype(self._env.observation_space.dtype)

        reward_dict = {"reward": reward, "success": info["success"]}

        if self.current_step == self._env.max_path_length:
            self._reset_next_step = True
            return dm_env.truncation(reward_dict, observation, self.discount)

        return dm_env.transition(reward_dict, observation, self.discount)

    def get_proprioceptive_observation(self, observation):
        observation = observation[0:self.NUM_PROPRIOCEPTIVE_STATES]
        return observation

    def observation_spec(self) -> specs.BoundedArray:
        if self._observation_spec is not None:
            return self._observation_spec

        spec = None
        for _, env in self.all_envs.items():
            if spec is None:
                spec = get_env_observation_spec(env)
                continue
            assert spec == get_env_observation_spec(
                env), "The observation spec should match for all environments"

        if self.proprioceptive_state:
            spec = get_proprioceptive_spec(spec, self.NUM_PROPRIOCEPTIVE_STATES)

        self._observation_spec = spec
        return self._observation_spec

    def action_spec(self) -> specs.BoundedArray:
        if self._action_spec is not None:
            return self._action_spec

        spec = None
        for _, env in self.all_envs.items():
            if spec is None:
                spec = get_env_action_spec(env)
                continue
            assert spec == get_env_action_spec(
                env), "The action spec should match for all environments"

        self._action_spec = spec
        return self._action_spec

    def __getattr__(self, name):
        return getattr(self._env, name)


def make_metaworld(name,
                   frame_stack,
                   action_repeat,
                   discount,
                   seed,
                   camera_name,
                   add_segmentation_to_obs,
                   noisy_mask_drop_prob,
                   use_rgbm=None,
                   slim_mask_cfg=None,
                   add_temporal_dim=False):
    # discount = 1.0
    env = MT_Wrapper(env_name=name,
                     discount=discount,
                     seed=seed,
                     proprioceptive_state=True)

    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env,
                              action_repeat,
                              use_metaworld_reward_dict=True)
    env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)

    frame_keys = []

    rgb_key = "pixels"
    frame_keys.append(rgb_key)
    render_kwargs = dict(height=84,
                         width=84,
                         mode="offscreen",
                         camera_name=camera_name)
    env = pixels.Wrapper(env,
                         pixels_only=False,
                         render_kwargs=render_kwargs,
                         observation_key=rgb_key)

    if add_segmentation_to_obs:
        segmentation_key = "segmentation"
        frame_keys.append(segmentation_key)
        segmentation_kwargs = dict(height=84*3,
                                   width=84*3,
                                   mode="offscreen",
                                   camera_name=camera_name,
                                   segmentation=True)
        env = pixels.Wrapper(env,
                             pixels_only=False,
                             render_kwargs=segmentation_kwargs,
                             observation_key=segmentation_key)
        env = SegmentationToRobotMaskWrapper(env, segmentation_key)

        env = SegmentationFilter(env, segmentation_key)

        if noisy_mask_drop_prob > 0:
            env = NoisyMaskWrapper(env,
                                   segmentation_key,
                                   prob_drop=noisy_mask_drop_prob)

        if slim_mask_cfg and slim_mask_cfg.use_slim_mask:
            env = SlimMaskWrapper(env, segmentation_key, slim_mask_cfg.scale,
                                  slim_mask_cfg.threshold, slim_mask_cfg.sigma)

        if use_rgbm:
            env = StackRGBAndMaskWrapper(env,
                                         rgb_key,
                                         segmentation_key,
                                         new_key="pixels")
            frame_keys = ["pixels"]

    env = FrameStackWrapper(env, frame_stack, frame_keys, add_temporal_dim)
    env = ExtendedTimeStepWrapper(env, has_success_metric=True)
    
    return env
