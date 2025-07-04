# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import datetime
import io
import random
import traceback
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import IterableDataset


def episode_len(episode):
    # subtract -1 because the dummy first transition
    return next(iter(episode.values())).shape[0] - 1


def save_episode(episode, fn):
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with fn.open("wb") as f:
            f.write(bs.read())


def load_episode(fn):
    with fn.open("rb") as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
        return episode


def get_nested_value_from_flattened_key(nested_dict, flattened_key,
                                        key_delimeter):
    value = nested_dict
    for key in flattened_key.split(key_delimeter):
        value = value[key]
    return value


def flatten_nested_dict(nested_dict, key_delimiter):
    flattened_dict = {}
    for key, value in nested_dict.items():
        assert (
            key_delimiter not in key
        ), f"Error flattening nested dictionary. The key delimiter '{key_delimiter}' was found in key '{key}'. "
        if isinstance(value, dict):
            for sub_key, sub_value in flatten_nested_dict(
                    value, key_delimiter).items():
                flattened_dict[str(key) + str(key_delimiter) +
                               str(sub_key)] = sub_value
        else:
            flattened_dict[str(key)] = value
    return flattened_dict


def unflatten_dict(flattened_dict, key_delimiter):
    nested_dict = {}
    for key, value in flattened_dict.items():
        sub_keys = key.split(key_delimiter)
        d = nested_dict
        for sub_key in sub_keys[0:-1]:
            if sub_key not in d:
                d[sub_key] = {}
            d = d[sub_key]
        d[sub_keys[-1]] = value
    return nested_dict


def get_timestep_from_nested_dict(nested_dict, time_step):
    result = {}
    for key, value in nested_dict.items():
        if isinstance(value, dict):
            result[str(key)] = get_timestep_from_nested_dict(value, time_step)
        else:
            result[str(key)] = value[time_step]
    return result


class ReplayBufferStorage:

    def __init__(self, data_specs, replay_dir):
        self._nesting_delimiter = "."
        self._data_specs = flatten_nested_dict(data_specs,
                                               self._nesting_delimiter)
        self._replay_dir = replay_dir
        replay_dir.mkdir(exist_ok=True)
        self._current_episode = defaultdict(list)
        self._preload()

    def __len__(self):
        return self._num_transitions

    def add(self, time_step):
        for spec_name, spec in self._data_specs.items():
            value = get_nested_value_from_flattened_key(time_step, spec_name,
                                                        self._nesting_delimiter)
            if np.isscalar(value):
                value = np.full(spec.shape, value, spec.dtype)
            assert spec.shape == value.shape, f"Shape of spec - {spec.shape}, does not match the value shape - {value.shape}"
            assert spec.dtype == value.dtype, f"Dtype of spec - {spec.dtype}, does not match the value dtype - {value.dtype}"
            self._current_episode[spec_name].append(value)
        if time_step.last():
            episode = {}
            for spec_name, spec in self._data_specs.items():
                value = self._current_episode[spec_name]
                episode[spec_name] = np.array(value, spec.dtype)
            self._current_episode = defaultdict(list)
            self._store_episode(episode)

    def _preload(self):
        self._num_episodes = 0
        self._num_transitions = 0
        for fn in self._replay_dir.glob("*.npz"):
            _, _, eps_len = fn.stem.split("_")
            self._num_episodes += 1
            self._num_transitions += int(eps_len)

    def _store_episode(self, episode):
        eps_idx = self._num_episodes
        eps_len = episode_len(episode)
        self._num_episodes += 1
        self._num_transitions += eps_len
        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        eps_fn = f"{ts}_{eps_idx}_{eps_len}.npz"
        save_episode(episode, self._replay_dir / eps_fn)


class ReplayBuffer(IterableDataset):

    def __init__(self, replay_dir, max_size, num_workers, nstep, discount,
                 fetch_every, save_buffer_snapshot, has_success_metric,
                 context_steps=None, horizon=None, requires_extended_obs=False):
        self._nesting_delimiter = "."
        self._replay_dir = replay_dir
        self._size = 0
        self._max_size = max_size
        self._num_workers = max(1, num_workers)
        self._episode_fns = []
        self._episodes = {}
        self._nstep = nstep
        self._discount = discount
        self._fetch_every = fetch_every
        self._samples_since_last_fetch = fetch_every
        self._save_buffer_snapshot = save_buffer_snapshot
        self.has_success_metric = has_success_metric
        self.context_steps = context_steps
        self.horizon = horizon
        self.requires_extended_obs = requires_extended_obs

    def _sample_episode(self):
        eps_fn = random.choice(self._episode_fns)
        return self._episodes[eps_fn]

    def _store_episode(self, eps_fn):
        try:
            episode = load_episode(eps_fn)
        except (OSError, IOError):
            return False
        eps_len = episode_len(episode)
        while eps_len + self._size > self._max_size:
            early_eps_fn = self._episode_fns.pop(0)
            early_eps = self._episodes.pop(early_eps_fn)
            self._size -= episode_len(early_eps)
            early_eps_fn.unlink(missing_ok=True)
        self._episode_fns.append(eps_fn)
        self._episode_fns.sort()
        self._episodes[eps_fn] = episode
        self._size += eps_len

        if not self._save_buffer_snapshot:
            eps_fn.unlink(missing_ok=True)
        return True

    def _try_fetch(self):
        if self._samples_since_last_fetch < self._fetch_every:
            return
        self._samples_since_last_fetch = 0
        try:
            worker_id = torch.utils.data.get_worker_info().id
        except AttributeError:
            worker_id = 0
        eps_fns = sorted(self._replay_dir.glob("*.npz"), reverse=True)
        fetched_size = 0
        for eps_fn in eps_fns:
            eps_idx, eps_len = [int(x) for x in eps_fn.stem.split("_")[1:]]
            if eps_idx % self._num_workers != worker_id:
                continue
            if eps_fn in self._episodes:
                break
            if fetched_size + eps_len > self._max_size:
                break
            fetched_size += eps_len
            if not self._store_episode(eps_fn):
                break

    def _sample(self):
        try:
            self._try_fetch()
        except Exception:
            traceback.print_exc()
        self._samples_since_last_fetch += 1
        episode = self._sample_episode()
        _episode_len = episode_len(episode)
        n = self._nstep if self.requires_extended_obs is False else \
            np.max([self._nstep, self.context_steps + self.horizon])
        episode = unflatten_dict(episode, self._nesting_delimiter)
        if self.requires_extended_obs:
            # add +1 for the first dummy transition
            idx_extended = np.random.randint(0, _episode_len - n + 1) + 1
            idx = idx_extended + self.context_steps - 1
        else:
            idx = np.random.randint(0, _episode_len - self._nstep + 1) + 1
        
        obs = get_timestep_from_nested_dict(episode["observation"], idx - 1)
        action = episode["action"][idx]
        next_obs = get_timestep_from_nested_dict(episode["observation"], idx + self._nstep - 1)
        
        if self.requires_extended_obs:
            extended_obs = {'pixels': [], 'action': []}
            for i in range(0, self.context_steps+self.horizon):
                obs_t = get_timestep_from_nested_dict(episode["observation"], idx_extended + i - 1)
                extended_obs['pixels'].append(obs_t['pixels'][-1])
            extended_obs['pixels'] = np.stack(extended_obs['pixels'], axis=0)
            extended_obs['action'] = episode["action"][idx_extended-1:idx_extended+self.context_steps+self.horizon-1]
        
        if self.has_success_metric:
            reward = np.zeros_like(episode["reward"]["reward"][idx])
        else:
            reward = np.zeros_like(episode["reward"][idx])
        discount = np.ones_like(episode["discount"][idx])
        for i in range(self._nstep):
            if self.has_success_metric:
                step_reward = episode["reward"]["reward"][idx + i]
            else:
                step_reward = episode["reward"][idx + i]
            reward += discount * step_reward
            discount *= episode["discount"][idx + i] * self._discount
        if self.requires_extended_obs:
            return (extended_obs, action, reward, discount, next_obs)
        else:
            return (obs, action, reward, discount, next_obs)

    def __iter__(self):
        while True:
            yield self._sample()


def _worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)


def make_replay_loader(replay_dir, max_size, batch_size, num_workers, save_buffer_snapshot, nstep, discount,
                       has_success_metric, context_steps, horizon, requires_extended_obs):
    max_size_per_worker = max_size // max(1, num_workers)

    iterable = ReplayBuffer(replay_dir,
                            max_size_per_worker,
                            num_workers,
                            nstep,
                            discount,
                            fetch_every=1000,
                            save_buffer_snapshot=save_buffer_snapshot,
                            has_success_metric=has_success_metric,
                            context_steps=context_steps,
                            horizon=horizon,
                            requires_extended_obs=requires_extended_obs)

    loader = torch.utils.data.DataLoader(
        iterable,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=_worker_init_fn,
    )
    return loader
