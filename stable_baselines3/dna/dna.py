from typing import Any, Dict, Optional, Type, TypeVar, Union

import numpy as np
import torch
import torch as th
from gym import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule, RolloutBufferSamples
from stable_baselines3.common.utils import explained_variance, get_schedule_fn, obs_as_tensor

DNASelf = TypeVar("DNASelf", bound="DNA")

class DNAOptimizer():

    def __init__(self, name: str, epochs: int, minibatch_size: int):
        self.name = name
        self.epochs = epochs
        self.minibatch_size = minibatch_size

class DNA(OnPolicyAlgorithm):
    """
    Proximal Policy Optimization with a Dual Network Archecture (PPO-DNA)

    Paper: https://arxiv.org/abs/2206.10027
    Code: This implementation borrows code from the authors repo (https://github.com/maitchison/PPO/tree/DNA)

    This implementation follows closely to that described in the paper with the following changes
    * Incorporation of gSDE
    * Distilation uses L1 constraint on log prob of action taken. (makes integration with SB simpler)
    * Default settings for beta1 have changed for Atari.

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0), this is used for all three optimizers.
    :param n_steps: The number of steps to run for each environment per update
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
        NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
        See https://github.com/pytorch/pytorch/issues/29372
    :param policy_minibatch_size: Minibatch size for policy update
    :param value_minibatch_size: Minibatch size for value update
    :param distil_minibatch_size: Minibatch size for value update
    :param policy_epochs: Number of epoch when optimizing the policy loss
    :param value_epochs: Number of epoch when optimizing the value loss
    :param distil_epochs: Number of epoch when optimizing the distil loss
    :param policy_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator when calculating advantages.
    :param value_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator when calculating value targets.
    :param beta: Weight of distilation constraint.
    :param gamma: Discount factor
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param normalize_advantage: Whether to normalize or not the advantage
    :param ent_coef: Entropy coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        policy_minibatch_size: int = 64,
        value_minibatch_size: int = 64,
        distil_minibatch_size: int = 64,
        policy_epochs: int = 10,
        value_epochs: int = 10,
        distil_epochs: int = 10,
        policy_lambda: float = 0.95,
        value_lambda: float = 0.95,
        gamma: float = 0.99,
        beta: float = 1.0,
        clip_range: Union[float, Schedule] = 0.2,
        normalize_advantage: bool = True,
        ent_coef: float = 0.01, # << stub: check if this is correct.
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):

        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=policy_lambda,
            ent_coef=ent_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
            vf_coef=0.5, # not used.
        )

        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        if normalize_advantage:
            assert (
                policy_minibatch_size > 1
            ), "`minibatch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert buffer_size > 1 or (
                not normalize_advantage
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            for (mb_size, mb_name) in [
                (policy_minibatch_size, "policy_minibatch_size"),
                (value_minibatch_size, "value_minibatch_size"),
                (distil_minibatch_size, "distil_minibatch_size"),
            ]:
                assert buffer_size % mb_size == 0, f"You have specified a {mb_name} of {mb_size} " \
                    f"but because the 'RolloutBuffer' is of size `n_steps * n_envs = {buffer_size}`," \
                    f" there will be a truncated mini-batch of size {buffer_size % mb_size}.\n" \
                    f"DNA does not yet support truncated mini-batches."

        self.policy_optimizer = DNAOptimizer("Policy", policy_epochs, policy_minibatch_size)
        self.value_optimizer = DNAOptimizer("Value", value_epochs, value_minibatch_size)
        self.distil_optimizer = DNAOptimizer("Distil", distil_epochs, distil_minibatch_size)

        self.policy_lambda = policy_lambda
        self.value_lambda = value_lambda
        self.beta = beta

        self.clip_range = clip_range
        self.normalize_advantage = normalize_advantage

        if _init_setup_model:
            self._setup_model()


    @staticmethod
    def default_atari(env: Union[GymEnv, str], **kwargs) -> DNASelf:
        """
        Creates a DNA instance with default args for atari.
        """
        # note: agents should be 128
        # todo: add sde
        atari_default_args = {}
        atari_default_args.update(
            policy="CnnPolicy",
            learning_rate=2.5e-4,
            n_steps=128,
            policy_minibatch_size=2048,
            value_minibatch_size=256,
            distil_minibatch_size=256,
            policy_epochs=2,
            value_epochs=1,
            distil_epochs=2,
            policy_lambda=0.8,
            value_lambda=0.95,
            gamma=0.999,
            beta=1.0,
            clip_range=0.2,
            normalize_advantage=True,
            ent_coef=0.01,  # << stub: check if this is correct.
            max_grad_norm=5.0,
        )

        args = atari_default_args.copy()
        args.update(kwargs)

        return DNA(env=env, **args)


    def _setup_model(self) -> None:
        super()._setup_model()

        # setup value network
        self.value = self.policy_class(  # pytype:disable=not-instantiable
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            **self.policy_kwargs  # pytype:disable=not-instantiable
        )
        self.value = self.value.to(self.device)

        # Setup optimizers
        # todo... need to create a few more optimizers and link them..

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)

    def _train_policy_minibatch(self, rollout_data: RolloutBufferSamples) -> None:

        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)

        entropy_losses = []
        pg_losses = []
        clip_fractions = []

        actions = rollout_data.actions
        if isinstance(self.action_space, spaces.Discrete):
            # Convert discrete action from float to long
            actions = rollout_data.actions.long().flatten()

        # Re-sample the noise matrix because the log_std has changed
        if self.use_sde:
            self.policy.reset_noise(self.policy_optimizer.minibatch_size)

        _, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)

        # Normalize advantage
        advantages = rollout_data.advantages

        # Normalization does not make sense if mini batchsize == 1, see GH issue #325
        if self.normalize_advantage and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ratio between old and new policy, should be one at the first iteration
        ratio = th.exp(log_prob - rollout_data.old_log_prob)

        # clipped surrogate loss
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
        policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

        # Logging
        pg_losses.append(policy_loss.item())
        clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
        clip_fractions.append(clip_fraction)

        # Entropy loss favor exploration
        if entropy is None:
            # Approximate entropy when no analytical form
            entropy_loss = -th.mean(-log_prob)
        else:
            entropy_loss = -th.mean(entropy)

        entropy_losses.append(entropy_loss.item())

        loss = policy_loss + self.ent_coef * entropy_loss

        # clip grad norm and perform update
        self.policy.optimizer.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy.optimizer.step()

        # logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        # self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)

    def _train_value_minibatch(self, rollout_data: RolloutBufferSamples) -> None:

        value_losses = []

        pred_values = self.value.predict_values(rollout_data.observations)
        pred_values = pred_values.flatten()

        # value loss using the TD(value_lambda) target
        value_loss = F.mse_loss(rollout_data.returns, pred_values)
        value_losses.append(value_loss.item())

        # clip grad norm and perform update
        self.value.optimizer.zero_grad()
        value_loss.backward()
        th.nn.utils.clip_grad_norm_(self.value.parameters(), self.max_grad_norm)
        self.value.optimizer.step()

        # todo move logs somewhere else, return losses
        # logs
        self.logger.record("train/value_loss", np.mean(value_losses))

    def _train_distil_minibatch(self, rollout_data: RolloutBufferSamples) -> None:

        distil_losses = []

        actions = rollout_data.actions
        if isinstance(self.action_space, spaces.Discrete):
            # Convert discrete action from float to long
            actions = rollout_data.actions.long().flatten()

        # todo: I should be able to do this one one call.. probably torch caches it anyway?
        pred_values, new_log_probs, _ = self.policy.evaluate_actions(rollout_data.observations, actions)

        # this constraint can be implemented without editing the other files in SB.
        # but it's not quite right...
        distil_loss_kl = th.abs(new_log_probs - rollout_data.old_log_prob).mean()

        # value part
        pred_values = pred_values.flatten()
        distil_loss_mse = F.mse_loss(rollout_data.returns, pred_values)

        # distil loss
        distil_loss = distil_loss_mse + self.beta * distil_loss_kl

        # clip grad norm and perform update
        self.policy.optimizer.zero_grad()
        distil_loss.backward()
        th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.policy.optimizer.step()

        # logs
        distil_losses.append(distil_loss.item())
        self.logger.record("train/distil_loss", np.mean(distil_losses))


    def _calculate_return_estimates(self, rollout_buffer:RolloutBuffer, last_values:np.ndarray, dones:np.ndarray, td_lambda: float) -> np.ndarray:
        """
        Generate a return estimate from replay buffer via TD(\lambda).
        """

        # Convert to numpy
        last_values = last_values.flatten()

        last_gae_lam = 0
        advantages = np.zeros_like(rollout_buffer.values)
        for step in reversed(range(rollout_buffer.buffer_size)):
            if step == rollout_buffer.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - rollout_buffer.episode_starts[step + 1]
                next_values = rollout_buffer.values[step + 1]
            delta = rollout_buffer.rewards[step] + self.gamma * next_values * next_non_terminal - rollout_buffer.values[step]
            last_gae_lam = delta + self.gamma * td_lambda * next_non_terminal * last_gae_lam
            advantages[step] = last_gae_lam

        return advantages + rollout_buffer.values

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        # todo: update distil optimizer..
        self._update_learning_rate(self.policy.optimizer)
        self._update_learning_rate(self.value.optimizer)

        # Update normalization
        # Note: we build normalization into the model rather than the environment, this is done for two reasons
        # 1. It the model takes as input standard frames like any other model
        # 2. It allows the normalization constants to be saved with the model.

        # Update target to use correct lambda
        if self.value_lambda != self.policy_lambda:
            dones = self._last_episode_starts
            with torch.no_grad():
                last_values = self.value.predict_values(obs_as_tensor(self._last_obs, self.device))
                last_values = last_values.clone().cpu().numpy()
            self.rollout_buffer.returns = self._calculate_return_estimates(self.rollout_buffer, last_values, dones, self.value_lambda)

        # Policy phase
        for epoch in range(self.policy_optimizer.epochs):
            for rollout_data in self.rollout_buffer.get(self.policy_optimizer.minibatch_size):
                self._train_policy_minibatch(rollout_data)

        # Get old log probs for distilation
        # todo: maybe better to do in minibatches?
        # this would require knowing the indices of the rollout sample ... (small change)
        # or just passing ids in to _get_samples (better)
        _, new_log_probs, _ = self.policy.evaluate_actions(
            self.rollout_buffer.to_torch(self.rollout_buffer.observations),
            self.rollout_buffer.to_torch(self.rollout_buffer.actions.flatten()), # this is not right for CC.
        )
        self.rollout_buffer.log_probs = new_log_probs

        # Value phase
        for epoch in range(self.value_optimizer.epochs):
            for rollout_data in self.rollout_buffer.get(self.value_optimizer.minibatch_size):
                self._train_value_minibatch(rollout_data)

        # Distil phase
        for epoch in range(self.distil_optimizer.epochs):
            for rollout_data in self.rollout_buffer.get(self.distil_optimizer.minibatch_size):
                self._train_distil_minibatch(rollout_data)

        # logging
        self._n_updates += self.policy_optimizer.epochs + self.value_optimizer.epochs + self.distil_optimizer.epochs

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
        self.logger.record("train/explained_variance", explained_var)

    def learn(
        self: DNASelf,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "DNA",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> DNASelf:

        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
