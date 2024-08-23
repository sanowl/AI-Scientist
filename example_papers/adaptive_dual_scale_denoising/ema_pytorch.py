from __future__ import annotations

from copy import deepcopy
from functools import partial
import math
import torch
from torch import Tensor
from torch.nn import Module
from typing import Set, Callable, Dict, Optional
import matplotlib.pyplot as plt
import logging

def exists(val):
    return val is not None

def inplace_copy(tgt: Tensor, src: Tensor, *, auto_move_device=False):
    if auto_move_device:
        src = src.to(tgt.device)
    tgt.copy_(src)

def inplace_lerp(tgt: Tensor, src: Tensor, weight, *, auto_move_device=False):
    if auto_move_device:
        src = src.to(tgt.device)
    tgt.lerp_(src, weight)

class EMA(Module):
    def __init__(
            self,
            model: Module,
            ema_model: Optional[Module] = None,
            beta=0.9999,
            update_after_step=100,
            update_every=10,
            inv_gamma=1.0,
            power=2 / 3,
            min_value=0.0,
            param_or_buffer_names_no_ema: Set[str] = set(),
            ignore_names: Set[str] = set(),
            ignore_startswith_names: Set[str] = set(),
            include_online_model=True,
            allow_different_devices=False,
            use_foreach=False,
            gradient_accumulation_steps=1,
            custom_decay_schedule: Optional[Callable[[int], float]] = None,
            param_group_ema_rates: Optional[Dict[str, float]] = None,
            adaptive_ema_params: Optional[Dict] = None,
            memory_efficient=False,
            warm_restart_every: Optional[int] = None
    ):
        super().__init__()
        self.beta = beta
        self.is_frozen = beta == 1.
        self.include_online_model = include_online_model
        self.online_model = model if include_online_model else [model]
        self.ema_model = ema_model or self._create_ema_model(model)
        self.parameter_names = {name for name, param in self.ema_model.named_parameters() if torch.is_floating_point(param) or torch.is_complex(param)}
        self.buffer_names = {name for name, buffer in self.ema_model.named_buffers() if torch.is_floating_point(buffer) or torch.is_complex(buffer)}
        self.inplace_copy = partial(inplace_copy, auto_move_device=allow_different_devices)
        self.inplace_lerp = partial(inplace_lerp, auto_move_device=allow_different_devices)
        self.update_every = update_every
        self.update_after_step = update_after_step
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value
        self.param_or_buffer_names_no_ema = param_or_buffer_names_no_ema
        self.ignore_names = ignore_names
        self.ignore_startswith_names = ignore_startswith_names
        self.allow_different_devices = allow_different_devices
        self.use_foreach = use_foreach
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.custom_decay_schedule = custom_decay_schedule
        self.param_group_ema_rates = param_group_ema_rates or {}
        self.adaptive_ema_params = adaptive_ema_params
        self.memory_efficient = memory_efficient
        self.warm_restart_every = warm_restart_every
        self.register_buffer('initted', torch.tensor(False))
        self.register_buffer('step', torch.tensor(0))
        self.register_buffer('decay_history', torch.zeros(1000))
        self.logger = logging.getLogger(self.__class__.__name__)
        logging.basicConfig(level=logging.INFO)

    def _create_ema_model(self, model):
        try:
            return deepcopy(model)
        except Exception as e:
            self.logger.error(f'Error while trying to deepcopy model: {e}')
            self.logger.error('Your model was not copyable. Please make sure you are not using any LazyLinear')
            raise

    @property
    def model(self):
        return self.online_model if self.include_online_model else self.online_model[0]

    def eval(self):
        return self.ema_model.eval()

    def restore_ema_model_device(self):
        device = self.initted.device
        self.ema_model.to(device)

    def get_params_iter(self, model):
        for name, param in model.named_parameters():
            if name not in self.parameter_names:
                continue
            yield name, param

    def get_buffers_iter(self, model):
        for name, buffer in model.named_buffers():
            if name not in self.buffer_names:
                continue
            yield name, buffer

    def copy_params_from_model_to_ema(self):
        for (_, ma_params), (_, current_params) in zip(self.get_params_iter(self.ema_model), self.get_params_iter(self.model)):
            self.inplace_copy(ma_params.data, current_params.data)
        for (_, ma_buffers), (_, current_buffers) in zip(self.get_buffers_iter(self.ema_model), self.get_buffers_iter(self.model)):
            self.inplace_copy(ma_buffers.data, current_buffers.data)

    def copy_params_from_ema_to_model(self):
        for (_, ma_params), (_, current_params) in zip(self.get_params_iter(self.ema_model), self.get_params_iter(self.model)):
            self.inplace_copy(current_params.data, ma_params.data)
        for (_, ma_buffers), (_, current_buffers) in zip(self.get_buffers_iter(self.ema_model), self.get_buffers_iter(self.model)):
            self.inplace_copy(current_buffers.data, ma_buffers.data)

    def get_current_decay(self):
        if self.custom_decay_schedule:
            return self.custom_decay_schedule(self.step.item())
        epoch = (self.step - self.update_after_step - 1).clamp(min=0.)
        value = 1 - (1 + epoch / self.inv_gamma) ** -self.power
        if epoch.item() <= 0:
            return 0.
        return value.clamp(min=self.min_value, max=self.beta).item()

    def update(self):
        step = self.step.item()
        self.step += 1
        if step % (self.update_every * self.gradient_accumulation_steps) != 0:
            return
        if step <= self.update_after_step:
            self.copy_params_from_model_to_ema()
            return
        if not self.initted.item():
            self.copy_params_from_model_to_ema()
            self.initted.data.copy_(torch.tensor(True))
        self.update_moving_average(self.ema_model, self.model)
        if self.warm_restart_every and step % self.warm_restart_every == 0:
            self.reset_decay_rate()

    @torch.no_grad()
    def update_moving_average(self, ma_model, current_model):
        if self.is_frozen:
            return
        current_decay = self.get_current_decay()
        self.decay_history[self.step % 1000] = current_decay
        tensors_to_copy = []
        tensors_to_lerp = []
        for (name, current_params), (_, ma_params) in zip(self.get_params_iter(current_model), self.get_params_iter(ma_model)):
            if self._should_ignore(name):
                continue
            if name in self.param_or_buffer_names_no_ema:
                tensors_to_copy.append((ma_params.data, current_params.data))
            else:
                group_decay = self.param_group_ema_rates.get(name, current_decay)
                tensors_to_lerp.append((ma_params.data, current_params.data, group_decay))
        for (name, current_buffer), (_, ma_buffer) in zip(self.get_buffers_iter(current_model), self.get_buffers_iter(ma_model)):
            if self._should_ignore(name):
                continue
            if name in self.param_or_buffer_names_no_ema:
                tensors_to_copy.append((ma_buffer.data, current_buffer.data))
            else:
                tensors_to_lerp.append((ma_buffer.data, current_buffer.data, current_decay))
        self._apply_update(tensors_to_copy, tensors_to_lerp)

    def _should_ignore(self, name):
        return name in self.ignore_names or any(name.startswith(prefix) for prefix in self.ignore_startswith_names)

    def _apply_update(self, tensors_to_copy, tensors_to_lerp):
        if not self.use_foreach:
            for tgt, src in tensors_to_copy:
                self.inplace_copy(tgt, src)
            for tgt, src, decay in tensors_to_lerp:
                self.inplace_lerp(tgt, src, 1. - decay)
        else:
            if self.allow_different_devices:
                tensors_to_copy = [(tgt, src.to(tgt.device)) for tgt, src in tensors_to_copy]
                tensors_to_lerp = [(tgt, src.to(tgt.device), decay) for tgt, src, decay in tensors_to_lerp]
            if tensors_to_copy:
                tgt_copy, src_copy = zip(*tensors_to_copy)
                torch._foreach_copy_(tgt_copy, src_copy)
            if tensors_to_lerp:
                tgt_lerp, src_lerp, decays = zip(*tensors_to_lerp)
                torch._foreach_lerp_(tgt_lerp, src_lerp, [1. - d for d in decays])

    def __call__(self, *args, **kwargs):
        return self.ema_model(*args, **kwargs)

    def reset_decay_rate(self):
        self.step.zero_()
        self.logger.info("EMA decay rate reset (warm restart)")

    def visualize_decay_rate(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.decay_history.cpu().numpy())
        plt.title("EMA Decay Rate Over Time")
        plt.xlabel("Step")
        plt.ylabel("Decay Rate")
        plt.show()

    def parameter_difference(self):
        diff = {}
        for (name, ma_params), (_, current_params) in zip(self.get_params_iter(self.ema_model), self.get_params_iter(self.model)):
            diff[name] = torch.norm(ma_params - current_params).item()
        return diff

    def log_parameter_difference(self):
        diff = self.parameter_difference()
        for name, value in diff.items():
            self.logger.info(f"Parameter difference for {name}: {value}")

    def adaptive_update(self, validation_loss):
        if self.adaptive_ema_params:
            current_decay = self.get_current_decay()
            new_decay = current_decay
            if validation_loss < self.adaptive_ema_params['best_loss']:
                new_decay = min(current_decay * self.adaptive_ema_params['increase_factor'], self.beta)
                self.adaptive_ema_params['best_loss'] = validation_loss
            else:
                new_decay = max(current_decay * self.adaptive_ema_params['decrease_factor'], self.min_value)
            self.beta = new_decay
            self.logger.info(f"Adaptive EMA rate updated to {new_decay}")

    def state_dict(self):
        state_dict = super().state_dict()
        if self.memory_efficient:
            ema_state_dict = self.ema_model.state_dict()
            for key in ema_state_dict:
                ema_state_dict[key] = ema_state_dict[key] - self.model.state_dict()[key]
            state_dict['ema_state_diff'] = ema_state_dict
        return state_dict

    def load_state_dict(self, state_dict):
        if self.memory_efficient and 'ema_state_diff' in state_dict:
            ema_state_diff = state_dict.pop('ema_state_diff')
            model_state = self.model.state_dict()
            for key in ema_state_diff:
                ema_state_diff[key] = ema_state_diff[key] + model_state[key]
            self.ema_model.load_state_dict(ema_state_diff)
        super().load_state_dict(state_dict)