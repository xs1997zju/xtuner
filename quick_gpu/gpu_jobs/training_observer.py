"""Read-only witnesses around the real native actor optimizer operation."""
import hashlib
import json
import os
from pathlib import Path
import time

import torch
from xtuner.v1.rl.trainer.worker import TrainingWorker
from xtuner.v1.utils.type_helper import ray_method


def digest(parameter, full):
    value = getattr(parameter, '_local_tensor', parameter).detach()
    if not full:
        value = value[tuple(slice(0, min(n, cap)) for n, cap in
                            zip(value.shape, [8, 1024, 1024]))]
    value = value.to('cpu').contiguous()
    return dict(shape=list(parameter.shape), local_shape=list(value.shape),
                numel=value.numel(), dtype=str(value.dtype), full_local_tensor=full,
                sha256=hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())


class ObservedTrainingWorker(TrainingWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._validation_mtp_calls = 0
        self._validation_mtp_hook_count = 0
        for name, module in self.actor.engine.model.named_modules():
            if name.endswith('mtp_block'):
                module.register_forward_pre_hook(self._observe_mtp_forward)
                self._validation_mtp_hook_count += 1
        assert self._validation_mtp_hook_count == 1

    def _observe_mtp_forward(self, *_):
        self._validation_mtp_calls += 1

    def _capture(self, phase):
        parameters = dict(self.actor.engine.model.named_parameters())
        mtp = {n: p for n, p in parameters.items() if '.mtp_block.' in n}
        assert len(mtp) == 19
        optimizer = self.actor.engine.optimizer
        members = {id(p) for group in optimizer.param_groups for p in group['params']}
        assert all(not p.requires_grad and p.grad is None and id(p) not in members for p in mtp.values())
        assert self.actor.mtp_config is None and not self.actor.mtp_cfgs
        assert self._validation_mtp_calls == 0
        assert self._validation_mtp_hook_count == 1
        main_names = sorted(n for n in parameters if '.mtp_block.' not in n and
                            'norm' in n and n.endswith('weight'))[:32]
        io_names = sorted(n for n in parameters if '.mtp_block.' not in n and
                          (n.endswith('embed_tokens.weight') or n.endswith('lm_head.weight')))
        assert main_names and io_names
        steps = []
        for state in optimizer.state.values():
            if 'step' in state:
                step = state['step']
                steps.append(int(step.item() if torch.is_tensor(step) else step))
        event = dict(phase=phase, rank=self.rank, pid=os.getpid(), time_ns=time.time_ns(),
            worker_step=self.actor._global_train_step, mtp_forward_calls=self._validation_mtp_calls,
            mtp_forward_hook_count=self._validation_mtp_hook_count,
            mtp_optimizer_memberships=0,
            mtp={n: digest(p, True) for n, p in mtp.items()},
            main={n: digest(parameters[n], False) for n in main_names},
            shared_io={n: digest(parameters[n], False) for n in io_names},
            optimizer_step_min=min(steps) if steps else None,
            optimizer_step_max=max(steps) if steps else None)
        target = Path(os.environ['WORK_DIR'])/'training_parameters'/f'rank-{self.rank}.jsonl'
        target.parent.mkdir(exist_ok=True)
        with target.open('a') as stream:
            stream.write(json.dumps(event, sort_keys=True)+'\n')
            stream.flush()
            os.fsync(stream.fileno())
        return event

    @ray_method
    def actor_fit(self):
        self.actor.require_train_residency()
        before = self._capture('before_actor_fit')
        result = super().actor_fit()
        after = self._capture('after_actor_fit')
        assert all(before['mtp'][name]['sha256'] == row['sha256']
                   for name, row in after['mtp'].items())
        return result

    @ray_method
    def validation_snapshot(self):
        return self._capture('validation_snapshot')
