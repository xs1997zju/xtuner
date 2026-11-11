from pathlib import Path
import os
from recipe.cybergym.main_config import build_config
from xtuner.v1.data_proto.rl_data import SampleParams
from xtuner.v1.datasets.config import DataloaderConfig,DatasetConfig
from xtuner.v1.datasets.rl_tokenize_fn.text_tokenize_fn import RLTextTokenizeFnConfig
from xtuner.v1.rl.agent_loop_manager import SamplerConfig
from xtuner.v1.train.trainer import LoadCheckpointConfig
from controlled_replay import ReplayLoopConfig
from runtime import ROOT

def build(checkpoint=None):
    cfg=build_config(phase='online',initialization='dcp-resume' if checkpoint else 'fresh',checkpoint=checkpoint)
    cfg.work_dir=Path(os.environ['WORK_DIR'])/'trainer'
    cfg.train_batch_size=1;cfg.total_train_steps=2
    cfg.checkpoint_interval=2;cfg.hf_interval=None
    cfg.resources.num_workers=8;cfg.resources.num_accelerators_per_node=8;cfg.resources.num_cpus_per_worker=4
    cfg.rollout_config.enable_proxy=False
    cfg.rollout_config.worker_log_dir=Path(os.environ['WORK_DIR'])/'rollout_logs'
    cfg.rollout_config.rollout_max_batch_size_per_instance=2
    cfg.rollout_config.max_prefill_token_num=512
    task=cfg.agent_loop_manager_cfg.tasks
    task.task_name='split_replay_control'
    task.agent_loop_config=ReplayLoopConfig(hf_checkpoint=cfg.tokenizer_path,sample_params=SampleParams(max_tokens=16))
    task.produce_strategy_config.max_pending_tasks=2
    task.sampler_config=SamplerConfig(prompt_repeat_k=8,dataloader_cfg=DataloaderConfig(
        dataset_config_list=[{'dataset':DatasetConfig(name='split_replay_control',anno_path=str(ROOT/'gpu_jobs/tasks.jsonl'),
             sample_ratio=1,class_name='VLMJsonlDataset'),'tokenize_fn':RLTextTokenizeFnConfig(max_length=1024)}],
        num_workers=0,collator='fake_collator',pack_level='none',pack_max_length=262144))
    # Native config validators were run by build_config. All enabled gates,
    # provenance, async strategy and complete resume requirements remain intact.
    assert cfg.train_worker_cfg.sp_size==4 and cfg.rollout_config.turn_resume_initially_closed
    assert cfg.agent_loop_manager_cfg.turn_keepalive
    assert cfg.train_worker_cfg.model_cfg.text_config.mtp_config.frozen
    assert cfg.checkpoint_no_save_optimizer is False and cfg.checkpoint_no_save_replay_buffer is False
    return cfg
