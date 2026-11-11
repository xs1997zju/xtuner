"""Portable environment adapter; mechanism modules retain the reviewed logic."""
from pathlib import Path
import os, socket
ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT
PYTHON=Path(os.environ['PYTHON_BIN']).resolve()
PT=Path(os.environ['REPLAY_PT']).resolve()
def environment(out):
 env=os.environ.copy()
 required=['MODEL_PATH','XTUNER_PATH','LAGENT_PATH','LMDEPLOY_PATH','CRG_PATH']
 for name in required:
  if not env.get(name):raise RuntimeError('Missing '+name)
 env.update(WORK_DIR=str(out), XTUNER_RUN_ID='quick-'+out.parent.name,
  DEBUG_ROLLOUT_DIR=str(PT.parent),TOTAL_TRAIN_STEPS='2',
  XTUNER_USE_LMDEPLOY='1',XTUNER_USE_VLLM='0',XTUNER_USE_SGLANG='0',XTUNER_USE_FA3='1',
  LMDEPLOY_FP32_MAMBA_SSM_DTYPE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
  OMP_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
  NCCL_IB_DISABLE='1',NCCL_SOCKET_IFNAME='lo',GLOO_SOCKET_IFNAME='lo',
  ROLLOUT_DIST_PORT_BASE=env.get('ROLLOUT_DIST_PORT_BASE','31000'),
  WEIGHT_UPDATE_PORT=env.get('WEIGHT_UPDATE_PORT','33000'),
  RL_LLM_BASE_URL='http://127.0.0.1:1/v1',RL_LLM_ADMIN_URL='http://127.0.0.1:1',
  XTUNER_ROLLOUT_RAY_GET_TIMEOUT='300',LOCALHOST_AGENT_SAMPLE_TIMEOUT_S='300',
  PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',XTUNER_ACTIVATION_OFFLOAD='1',
  TORCH_ALLOW_TF32_CUBLAS_OVERRIDE='1',CUBLAS_WORKSPACE_CONFIG=':16:8',PERMUTE_COMPUTE_DTYPE='fp32')
 crg=Path(env['CRG_PATH'])
 env['PYTHONPATH']=':'.join(map(str,[ROOT/'gpu_jobs',crg,crg/'src',env['XTUNER_PATH'],env['LAGENT_PATH'],env['LMDEPLOY_PATH']]))
 hosts=['127.0.0.1','localhost',socket.gethostname()]
 try:hosts.append(socket.gethostbyname(socket.gethostname()))
 except socket.gaierror:pass
 env['NO_PROXY']=env['no_proxy']=','.join(hosts)
 for key in ['TRITON_CACHE_DIR','TORCH_EXTENSIONS_DIR','TORCHINDUCTOR_CACHE_DIR','CUDA_CACHE_PATH','DG_JIT_CACHE_DIR']:
  env[key]=str(out.parent/'cache'/key.lower())
 for key in ['FROZEN_MTP_SOURCE','FROZEN_MTP_MANIFEST','FROZEN_MTP_SHA256','XT_FROZEN_MTP_SOURCE','CYBERGYM_MTP_ROUTE']:
  env.pop(key,None)
 return env
