"""One owned RJob allocation; real replay/HTTP then actual full cold restore."""
import os,subprocess,json,hashlib,signal,time,shutil
from pathlib import Path
from runtime import ROOT,PYTHON,environment
out=Path(os.environ['SPLIT_RJOB_WORK_DIR']);out.mkdir(parents=True,exist_ok=True)
manifest=json.loads((ROOT/'GPU_FROZEN_INPUTS.json').read_text())
for name,digest in manifest['files'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
for name,sha in manifest['repositories'].items():
 assert subprocess.check_output(['git','-C',name,'rev-parse','HEAD'],text=True).strip()==sha
 assert not subprocess.check_output(['git','-C',name,'status','--porcelain'],text=True).strip(),name
import torch
assert torch.cuda.device_count()==8,torch.cuda.device_count()
(out/'GPU.csv').write_text(subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.total','--format=csv,noheader'],text=True))
assert all('H200' in line for line in (out/'GPU.csv').read_text().splitlines()),'Pinned model/topology smoke requires H200'
result={'passed':False,'natural_online_rollout':False,'scale64':False}
for phase in ['replay','restore']:
 stage=out/phase;stage.mkdir(exist_ok=False);env=environment(stage)
 command=[str(PYTHON),str(ROOT/'gpu_jobs/driver.py'),'--phase',phase]
 if phase=='restore':
  expect=out/'replay/RESTORE_EXPECTATIONS.json';env['RESTORE_EXPECTATIONS']=str(expect)
  command+=['--checkpoint',json.loads(expect.read_text())['checkpoint']]
 (stage/'COMMAND.json').write_text(json.dumps(command));started=time.monotonic()
 with (stage/'console.log').open('w') as log:
  child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:code=child.wait(timeout=1800 if phase=='replay' else 1200)
  except subprocess.TimeoutExpired:
   os.killpg(child.pid,signal.SIGINT)
   try:child.wait(timeout=120)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
   code=124
 (stage/'EXIT.json').write_text(json.dumps({'exit':code,'seconds':time.monotonic()-started}))
 if code:raise SystemExit(code)
 row=json.loads((stage/'RESULT.json').read_text());assert row['passed'] and not row['cleanup_errors']
 result[phase]=row
if result.get('restore',{}).get('full_trainer_cold_restore'):
 checkpoint=Path(result['replay']['checkpoint']).resolve()
 assert checkpoint.is_relative_to((out/'replay').resolve()) and checkpoint.name=='ckpt-step-2'
 n=sum(p.stat().st_size for p in checkpoint.rglob('*') if p.is_file())
 metadata=out/'SAVED_CHECKPOINT_METADATA';metadata.mkdir()
 for path in checkpoint.rglob('*'):
  if path.is_file() and (path.name=='.metadata' or path.name=='train_state.json'):
   target=metadata/path.relative_to(checkpoint);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
 shutil.rmtree(checkpoint)
 (out/'TEMPORARY_WEIGHT_CLEANUP.json').write_text(json.dumps({'path':str(checkpoint),'bytes_released':n,'reason':'cold restore verified; no later model/eval reference','exists_after':checkpoint.exists()}))
result['passed']=True;(out/'ACCEPTANCE.json').write_text(json.dumps(result,indent=2))
print('XT2257_SPLIT_QUICK_GPU_PASSED',flush=True)
