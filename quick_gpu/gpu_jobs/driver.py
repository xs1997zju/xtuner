"""Replay and cold-restore probe; not a natural CyberGym online experiment."""
import argparse,asyncio,json,os,time,traceback,hashlib
from pathlib import Path
import ray
from runtime import ROOT,PT
OUT=Path(os.environ['WORK_DIR']);OUT.mkdir(parents=True,exist_ok=True)
def save(name,value):(OUT/name).write_text(json.dumps(value,indent=2,default=str)+'\n')
def event(kind,**kw):
 row=dict(kind=kind,at=time.time(),**kw)
 with (OUT/'EVENTS.jsonl').open('a') as f:f.write(json.dumps(row,default=str)+'\n')
 print(kind,kw,flush=True)
def call(manager,coro):return asyncio.run(manager._owner.call(coro))

def main():
 p=argparse.ArgumentParser();p.add_argument('--phase',choices=['replay','restore','export'],required=True);p.add_argument('--checkpoint');a=p.parse_args()
 import xtuner.v1.rl.rollout.controller as controllers
 from controlled_replay import ObservedController
 controllers.RolloutController=ObservedController  # Read-only discovery method, original behavior inherited.
 import xtuner.v1.rl.trainer.worker as workers
 from training_observer import ObservedTrainingWorker
 workers.TrainingWorker=ObservedTrainingWorker
 from config import build
 cfg=build(a.checkpoint)
 save('CONFIG.json',json.loads(cfg.model_dump_json(serialize_as_any=True,fallback=str)))
 if a.phase=='export':
  from importlib.metadata import version
  import xtuner,transformers
  save('IMPORTS.json',dict(xtuner=xtuner.__file__,transformers=transformers.__file__,python=os.sys.executable,
       torch=version('torch'),transformers_version=version('transformers'),glm5_next=__import__('importlib').util.find_spec('transformers.models.glm5_next').origin))
  print('CPU_CONFIG_EXPORTED',flush=True);return
 trainer=None;loop=None;result=dict(passed=False,natural_online_rollout=False,real_cybergym_sandbox=False,controlled_tools=True)
 ray.init(address='local',num_gpus=8,num_cpus=64,include_dashboard=False,namespace=os.environ['XTUNER_RUN_ID'],_temp_dir=f'/tmp/xtsplit-{os.getpid()}',object_store_memory=8*1024**3)
 try:
  event('trainer_build_begin',phase=a.phase);trainer=cfg.build();event('trainer_built',step=trainer._cur_step)
  manager=trainer.agent_loop_manager;task=manager.task_runners[0];loop=task.agent_loop
  endpoints=call(manager,loop.initialize_fixture());save('ENDPOINTS.json',endpoints)
  def snapshots():return ray.get([w.validation_snapshot.remote() for w in trainer.train_controller.workers],timeout=300)
  if a.phase=='restore':
   original=json.loads(Path(os.environ['RESTORE_EXPECTATIONS']).read_text())
   assert trainer._cur_step==2
   actual=snapshots();save('RESTORED_PARAMETERS.json',actual)
   for before,after in zip(original['parameters'],actual):
    assert before['rank']==after['rank']
    assert before['mtp']==after['mtp'],('MTP restore mismatch',after['rank'])
    assert before['main']==after['main'],('Main probe restore mismatch',after['rank'])
    assert before['shared_io']==after['shared_io'],('Shared IO restore mismatch',after['rank'])
    assert after['optimizer_step_min']==before['optimizer_step_min']==2
    assert after['optimizer_step_max']==before['optimizer_step_max']==2
   state=dict(consumed=task.sampler._consumed_samples,epoch=task.sampler.cur_epoch,replay_groups=len(manager.replay_buffer))
   assert state==original['manager_state'],(state,original['manager_state'])
   _,response=call(manager,loop.request('split-reference-restored',[{'role':'user','content':'Say hello.'}]))
   assert response['model_step']==2
   result.update(passed=True,full_trainer_cold_restore=True,optimizer_step_verified=2,manager_state=state,response=response,post_restore_optimizer_updates=0,live_agents_restored=False)
  else:
   trainer._debug_train_files={1:PT};loop.loader=lambda:trainer._load_debug_rollout_batch(1)
   _,before=call(manager,loop.request('split-reference-before',[{'role':'user','content':'Say hello.'}]))
   assert before['model_step']==0
   for step in [1,2]:
    if step==2:manager._owner.loop.call_soon_threadsafe(loop.tool.set)
    batch=asyncio.run(manager.produce_batch(1,step,model_step=step-1,quiesce=step==2))
    pending=task.produce_strategy.pending_task_count();event('produce_return',step=step,pending=pending)
    assert len(batch.rollout_states)==1
    if step==1:assert pending>=1 and not loop.tool.is_set(),'Live controlled tool task was lost'
    if step==2:
     assert pending==0 and loop.tail_finished.is_set()
     assert [x['model_step'] for x in loop.turns]==[0,1]
     assert loop.turns[0]['binding']==loop.turns[1]['binding']
    timing={};event('train_begin',step=step)
    info=trainer._train_one_batch(batch.rollout_states,step,timing,offload_rollout_before_train=True,
        raw_rewards_sum=batch.raw_rewards_sum,raw_rewards_count=batch.raw_rewards_count,produce_result=batch)
    save(f'TRAIN_{step}.json',dict(timing=timing,data_info=info['data_info']))
    trainer._cur_step=step
    trainer.train_controller.offload(target='optimizer')
    trainer._sync_weights_from_train_workers();event('native_ipc_sync_returned',step=step)
    if step==2:
     asyncio.run(manager.publish_initial_version(step))  # Native policy.resume under the current barrier.
   actual=snapshots();save('POST_TRAIN_PARAMETERS.json',actual)
   assert all(x['optimizer_step_min']==2 and x['optimizer_step_max']==2 for x in actual)
   _,after=call(manager,loop.request('split-reference-after',[{'role':'user','content':'Say hello.'}]))
   assert after['model_step']==2
   assert before['logprobs']!=after['logprobs'],'No observed backend probability change'
   # Ensure the replay checkpoint is non-empty, without altering recorded payload.
   if len(manager.replay_buffer)==0:
    call(manager,manager.replay_buffer.put(trainer._load_debug_rollout_batch(1)[0],task.task_name))
   state=dict(consumed=task.sampler._consumed_samples,epoch=task.sampler.cur_epoch,replay_groups=len(manager.replay_buffer))
   assert state['consumed']>=2 and state['replay_groups']>=1
   asyncio.run(trainer._maybe_save_checkpoint(2));event('full_checkpoint_saved')
   checkpoint=trainer.exp_dir/trainer._CHECKPOINT_DIR/'ckpt-step-2'
   files=[dict(path=str(p.relative_to(checkpoint)),bytes=p.stat().st_size) for p in checkpoint.rglob('*') if p.is_file()]
   assert any(x['path'].endswith('.metadata') for x in files)
   save('CHECKPOINT_MANIFEST.json',dict(path=str(checkpoint),files=files,total_bytes=sum(x['bytes'] for x in files)))
   save('RESTORE_EXPECTATIONS.json',dict(parameters=actual,manager_state=state,checkpoint=str(checkpoint)))
   for rank in range(8):
    rows=[json.loads(s) for s in (OUT/'training_parameters'/f'rank-{rank}.jsonl').read_text().splitlines()]
    pairs=[(rows[i],rows[i+1]) for i in range(len(rows)-1) if rows[i]['phase']=='before_actor_fit' and rows[i+1]['phase']=='after_actor_fit']
    assert len(pairs)==2
    for old,new in pairs:
     assert any(old['main'][n]['sha256']!=v['sha256'] for n,v in new['main'].items()),('No main update',rank)
     assert old['mtp']==new['mtp'],('Frozen MTP changed',rank)
   result.update(passed=True,actual_optimizer_updates=2,ranks=8,ipc_sync_executed=True,backend_logprobs_changed=True,
       same_live_controlled_task_crossed_update=True,request_versions=[0,1,2],same_session_affinity=True,
       checkpoint=str(checkpoint),full_checkpoint_saved=True,full_restore_verified=False,manager_state=state,turns=loop.turns)
 finally:
  cleanup=[]
  if trainer is not None:
   manager=getattr(trainer,'agent_loop_manager',None)
   if manager is not None:
    if loop is not None and hasattr(loop,'router'):
     try:call(manager,loop.finish_fixture())
     except Exception as e:cleanup.append('fixture:'+repr(e))
    try:asyncio.run(manager.close())
    except Exception as e:cleanup.append('manager:'+repr(e))
   try:trainer.rl_health_manager.stop()
   except Exception as e:cleanup.append('health:'+repr(e))
   if hasattr(trainer,'rollout_controller'):
    try:ray.get(trainer.rollout_controller.shutdown.remote(),timeout=120)
    except Exception as e:cleanup.append('rollout:'+repr(e))
   try:trainer._exp_tracker.close()
   except Exception as e:cleanup.append('tracker:'+repr(e))
  ray.shutdown();result['cleanup_errors']=cleanup
  if cleanup:result['passed']=False
  save('RESULT.json',result)
  if cleanup:raise RuntimeError('Own resources cleanup failed: '+repr(cleanup))
if __name__=='__main__':
 try:main()
 except BaseException:(OUT/'ERROR.txt').write_text(traceback.format_exc());raise
