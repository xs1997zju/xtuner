"""Real native manager/HTTP/R3, controlled tool wait, unmodified recorded training payload."""
import asyncio,json,os,copy
from pathlib import Path
import aiohttp,ray
from xtuner.v1.rl.agent_loop import AgentLoopConfig
from xtuner.v1.rl.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
from xtuner.v1.rl.rollout.affinity_router import SessionAffinityRouter
from xtuner.v1.rl.rollout.trace_store import get_store
from xtuner.v1.rl.rollout.controller import RolloutController
from runtime import ROOT

class ObservedController(RolloutController):
    def validation_endpoints(self):
        return [w.session_url for w in self.registry.active_entrypoints()]

class ReplayLoop(SingleTurnAgentLoop):
    async def initialize_fixture(self):
        self.router=SessionAffinityRouter()
        urls=await self.rollout_ctl.validation_endpoints.remote()
        self.router.replace_backends(urls)
        self.endpoint=await self.router.start('127.0.0.1')
        self.tail_started=asyncio.Event();self.tool=asyncio.Event();self.tail_finished=asyncio.Event()
        self.session='split-tail-'+os.environ['XTUNER_RUN_ID']
        self.messages=[];self.turn_ids=[];self.turns=[]
        return dict(backends=urls,affinity_endpoint=self.endpoint)

    async def request(self, session, messages):
        body=dict(model=os.environ['XTUNER_RUN_ID'],session_id=session,messages=messages,max_tokens=16,
                  temperature=0,stream=False,return_token_ids=True,return_routed_experts=True)
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=240),trust_env=False) as client:
            async with client.post(self.endpoint+'/v1/chat/completions',json=body) as response:
                payload=await response.json();assert response.status==200,(response.status,str(payload)[:500])
        store=get_store();values={}
        for key in await store.keys.remote(session):
            _,nodes=await store.search.remote(session,key,True)
            for node in nodes:
                v=node.value
                if v.request_id and v.labels and any(x!=-100 for x in v.labels):values[v.request_id]=v
        req=payload['id'];assert req in values,(req,list(values))
        v=values[req];assert v.model_step is not None and v.expert_key is not None
        if isinstance(v.expert_key,ray.ObjectRef):
            expert=await v.expert_key
            assert len(expert)==len(v.token_ids),(len(expert),len(v.token_ids))
        row=dict(session=session,request_id=req,model_step=v.model_step,logprobs=v.logprobs,
                 token_ids=v.token_ids,routed_experts_present=True,binding=self.router.bindings[session],
                 retained_request_ids=list(values))
        with (Path(os.environ['WORK_DIR'])/'REQUESTS.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
        return payload,row

    async def model_turn(self):
        self.messages.append({'role':'user','content':'Say hello.' if not self.messages else 'Now say goodbye.'})
        result,row=await self.request(self.session,self.messages)
        self.messages.append(result['choices'][0]['message']);self.turn_ids.append(row['request_id'])
        assert set(self.turn_ids)<=set(row['retained_request_ids'])
        self.turns.append(row)

    async def generate_group(self, group, **kwargs):
        identity=group[0].group_id
        # Real sampler chooses these groups. Tool actions and scheduling are
        # controlled; the GPU training payload remains the recorded PT group.
        if identity==0:await self.tail_started.wait()
        if identity==1:
            await self.model_turn();self.tail_started.set()
            await self.tool.wait();await self.model_turn();self.tail_finished.set()
        train_batch=self.loader()
        for sample in train_batch[0]:sample.extra_fields['_replay_control_group']=identity
        return train_batch[0]

    async def finish_fixture(self):
        self.tool.set()
        await self.router.stop()
        await get_store().release_sessions.remote([self.session,'split-reference-before','split-reference-after','split-reference-restored'])

class ReplayLoopConfig(AgentLoopConfig):
    require_turn_provenance: bool=True
    def build_local(self,rollout_controller,judger=None,logger=None):
        return ReplayLoop(rollout_ctl=rollout_controller,hf_checkpoint=self.hf_checkpoint,
            sample_params=self.sample_params,judger=judger,logger=logger)
