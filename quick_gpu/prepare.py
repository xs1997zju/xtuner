"""Freeze explicit caller inputs; this does not run training or migrate data."""
import os,sys,json,hashlib,subprocess
from pathlib import Path
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'gpu_jobs'))
from runtime import PT
assert PT.is_file(), PT
files={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'gpu_jobs').iterdir() if p.is_file()}
files[str(PT)]=hashlib.sha256(PT.read_bytes()).hexdigest()
repos={}
for name in ['XTUNER_PATH','LMDEPLOY_PATH','LAGENT_PATH','CRG_PATH']:
 p=Path(os.environ[name]).resolve()
 assert not subprocess.check_output(['git','-C',str(p),'status','--porcelain'],text=True).strip(),p
 repos[str(p)]=subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()
(root/'GPU_FROZEN_INPUTS.json').write_text(json.dumps(dict(files=files,repositories=repos),indent=2)+'\n')
print('Input identities recorded. Check model/PT/native-loader/packing compatibility before --run.')
