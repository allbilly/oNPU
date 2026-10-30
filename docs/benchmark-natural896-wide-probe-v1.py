"""Unexecuted raw INT8 K=896 full-model geometry qualification probe.

Run only under an explicit hardware lease. Natural K=896 commands retain the
stock next-512 CBUF recipe and patch only K-address fields, as in
extra/rockchip/probe_int8_natural.py. The control arm uses stock padded K=1024
with zero-extended operands. Every output plane is poisoned before submission;
readback and exact int32 comparison happen after the timed submit.
"""
import argparse, array, ctypes, hashlib, json, platform, time
from pathlib import Path
from pathlib import Path as FilePath
import numpy as np
from tinygrad import Device
from tinygrad.helpers import ceildiv, round_up, to_mv
from tinygrad.runtime.autogen import rockchip as rk
from tinygrad.runtime.ops_rockchip import _cmd
from tinygrad.runtime.support.rockchip_int8 import int8_body

K = 896
SEED = 20261028
POISON = np.int32(0x5a17c0de)
REQUIRED = ((128, 9728, "qwen2_joint_ffn_prefill"), (1, 24576, "qwen2_vocab_tile_decode"))
OPTIONAL = ((128, 896, "qwen2_square_prefill"), (128, 1152, "qwen2_qkv_prefill"),
            (1, 9728, "qwen2_joint_ffn_decode"), (1, 896, "qwen2_square_decode"))

def natural_body(m, n, k, lhs, rhs, dst):
  body = int8_body(m, n, round_up(k, 512), lhs, rhs, dst, n)
  patch = {0x1024: ((k-1)<<16)|k, 0x1030: k*n, 0x1034: k,
           0x1044: ceildiv(k,64), 0x107c: k//16, 0x1088: k}
  return tuple(_cmd(0x201, word&0xffff, patch[word&0xffff])
               if word>>48 == 0x201 and word&0xffff in patch else word for word in body)

def split(n):
  width = round_up(ceildiv(n,3),32)
  parts = [(start,min(width,n-start)) for start in range(0,n,width)]
  if len(parts) != 3: raise ValueError(f"3-core split invalid for N={n}: {parts}")
  return parts

def pack_weights(weights, physical_k):
  n, k = weights.shape
  padded = np.zeros((n,physical_k),dtype=np.int8)
  padded[:,:k] = weights
  return padded.reshape(n//32,32,physical_k//32,32).transpose(0,2,1,3).copy()

def run_arm_once(dev, alloc, m, n, arm, x, w, expected, case, report):
  physical_k = K if arm == "natural_k896" else 1024
  parts = split(n)
  # The existing raw probe's three-core private output planes and packed-weight
  # layout are used unchanged. The 5 subcore entries match its submit recipe.
  a,b,c = alloc(m*physical_k),alloc(n*physical_k),alloc(m*n*4)
  command,task = alloc(8192),alloc(4096,rk.RKNPU_MEM_KERNEL_MAPPING)
  px = np.zeros((m,physical_k),dtype=np.int8); px[:,:K] = x
  upload(dev,a,px); upload(dev,b,pack_weights(w,physical_k))
  tail = (_cmd(1,0),_cmd(rk.TARGET_PC_REG,rk.REG_PC_REGISTER_AMOUNTS),_cmd(rk.TARGET_VERSION,0),
          _cmd(rk.TARGET_PC,rk.REG_PC_OPERATION_ENABLE,0xd))
  offset, recipe, descriptors = 0, [], []
  for i,(start,count) in enumerate(parts):
    body = (natural_body(m,count,K,a.dma,b.dma+start*physical_k,c.dma+start*m*4) if arm == "natural_k896" else
            int8_body(m,count,physical_k,a.dma,b.dma+start*physical_k,c.dma+start*m*4,count))
    if arm == "natural_k896":
      assert body == int8_body(m,count,K,a.dma,b.dma+start*K,c.dma+start*m*4,count)
    words = array.array('Q',body+tail)
    regs={(word>>48,word&0xffff):(word>>16)&0xffffffff for word in body}
    ctypes.memmove(command.addr+offset*8,words.buffer_info()[0],len(words)*8)
    desc = rk.struct_rknpu_task(0,0,0xd,0x300,0x1ffff,0,len(body),0,command.dma+offset*8)
    ctypes.memmove(task.addr+i*ctypes.sizeof(desc),ctypes.addressof(desc),ctypes.sizeof(desc))
    descriptors.append({'index':i,'output_start':start,'output_width':count,'body_words':len(body),
                        'task_descriptor':{name:getattr(desc,name) for name,_ in desc._fields_}})
    selected={(0x201,r):regs.get((0x201,r)) for r in (0x1020,0x1024,0x1030,0x1034,0x1040,0x1044,0x1070,0x107c,0x1088,0x1110)}
    recipe.append({'output_start':start,'output_width':count,
                   'register_config':{f'0x{target:x}:0x{reg:04x}':value for (target,reg),value in selected.items()},
                   'k_registers':{'0x1024':((K-1)<<16)|K if arm == 'natural_k896' else ((physical_k-1)<<16)|physical_k,
                                  '0x1030':physical_k*count,'0x1034':physical_k,
                                  '0x1044':ceildiv(physical_k,64),'0x107c':physical_k//16,'0x1088':physical_k}})
    offset += round_up(len(words),2)
  for buf in (command,task): dev.sync(buf,rk.RKNPU_MEM_SYNC_TO_DEVICE)
  poison=np.full(m*n,POISON,dtype=np.int32)
  ctypes.memmove(c.addr,poison.ctypes.data,poison.nbytes)
  dev.sync(c,rk.RKNPU_MEM_SYNC_TO_DEVICE)
  subcores=(rk.struct_rknpu_subcore_task*5)(*(rk.struct_rknpu_subcore_task(*v) for v in ((0,0),(0,0),(0,1),(1,1),(2,1))))
  started=time.perf_counter()
  try:
    rk.DRM_IOCTL_RKNPU_SUBMIT(dev.fd,flags=rk.RKNPU_JOB_PC|rk.RKNPU_JOB_BLOCK|rk.RKNPU_JOB_PINGPONG,
      timeout=6000,task_start=0,task_number=3,task_counter=0,priority=0,task_obj_addr=task.meta.obj_addr,
      iommu_domain_id=0,task_base_addr=0,hw_elapse_time=0,core_mask=7,fence_fd=-1,subcore_task=subcores)
  except OSError:
    dev.poisoned=True
    raise
  elapsed=(time.perf_counter()-started)*1000
  dev.sync(c,rk.RKNPU_MEM_SYNC_FROM_DEVICE)
  raw=np.frombuffer(to_mv(c.addr,m*n*4),np.int32).copy()
  actual=np.concatenate([raw[start*m:(start+count)*m].reshape(m,count) for start,count in parts],axis=1)
  error=int(np.max(np.abs(actual.astype(np.int64)-expected.astype(np.int64))))
  row={'arm':arm,'input_case':case,'shape':[m,K,n],'physical_k':physical_k,
       'split':parts,'submit_ms':elapsed,'output_values':int(actual.size),'max_abs_error':error,
       'poison_remaining':int(np.count_nonzero(actual==POISON)),
       'input_sha256':hashlib.sha256(x.tobytes()).hexdigest(),
       'weight_sha256':hashlib.sha256(w.tobytes()).hexdigest(),
       'actual_sha256':hashlib.sha256(actual.tobytes()).hexdigest(),
       'expected_sha256':hashlib.sha256(expected.tobytes()).hexdigest()}
  print(json.dumps(row),flush=True)
  details=report['arm_details'].setdefault(f'{m}x{n}:{arm}',{'shape':[m,K,n],'arm':arm,
    'register_recipe':recipe,'task_descriptors':descriptors,'results':[]})
  details['results'].append(row)
  np.testing.assert_array_equal(actual,expected)
  if row['poison_remaining']: raise AssertionError('poison remained in output')

def upload(dev, buf, values):
  ctypes.memmove(buf.addr,values.ctypes.data,values.nbytes)
  dev.sync(buf,rk.RKNPU_MEM_SYNC_TO_DEVICE)

def make_inputs(m,n,kind,rng):
  if kind.startswith('random'):
    return rng.integers(-128,128,(m,K),dtype=np.int8),rng.integers(-128,128,(n,K),dtype=np.int8)
  if kind == 'alternating_extrema':
    x=np.empty((m,K),dtype=np.int8); w=np.empty((n,K),dtype=np.int8)
    x[:,::2],x[:,1::2]=-128,127; w[:,::2],w[:,1::2]=127,-128
    return x,w
  if kind == 'all_extrema':
    return np.full((m,K),-128,dtype=np.int8),np.full((n,K),-128,dtype=np.int8)
  raise ValueError(kind)

def main():
  p=argparse.ArgumentParser(description=__doc__)
  p.add_argument('--report',type=Path,required=True)
  args=p.parse_args()
  dev=Device['ROCKCHIP']; owned=[]
  def alloc(size,flags=0):
    b=dev.allocate(size,flags); owned.append(b); return b
  report={'schema':'natural896-wide-qualification-report-v1','complete':False,'numeric_pass':False,'cases':[],
          'quality_qualified':False,'model_speed_qualified':False,'goal_complete':False,
          'environment':{'platform':platform.platform(),'machine':platform.machine()},'arm_details':{},
          'scope':'raw direct low-level INT8 command probe; no model execution'}
  try:
    compatible='/proc/device-tree/compatible'
    report['environment'].update({'device':str(dev),'device_type':type(dev).__name__,
      'device_tree_compatible':FilePath(compatible).read_bytes().replace(b'\0',b',').decode(errors='replace') if FilePath(compatible).exists() else None,
      'device_attrs':{k:str(getattr(dev,k)) for k in ('version','core_mask','npu_version','driver_version') if hasattr(dev,k)}})
    rng=np.random.default_rng(SEED)
    for m,n,label in REQUIRED+OPTIONAL:
      parts=split(n)
      if max(count for _,count in parts)>(8192 if m==1 else 4096):raise ValueError('Per-core N exceeds qualified guard')
      if n%32: raise ValueError(f'current INT8 body requires N divisible by 32: {n}')
      shape_case={'shape':[m,K,n],'label':label,'status':'planned'}
      report['cases'].append(shape_case)
      # Workload sets are built separately per shape to keep reference dimensions exact.
      for kind in ('random_a','random_b','alternating_extrema','all_extrema'):
        x,w=make_inputs(m,n,kind,rng)
        expected=x.astype(np.int32) @ w.astype(np.int32).T
        for arm in ('natural_k896','padded_k1024'):
          run_arm_once(dev,alloc,m,n,arm,x,w,expected,kind,report)
      shape_case['status']='passed'
    rows=[row for detail in report['arm_details'].values() for row in detail['results']]
    assert len(rows)==len(REQUIRED+OPTIONAL)*4*2==48
    assert all(row['max_abs_error']==0 and row['actual_sha256']==row['expected_sha256'] for row in rows)
    report.update(complete=True,numeric_pass=True,independent_integer_checks=len(rows))
  except BaseException as exc:
    report['error']=f'{type(exc).__name__}: {exc}'
    raise
  finally:
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2)+'\n')
    if not dev.poisoned: dev.reset()
    for b in reversed(owned): dev.free(b)
    dev.finalize()

if __name__=='__main__': main()
