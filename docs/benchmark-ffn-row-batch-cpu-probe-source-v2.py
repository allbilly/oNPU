"""Qualify and time CPU raw-FFN row batching without opening an accelerator."""
import argparse,functools,gc,hashlib,importlib.util,json,os,statistics,time
from pathlib import Path
import numpy as np
from tinygrad import Device,Tensor,TinyJit,UOp
from tinygrad.helpers import ceildiv,round_up
from tinygrad.llm.kernels import rockchip_gelu_ffn as base
from tinygrad.llm.kernels.rockchip_int8 import quantize
from tinygrad.llm.kernels.rockchip_pool import AttentionPool
from tinygrad.uop.ops import Ops

M=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('direct_row_batch_proposal',M/'proposed_ffn_row_batch_direct_v1.py')
prototype=importlib.util.module_from_spec(spec);spec.loader.exec_module(prototype)
ORIGINAL_PROGRAM=base._program
CURRENT_BATCH=1
GENERATED={};DESCRIPTORS={}

def sha_bytes(value):return hashlib.sha256(value).hexdigest()
def source_of(program):return next(node.arg for node in program.src if node.op is Ops.SOURCE)
@functools.cache
def variant_program(batch,*args):
  original=ORIGINAL_PROGRAM(*args)
  rows,hidden,depths,tiles,static,activation=args[6:12]
  assert static is True and activation=='silu'
  result=original if batch==1 else prototype._program(*args,row_batch=batch)
  if rows==1:assert result is original
  source=source_of(result);digest=sha_bytes(source.encode());GENERATED[digest]=source
  DESCRIPTORS[(rows,hidden,tuple(depths),tuple(tiles),batch)]={
   'rows':rows,'hidden':hidden,'depths':list(depths),'tiles':[list(t) for t in tiles],
   'batch':batch,'source_sha256':digest,'single_row_original_program':rows==1 and result is original}
  return result
def selected_program(*args):return variant_program(CURRENT_BATCH,*args)
base._program=selected_program

def clocks():
  result={}
  for path in ('/sys/devices/system/cpu/cpufreq/policy4/scaling_cur_freq',
               '/sys/devices/system/cpu/cpufreq/policy6/scaling_cur_freq','/sys/class/devfreq/dmc/cur_freq'):
    try:result[path]=int(Path(path).read_text())
    except OSError:pass
  return result

def inputs(rows,hidden,tile,seed,active,nonfinite=False):
  rng=np.random.default_rng(seed);raw=[];sa=[];tiles=[]
  sw_values=rng.uniform(.001,.03,hidden*2).astype(np.float32)
  sw_values[0]=0.;sw_values[hidden+1]=-0.
  if nonfinite:sw_values[17]=np.inf
  sw=Tensor(sw_values,device='CPU').realize()
  for first in range(0,rows,tile):
    count=min(tile,rows-first);scale=rng.uniform(.001,.01,count).astype(np.float32)
    scale[max(0,active-first):]=np.nan
    sa.append(Tensor(scale,device='CPU').realize());scale_index=len(sa)-1
    limit=24576 if count==1 else 12288
    for column in range(0,hidden*2,limit):
      width=min(limit,hidden*2-column)
      values=rng.integers(-100000,100000,(count,width),dtype=np.int32)
      if first==0:
        values[0,:]=0
        if count>1:values[1,:32]=np.where(np.arange(32)%2,1000000,-1000000)
      core=round_up(ceildiv(width,3),32);parts=[(start,min(core,width-start)) for start in range(0,width,core)]
      if len(parts)!=3:parts=[(0,width)]
      packed=np.concatenate([values[:,start:start+size].ravel() for start,size in parts])
      raw.append(Tensor(packed,device='CPU').realize())
      tiles.append(base.ProjectionTile(first,count,column,width,scale_index))
  return raw,sa,sw,tiles

def dense_reference(raw,sa,sw,tiles,rows,hidden,depths,active):
  projected=np.empty((rows,hidden*2),np.float32);weights=sw.numpy()
  for tensor,tile in zip(raw,tiles):
    flat=tensor.numpy();core=round_up(ceildiv(tile.columns,3),32)
    parts=[(start,min(core,tile.columns-start)) for start in range(0,tile.columns,core)]
    if len(parts)!=3:parts=[(0,tile.columns)]
    for start,width in parts:
      values=flat[start*tile.rows:(start+width)*tile.rows].reshape(tile.rows,width).astype(np.float32)
      with np.errstate(invalid='ignore',over='ignore'):
        projected[tile.first_row:tile.first_row+tile.rows,tile.first_column+start:tile.first_column+start+width]=(
          values*sa[tile.scale_index].numpy()[:,None])*weights[tile.first_column+start:tile.first_column+start+width]
  joint=Tensor(projected,device='CPU').realize();gate=joint[:,:hidden]
  sigmoid=gate.sigmoid().contiguous().realize()
  activation=((gate*sigmoid)*joint[:,hidden:]).contiguous().realize()
  activation=activation[:active].pad(((0,rows-active),(0,0))).contiguous().realize()
  packets=[];scales=[];column=0
  for depth,(_,padded,row_tile,_) in zip(depths,base.payload_layout(rows,depths)):
    for first in range(0,rows,row_tile):
      packet,scale=quantize(activation[first:first+row_tile,column:column+depth].contiguous(),padded)
      packet.realize(scale);packets.append(packet.numpy().copy());scales.append(scale.numpy().view(np.uint32).copy())
    column+=depth
  return np.concatenate(packets),np.concatenate(scales)

def make_runs(pool,rows,hidden,tiles,n_raw,n_sa):
  depths=tuple(min(4096,hidden-start) for start in range(0,hidden,4096))
  def compute(*args):
    packet,scales=base.pack_swiglu_projection(list(args[:n_raw]),list(args[n_raw:n_raw+n_sa]),args[-2],
      tiles,rows,hidden,depths,args[-1],pool=pool)
    packet.realize(scales)
    return packet,scales
  return {batch:TinyJit(compute) for batch in (1,2,4)},depths

def read_outputs(outputs):return outputs[0].numpy().copy(),outputs[1].numpy().view(np.uint32).copy()
def output_hash(outputs):return sha_bytes(outputs[0].tobytes()+outputs[1].tobytes())
def poison(outputs):
  np.frombuffer(outputs[0].data(),np.uint8).fill(0xa5)
  np.frombuffer(outputs[1].data(),np.uint32).fill(0x7fc01234)

def call(pool,run,batch,args,previous=None,timed=False):
  global CURRENT_BATCH
  CURRENT_BATCH=batch
  if previous is not None:poison(previous)
  count=pool.lib.pool_calls();affinity=os.sched_getaffinity(0)
  with pool.prefill():
    began=time.perf_counter_ns()
    outputs=run(*args)
    elapsed=time.perf_counter_ns()-began
  assert os.sched_getaffinity(0)==affinity and pool.lib.pool_parked()
  groups=(args[-2].shape[0]//2+4095)//4096
  rows=outputs[1].numel()//groups
  assert pool.lib.pool_calls()-count==(1 if rows>=16 else 0)
  return outputs,elapsed/1e6 if timed else None

def captured_source(run,batch,rows):
  assert run.captured is not None
  programs=[call.body for call in run.captured.linear.src if call.body.op is Ops.PROGRAM
    and call.body.src[0].arg.function_name=='rockchip_int8_swiglu_pack']
  assert len(programs)==1
  source=source_of(programs[0])
  assert ('float batch_values' in source)==(batch>1 and rows>=batch)
  digest=sha_bytes(source.encode());assert digest in GENERATED
  return digest

def main():
  parser=argparse.ArgumentParser();parser.add_argument('--report',type=Path,required=True);args=parser.parse_args()
  assert not args.report.exists()
  report={'schema':'ffn-row-batch-cpu-probe-v2','complete':False,'numeric_pass':False,'accelerators_opened':False,
    'goal_complete':False,'quality_qualified':False,'model_speed_qualified':False,'qualification':[],'timings':[],
    'captured_programs':[],
    'scope':'CPU-only raw FFN packer prototype, not full-model performance. Direct production-generator proposal; both batches retain baseline arithmetic.'}
  pool=AttentionPool()
  try:
    cases=[(1,4864,1,[1]),(17,4608,17,[1,16,17]),(33,4928,33,[1,17,33]),
      (126,4864,64,[1,17,63,65,125,126]),(128,4864,128,[1,17,65,127,128]),
      (128,4608,128,[1,17,65,127,128]),(128,5632,64,[17,65,128]),(128,8960,64,[17,65,128])]
    for case_index,(rows,hidden,tile,actives) in enumerate(cases):
      variable=UOp.variable('ffn_probe_active',1,rows) if rows>1 else None
      first=inputs(rows,hidden,tile,4000+case_index,rows)
      runs,depths=make_runs(pool,rows,hidden,first[3],len(first[0]),len(first[1]));last={}
      warm_arguments=(*first[0],*first[1],first[2],variable.bind(rows) if variable is not None else 1)
      for batch in (1,2,4):
        for _ in range(4):last[batch],_=call(pool,runs[batch],batch,warm_arguments,last.get(batch))
        report['captured_programs'].append({'rows':rows,'hidden':hidden,'input_tile':tile,'batch':batch,
          'source_sha256':captured_source(runs[batch],batch,rows)})
      for seed in (0,1):
        for active in actives:
          raw,sa,sw,tiles=inputs(rows,hidden,tile,5000+case_index*10+seed,active,nonfinite=bool(seed))
          arguments=(*raw,*sa,sw,variable.bind(active) if variable is not None else 1)
          expected=None
          for batch in (1,2,4):
            last[batch],_=call(pool,runs[batch],batch,arguments,last.get(batch))
            actual=read_outputs(last[batch])
            if expected is None:expected=actual
            else:
              np.testing.assert_array_equal(actual[0],expected[0]);np.testing.assert_array_equal(actual[1],expected[1])
            report['qualification'].append({'rows':rows,'hidden':hidden,'input_tile':tile,'active':active,
              'seed':seed,'batch':batch,'payload_sha256':sha_bytes(actual[0].tobytes()),
              'scale_bits_sha256':sha_bytes(actual[1].tobytes()),'exact':True,'poisoned':True})
          if active==rows and seed==0 and rows==128 and hidden in (4608,4864):
            reference=dense_reference(raw,sa,sw,tiles,rows,hidden,depths,active)
            np.testing.assert_array_equal(expected[0],reference[0]);np.testing.assert_array_equal(expected[1],reference[1])
            report.setdefault('independent_dense_references',[]).append({'rows':rows,'hidden':hidden,'exact':True,'sha256':output_hash(reference)})
      print(json.dumps({'phase':'geometry_qualified','rows':rows,'hidden':hidden,'cases':len(report['qualification'])}),flush=True)
      del runs,last,first;gc.collect();Device['CPU'].allocator.free_cache()
    for hidden in (4864,4608):
      rows=128;raw,sa,sw,tiles=inputs(rows,hidden,128,9000+hidden,rows)
      variable=UOp.variable('ffn_probe_active',1,rows);arguments=(*raw,*sa,sw,variable.bind(rows))
      runs,depths=make_runs(pool,rows,hidden,tiles,len(raw),len(sa));last={}
      for batch in (1,2,4):
        for _ in range(4):last[batch],_=call(pool,runs[batch],batch,arguments,last.get(batch))
        captured_source(runs[batch],batch,rows)
      expected=read_outputs(last[1]);record={'rows':rows,'hidden':hidden,'blocks':[],'summaries':{}}
      for candidate in (2,4):
        paired=[];samples={1:[],candidate:[]}
        for block in range(40):
          order=(1,candidate,candidate,1) if block%2==0 else (candidate,1,1,candidate)
          block_samples=[];before=clocks()
          for batch in order:
            last[batch],ms=call(pool,runs[batch],batch,arguments,last[batch],timed=True)
            actual=read_outputs(last[batch]);np.testing.assert_array_equal(actual[0],expected[0]);np.testing.assert_array_equal(actual[1],expected[1])
            samples[batch].append(ms);block_samples.append({'batch':batch,'ms':ms,'exact':True,'sha256':output_hash(actual)})
          baseline=statistics.median(x['ms'] for x in block_samples if x['batch']==1)
          proposed=statistics.median(x['ms'] for x in block_samples if x['batch']==candidate)
          paired.append(baseline/proposed)
          record['blocks'].append({'candidate_batch':candidate,'index':block,'order':list(order),'samples':block_samples,
            'paired_speedup':baseline/proposed,'clocks_before':before,'clocks_after':clocks()})
        record['summaries'][str(candidate)]={'baseline_median_ms':statistics.median(samples[1]),
          'candidate_median_ms':statistics.median(samples[candidate]),'paired_median_speedup':statistics.median(paired),
          'faster_blocks':sum(ratio>1 for ratio in paired),'paired_blocks':40,'timed_calls_per_arm':80}
      report['timings'].append(record);print(json.dumps({'phase':'timings_complete','hidden':hidden,'summaries':record['summaries']}),flush=True)
      del runs,last;gc.collect();Device['CPU'].allocator.free_cache()
    report['accelerators_opened']=any(name.split(':')[0] in ('CL','ROCKCHIP','AMD','NV','CUDA','METAL','WEBGPU','HIP') for name in Device._opened_devices)
    assert not report['accelerators_opened']
    assert len(report['qualification'])==174 and len(report['independent_dense_references'])==2
    assert sum(len(block['samples']) for row in report['timings'] for block in row['blocks'])==640
    report.update(complete=True,numeric_pass=True,qualification_checks=174,timed_exact_checks=640,
      generated_sources=GENERATED,kernel_descriptors=list(DESCRIPTORS.values()),opened_devices=sorted(Device._opened_devices))
  except BaseException as error:
    report['error']={'type':type(error).__name__,'message':str(error)[:4000]}
    raise
  finally:
    base._program=ORIGINAL_PROGRAM;pool.close()
    args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2)+'\n')
  print(json.dumps({'complete':True,'qualification':174,'timed_exact_checks':640,'accelerators_opened':False}),flush=True)

if __name__=='__main__':main()
