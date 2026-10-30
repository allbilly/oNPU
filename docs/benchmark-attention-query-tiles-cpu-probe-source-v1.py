"""CPU-only exactness and operator timing probe for CPU attention query/PV tiles."""
import argparse
import gc
import hashlib
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
from tinygrad import Device, Tensor, TinyJit
from tinygrad.llm.kernels import rockchip_cpu as cpu_attention
from tinygrad.llm.kernels.rockchip_pool import AttentionPool
from tinygrad.uop.ops import Ops


VARIANTS=((4,2),(8,1),(8,2),(8,4),(16,1),(16,2))
BASELINE=(4,2)
SELECTED=BASELINE
ORIGINAL_ATTENTION=cpu_attention._attention
GENERATED={}


def sha256(value): return hashlib.sha256(value).hexdigest()


def program_source(program):
  return next(node.arg for node in program.src if node.op is Ops.SOURCE)


def selected_attention(*args,**kwargs):
  query_rows,channel_tile=SELECTED
  kwargs['query_rows']=query_rows
  kwargs['channel_tile']=channel_tile
  program=ORIGINAL_ATTENTION(*args,**kwargs)
  source=program_source(program)
  digest=sha256(source.encode())
  GENERATED[digest]={'source_sha256':digest,'source':source,'query_rows':query_rows,'channel_tile':channel_tile,
    'heads':args[4],'kv_heads':args[5],'dim':args[6],'context':args[7],'max_rows':args[9],
    'fast_softmax':kwargs.get('fast_softmax',False),'program_bindings':{
      'outs':list(program.arg.outs),'ins':list(program.arg.ins),'linear_inputs':len(program.src[1].src)}}
  return program


def clock_snapshot():
  snapshot={}
  for path in ('/sys/devices/system/cpu/cpufreq/policy4/scaling_cur_freq',
               '/sys/devices/system/cpu/cpufreq/policy6/scaling_cur_freq',
               '/sys/class/devfreq/dmc/cur_freq'):
    try: snapshot[path]=int(Path(path).read_text())
    except OSError: pass
  return snapshot


def make_inputs(active,total,seed):
  heads,kv_heads,dim,context=14,2,64,512
  rng=np.random.default_rng(seed)
  query=Tensor(rng.normal(0,.25,(1,heads,active,dim)).astype(np.float32),device='CPU').realize()
  cache_values=rng.normal(0,.35,(2,1,kv_heads,context,dim)).astype(np.float16)
  if total<context: cache_values[:,:,:,total:,:]=np.nan
  cache=Tensor(cache_values,device='CPU').realize()
  return query,cache


def output_array(output):
  output.realize()
  return output.numpy().copy()


def poison(output):
  np.frombuffer(output.data(),dtype=np.uint32).fill(0x7fc01234)


def check_dispatch(pool,calls_before,affinity):
  assert pool.lib.pool_calls()-calls_before==1
  assert pool.lib.pool_parked()
  assert os.sched_getaffinity(0)==affinity


def invoke(pool,run,query,cache,previous=None,timed=False):
  if previous is not None: poison(previous)
  before=pool.lib.pool_calls()
  affinity=os.sched_getaffinity(0)
  with pool.prefill():
    assert pool.active
    assert os.sched_getaffinity(0)==affinity
    began=time.perf_counter_ns() if timed else 0
    result=run(query,cache)
    elapsed=time.perf_counter_ns()-began if timed else None
    values=output_array(result)
  check_dispatch(pool,before,affinity)
  return result,values,(elapsed/1e6 if elapsed is not None else None)


def captured_program(run,variant,active,total):
  assert run.captured is not None
  programs=[call.body for call in run.captured.linear.src
    if call.body.op is Ops.PROGRAM and call.body.src[0].arg.function_name=='rockchip_cpu_attention']
  assert len(programs)==1
  program=programs[0]
  source=program_source(program)
  digest=sha256(source.encode())
  assert digest in GENERATED
  query_rows,channel_tile=variant
  assert f'for(;row+{query_rows}<=active_rows;row+={query_rows})' in source
  assert f'for(int c=0;c<16;c+={channel_tile})' in source
  descriptor=GENERATED[digest]
  assert descriptor['query_rows']==query_rows and descriptor['channel_tile']==channel_tile
  descriptor={key:value for key,value in descriptor.items() if key!='source'}
  descriptor.update({'active_rows':active,'total_length':total})
  return descriptor


def clocks(): return clock_snapshot()


def make_run(pool,variant,total):
  def run(query,cache):
    global SELECTED
    SELECTED=variant
    return cpu_attention.prefill_attention(query,cache,total,fast_softmax=True,pool=pool).realize()
  return TinyJit(run)


def time_pair(pool,runs,outputs,query,cache,candidate,expected,report_case):
  ratios=[]
  samples={BASELINE:[],candidate:[]}
  for block in range(40):
    order=(BASELINE,candidate,candidate,BASELINE) if block%2==0 else (candidate,BASELINE,BASELINE,candidate)
    before=clocks()
    block_samples=[]
    for variant in order:
      outputs[variant],actual,ms=invoke(pool,runs[variant],query,cache,outputs[variant],timed=True)
      np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))
      samples[variant].append(ms)
      block_samples.append({'variant':list(variant),'ms':ms,'exact':True,'sha256':sha256(actual.tobytes())})
    baseline=statistics.median(entry['ms'] for entry in block_samples if tuple(entry['variant'])==BASELINE)
    proposed=statistics.median(entry['ms'] for entry in block_samples if tuple(entry['variant'])==candidate)
    ratio=baseline/proposed
    ratios.append(ratio)
    report_case['blocks'].append({'candidate':list(candidate),'index':block,'order':[list(x) for x in order],
      'samples':block_samples,'paired_speedup':ratio,'clocks_before':before,'clocks_after':clocks()})
  report_case['summaries'][f'{candidate[0]}x{candidate[1]}']={
    'baseline_median_ms':statistics.median(samples[BASELINE]),
    'candidate_median_ms':statistics.median(samples[candidate]),
    'paired_median_speedup':statistics.median(ratios),'faster_blocks':sum(value>1 for value in ratios),
    'paired_blocks':40,'timed_calls_per_arm':80}


def main():
  parser=argparse.ArgumentParser()
  parser.add_argument('--report',type=Path,required=True)
  args=parser.parse_args()
  if args.report.exists(): raise FileExistsError(args.report)
  report={'schema':'attention-query-tiles-cpu-probe-v1','complete':False,'numeric_pass':False,
    'accelerators_opened':False,'quality_qualified':False,'llm_qualified':False,'goal_complete':False,
    'scope':'CPU operator comparison only; Qwen2 geometry (14 query heads, 2 KV heads, dim 64, context 512).',
    'variants':[list(value) for value in VARIANTS],'qualification':[],'timings':[],'captured_programs':[]}
  pool=None
  try:
    cpu_attention._attention=selected_attention
    pool=AttentionPool()
    deadline=time.monotonic()+2
    while any(pool.lib.pool_thread_tid(i)==0 for i in range(3)) and time.monotonic()<deadline:time.sleep(.01)
    worker_masks=[sorted(os.sched_getaffinity(pool.lib.pool_thread_tid(i))) for i in range(3)]
    assert sorted(os.sched_getaffinity(0))==[4,5,6,7] and worker_masks==[[5],[6],[7]]
    report['affinity']={'outer_mask':[4,5,6,7],'worker_masks':worker_masks,
      'caller_during_pool_work':4,'caller_pin_evidence':'parallel_parked source and cpu-core-affinity-audit-v1.json'}
    cases=[]
    for active in (17,33,65,128):
      for total in (active,512):
        for seed in (0,1): cases.append((active,total,seed))

    timing_results=[]
    for active,total,seed in cases:
      query,cache=make_inputs(active,total,1000+active*10+total+seed)
      runs={}
      outputs={}
      for variant in VARIANTS:
        runs[variant]=make_run(pool,variant,total)

      for variant in VARIANTS:
        previous=None
        for warm in range(4):
          outputs[variant],actual,_=invoke(pool,runs[variant],query,cache,previous)
          if previous is not None: np.testing.assert_array_equal(actual.view(np.uint32),previous_values.view(np.uint32))
          previous=outputs[variant]
          previous_values=actual
        report['captured_programs'].append(captured_program(runs[variant],variant,active,total))

      expected=output_array(outputs[BASELINE])
      assert np.isfinite(expected).all()
      for variant in VARIANTS:
        actual=output_array(outputs[variant])
        np.testing.assert_array_equal(actual.view(np.uint32),expected.view(np.uint32))
        report['qualification'].append({'active_rows':active,'total_length':total,'seed':seed,
          'query_rows':variant[0],'channel_tile':variant[1],'exact':True,
          'output_sha256':sha256(actual.tobytes()),'pool_dispatches_per_call':1})

      if active==128 and seed==0:
        timing_results.append((active,total,query,cache,runs,outputs,expected))
      del runs,outputs,query,cache
      gc.collect()
      Device['CPU'].allocator.free_cache()

    for active,total,query,cache,runs,outputs,expected in timing_results:
      timing={'active_rows':active,'total_length':total,'blocks':[],'summaries':{}}
      report['timings'].append(timing)
      for candidate in VARIANTS:
        if candidate==BASELINE: continue
        time_pair(pool,runs,outputs,query,cache,candidate,expected,timing)
      del runs,outputs,query,cache
      gc.collect()
      Device['CPU'].allocator.free_cache()

    opened=sorted(Device._opened_devices)
    accelerators=[name for name in opened if name.split(':')[0] in ('CL','ROCKCHIP','AMD','NV','CUDA','METAL','WEBGPU','HIP')]
    report['opened_devices']=opened
    report['accelerators_opened']=bool(accelerators)
    assert not accelerators
    assert len(report['qualification'])==96
    assert len(report['timings'])==2
    assert sum(len(item['blocks']) for item in report['timings'])==400
    assert sum(len(block['samples']) for item in report['timings'] for block in item['blocks'])==1600
    report.update(complete=True,numeric_pass=True,qualification_checks=96,timed_exact_checks=1600)
  except BaseException as error:
    report['error']={'type':type(error).__name__,'message':str(error)[:4000]}
    raise
  finally:
    cpu_attention._attention=ORIGINAL_ATTENTION
    if pool is not None:
      try: pool.close()
      except BaseException as error:
        report['cleanup_error']={'type':type(error).__name__,'message':str(error)[:2000]}
        report.update(complete=False,numeric_pass=False)
    try:
      opened=sorted(Device._opened_devices)
      report['opened_devices']=opened
      report['accelerators_opened']=any(name.split(':')[0] in ('CL','ROCKCHIP','AMD','NV','CUDA','METAL','WEBGPU','HIP') for name in opened)
    except BaseException: pass
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2)+'\n')
  print(json.dumps({'complete':report['complete'],'qualification':len(report['qualification']),
                    'timed_exact_checks':1600 if report['complete'] else 0,
                    'accelerators_opened':report['accelerators_opened']}),flush=True)


if __name__=='__main__': main()
