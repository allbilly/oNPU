"""Unexecuted CPU prototype: interleave raw FFN rows without changing their arithmetic.

This transforms the actual generated source, keeping its layout-specific loads,
activation expression, scalar reduction and quantization. It is not a production
backend and makes no performance claim. Call program_variant from an isolated
CPU probe and compare complete payload/scale bits before timing it.
"""
import re
from tinygrad.llm.kernels import rockchip_gelu_ffn as base
from tinygrad.uop.ops import Ops,UOp

LOOP='for(int row=begin;row<end;row++) {'

def closing_brace(text,opening):
  depth=0
  for index in range(opening,len(text)):
    if text[index]=='{':depth+=1
    elif text[index]=='}':
      depth-=1
      if depth==0:return index
  raise ValueError('Unbalanced generated C scope')

def words(text,substitutions):
  return re.sub(r'\b('+ '|'.join(map(re.escape,substitutions))+r')\b',lambda match:substitutions[match[0]],text)

def interleaved_projection(projection,batch):
  pieces=[];position=0
  for match in re.finditer(r'for\(int c=\d+;c<\d+;c\+=4\) \{',projection):
    if match.start()<position:continue
    opening=match.end()-1;end=closing_brace(projection,opening)
    pieces.append(projection[position:opening+1])
    body=projection[opening+1:end]
    for lane in range(batch):
      pieces.append('{'+words(body,{'row':f'(row+{lane})','values':f'batch_values[{lane}]',
        'maximum':f'batch_maximum[{lane}]','bad':f'batch_bad[{lane}]'})+'}')
    pieces.append('}');position=end+1
  if position==0:raise ValueError('No static channel fragments')
  pieces.append(projection[position:])
  result=''.join(pieces)
  result=re.sub(r'if\(row>=(\d+) && row<(\d+)\)',
                lambda match:f'if(row>={match[1]} && row+{batch}<={match[2]})',result)
  return result

def transform(source,rows,hidden,depths,tiles,activation,batch):
  if batch not in (2,4):raise ValueError('Prototype row batch must be two or four')
  if rows<batch:return source
  if source.count(LOOP)!=len(depths):raise ValueError('Unexpected raw FFN loop structure')
  bounds=sorted({0,rows,*(t.first_row for t in tiles),*(t.first_row+t.rows for t in tiles)})
  safe=' || '.join(f'(row>={first} && row+{batch}<={last})' for first,last in zip(bounds,bounds[1:]))
  pieces=[];position=0;first_channel=0
  for depth in depths:
    start=source.index(LOOP,position);opening=start+len(LOOP)-1;end=closing_brace(source,opening)
    body=source[opening+1:end]
    projection=base._static_projection_loop(tiles,rows,hidden,first_channel,depth,activation)
    if body.count(projection)!=1:raise ValueError('Static projection did not match the compiled baseline')
    _,tail=body.split(projection)
    metadata=body[:body.index('if(row>=active)')]
    if 'float values' in metadata:raise ValueError('Unexpected metadata boundary')
    paired=f'''if(row+{batch}<=end && row+{batch}<=active && ({safe})) {{
      float batch_values[{batch}][{depth}] __attribute__((aligned(16)));
      float4 batch_maximum[{batch}]={{0}};uint4 batch_bad[{batch}]={{0}};
      {interleaved_projection(projection,batch)}
    '''
    for lane in range(batch):
      mapping={'row':'batch_row','values':f'batch_values[{lane}]',
               'maximum':f'batch_maximum[{lane}]','bad':f'batch_bad[{lane}]'}
      paired+='{'+f'int batch_row=row+{lane};'+words(metadata+tail,mapping)+'}'
    paired+=f'row+={batch-1};continue;}}\n'
    pieces.append(source[position:opening+1]+paired+body+'}')
    position=end+1;first_channel+=depth
  pieces.append(source[position:])
  return ''.join(pieces)

def program_variant(program,rows,hidden,depths,tiles,activation,batch):
  """Recompile one CPU program; its UOp bindings and single-row code stay intact."""
  sources=[node for node in program.src if node.op is Ops.SOURCE]
  binaries=[node for node in program.src if node.op is Ops.BINARY]
  if len(sources)!=1 or len(binaries)!=1:raise ValueError('Unexpected CPU program representation')
  source=transform(sources[0].arg,rows,hidden,depths,tiles,activation,batch)
  if source==sources[0].arg:return program
  binary=base._renderer().compiler.compile_cached(source)
  children=tuple(UOp(Ops.SOURCE,arg=source) if node.op is Ops.SOURCE else
                 UOp(Ops.BINARY,arg=binary) if node.op is Ops.BINARY else node for node in program.src)
  return UOp(program.op,arg=program.arg,src=children)
