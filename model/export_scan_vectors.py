"""Export trained-model scan transactions with an independent integer scoreboard.

Only the recurrent scan executes in RTL. All projections, convolution, gating
and the classification head remain on the Python host.
"""
import argparse
import json
from pathlib import Path
import random
import torch
from .network import SelectiveClassifier, digits_data, DIM, STATE
from .scan_reference import run_hybrid


def pack(value):
    values=torch.as_tensor(value).reshape(-1).tolist()
    return ''.join(f'{int(v)&0xffff:04x}' for v in reversed(values))


def export(checkpoint, bits_file, vectors, configuration, samples=50, seed=909):
    torch.set_num_threads(2)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model=SelectiveClassifier(saved['scheme'],saved['qat'])
    model.load_state_dict(saved['weights']);model.eval()
    bits=torch.tensor(json.loads(Path(bits_file).read_text())['bits_per_state'],dtype=torch.int64)
    if tuple(bits.shape)!=(DIM,STATE): raise ValueError('precision bit-map shape mismatch')
    _,_,(images,labels)=list(digits_data())
    if not 1<=samples<=len(images): raise ValueError('invalid sample count')
    records={}
    def capture(s,t,u,a,e,c,skip,y):
        records[s,t]=(u.clone(),a.clone(),e.clone(),c.clone(),skip.clone(),y.clone())
    with torch.no_grad():
        scores=run_hybrid(model,images[:samples],bits,capture)
    accuracy=float((scores.argmax(1)==labels[:samples]).float().mean().item())
    map_int=sum(int(bit)<<(4*i) for i,bit in enumerate(bits.reshape(-1).tolist()))
    configuration.parent.mkdir(parents=True,exist_ok=True)
    configuration.write_text(
        '`ifndef PRECISION_CONFIG_SVH\n`define PRECISION_CONFIG_SVH\n'
        f'`define DIGITS_STATE_FRAC_MAP 128\'h{map_int:032x}\n`endif\n')
    rng=random.Random(seed)
    queue=[(s,t,records[s,t]) for s in range(samples) for t in range(8)]
    pending=None
    index=0
    out_valid=False
    out_y='0'*(DIM*4)
    count=0
    covered={'transfers':0,'stalls':0,'bubbles':0,'reset':0,'sequences':0}
    vectors.parent.mkdir(parents=True,exist_ok=True)
    with vectors.open('w') as f:
        for cycle in range(samples*24+32):
            reset=cycle>=3
            ready=(cycle%73 not in range(11,27)) and (rng.randrange(5)!=0)
            if pending is None and index<len(queue) and rng.randrange(6)!=0 and reset:
                pending=queue[index]
            valid=pending is not None and reset
            if pending is None:
                s=t=0
                u=torch.zeros(DIM,dtype=torch.int64)
                a=e=torch.zeros((DIM,STATE),dtype=torch.int64)
                c=torch.zeros(STATE,dtype=torch.int64)
                skip=torch.zeros(DIM,dtype=torch.int64)
                pred_y=torch.zeros(DIM,dtype=torch.int64)
            else:
                s,t,(u,a,e,c,skip,pred_y)=pending
            seq=bool(valid and t==0)
            in_ready=not out_valid or ready
            expected_valid=out_valid if reset else False
            expected_y=out_y if reset else '0'*(DIM*4)
            if not reset:in_ready=True
            row=[int(reset),int(valid),int(ready),int(seq),
                 pack(u),pack(a),pack(e),pack(c),pack(skip),
                 int(in_ready),int(expected_valid),expected_y]
            f.write(' '.join(map(str,row))+'\n')
            count+=1
            if not reset:
                out_valid=False;out_y='0'*(DIM*4);covered['reset']+=1
            elif in_ready:
                out_valid=valid
                if valid:
                    out_y=pack(pred_y);index+=1;pending=None;covered['transfers']+=1
                    covered['sequences']+=int(seq)
                else:
                    covered['bubbles']+=1
            else:
                covered['stalls']+=1
            if index==len(queue) and not out_valid and cycle>12:break
    if index!=len(queue):raise RuntimeError(f'only {index}/{len(queue)} tokens emitted')
    if not covered['stalls'] or not covered['bubbles']:raise RuntimeError('insufficient protocol coverage')
    report={'dataset':'sklearn handwritten digits, fixed held-out partition',
      'samples':samples,'scan_tokens':index,'token_cycles':count,
      'hybrid_accuracy':accuracy,'coverage':covered,
      'state_fractional_bits':bits.tolist(),
      'scope':'PyTorch front end/head; RTL replay only for recurrent scan'}
    summary=vectors.with_suffix('.json')
    summary.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,default=Path('weights/digits_bilinear.pt'))
    p.add_argument('--bits',type=Path,default=Path('results/precision64.json'))
    p.add_argument('--vectors',type=Path,default=Path('build/model_scan_vectors.txt'))
    p.add_argument('--configuration',type=Path,default=Path('rtl/precision_config.svh'))
    p.add_argument('--samples',type=int,default=50)
    a=p.parse_args();export(a.checkpoint,a.bits,a.vectors,a.configuration,a.samples)
