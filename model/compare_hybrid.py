"""Compare held-out PyTorch logits with integer selective-scan replay."""
import argparse
import json
from pathlib import Path
import torch
from .network import DIM, STATE, SelectiveClassifier, digits_data
from .scan_reference import run_hybrid


def compare(checkpoint, precision, output):
    torch.set_num_threads(2)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model=SelectiveClassifier(saved['scheme'],saved['qat'])
    model.load_state_dict(saved['weights']);model.eval()
    _,_,(images,labels)=list(digits_data())
    bits=torch.tensor(json.loads(Path(precision).read_text())['bits_per_state'])
    with torch.no_grad():
        logits=model(images)
        full=run_hybrid(model,images,torch.full((DIM,STATE),10,dtype=torch.int64))
        mixed=run_hybrid(model,images,bits)
        def metrics(pred):
            return {'accuracy':float((pred.argmax(1)==labels).float().mean()),
                'max_logit_abs_error':float((pred-logits).abs().max()),
                'logit_mse':float(((pred-logits)**2).mean()),
                'prediction_changes':int((pred.argmax(1)!=logits.argmax(1)).sum())}
        result={'samples':len(labels),'model':metrics(logits),
                'integer_scan_full':metrics(full),'integer_scan_mixed':metrics(mixed),
                'scope':'Only scan is integer; other network operations in PyTorch'}
    path=Path(output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,default=Path('weights/digits_bilinear.pt'))
    p.add_argument('--precision',type=Path,default=Path('results/precision64.json'))
    p.add_argument('--output',type=Path,default=Path('build/hybrid.json'))
    args=p.parse_args()
    compare(args.checkpoint,args.precision,args.output)
