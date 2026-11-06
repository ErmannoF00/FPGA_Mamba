"""Calibration-only bit allocation and held-out evaluation of recurrent state storage.

This is an exploratory software study, not a hardware result. State precision is
applied after each recurrence update; other operations remain in Q6.10.
"""
import argparse
import json
from pathlib import Path
import torch
from .network import SelectiveClassifier, digits_data, DIM, STATE


def scores(model, images, bit_choices):
    with torch.no_grad():
        target=model(images)
        errors=torch.zeros((DIM,STATE,len(bit_choices)), dtype=torch.float64)
        for d in range(DIM):
            for n in range(STATE):
                for j,bits in enumerate(bit_choices):
                    assignment=torch.full((DIM,STATE),10.)
                    assignment[d,n]=bits
                    pred=model(images, state_bits=assignment)
                    errors[d,n,j]=torch.mean((pred-target).square()).item()
    return errors


def allocate(scores, bits, total_budget):
    """Dynamic program minimizing additive single-lane proxy error."""
    dims= scores.shape[0]*scores.shape[1]
    costs=scores.reshape(dims,len(bits)).tolist()
    table={0:(0.0,())}
    for idx in range(dims):
        nxt={}
        for used,(error,assignment) in table.items():
            for j,bit in enumerate(bits):
                new_used=used+bit
                if new_used>total_budget:continue
                new_err=error+costs[idx][j]
                if new_used not in nxt or new_err<nxt[new_used][0]:
                    nxt[new_used]=(new_err,assignment+(bit,))
        table=nxt
        if not table:raise ValueError('insufficient budget')
    # Spend exactly the allocated budget to compare equal-size designs.
    if total_budget not in table:raise ValueError('budget cannot be represented')
    return torch.tensor(table[total_budget][1]).reshape(DIM,STATE)


def run(checkpoint, output, budget=64, maximum=6, stability_weight=0.10):
    torch.set_num_threads(2)
    data=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model=SelectiveClassifier(data['scheme'],data['qat'])
    model.load_state_dict(data['weights']);model.eval()
    _,(x_val,y_val),(x_test,y_test)=list(digits_data())
    bit_choices=list(range(maximum+1))
    local_errors=scores(model,x_val,bit_choices)
    with torch.no_grad():
        poles=-(0.05+1.45*torch.sigmoid(model.a_logits))
        dt_min=0.125
        rho=((1+0.5*dt_min*poles)/(1-0.5*dt_min*poles)).abs()
        lifetime=1/(1-rho)
        influence=lifetime/lifetime.mean()
    weighted=local_errors*(1+stability_weight*(influence.double().unsqueeze(-1)-1))
    allocation=allocate(weighted,bit_choices,budget)
    def summary(x,y):
        with torch.no_grad():
            target=model(x)
            uniform=model(x,state_bits=torch.full((DIM,STATE),budget//(DIM*STATE)))
            adaptive=model(x,state_bits=allocation)
            def metrics(z):
                return {'accuracy':float((z.argmax(1)==y).float().mean()),
                  'logit_mse':float(torch.mean((z-target).square())),
                  'logit_max_abs':float(torch.max(torch.abs(z-target))),
                  'prediction_changes':int((z.argmax(1)!=target.argmax(1)).sum())}
            return {'reference':metrics(target),'uniform':metrics(uniform),'adaptive':metrics(adaptive)}
    report={'method':'independent-lane output-error calibration + exact-bit-budget DP',
            'budget_fractional_bits':budget,'state_count':DIM*STATE,
            'stability_weight':stability_weight,
            'max_pole_magnitude':float(rho.max()),
            'bits_per_state':allocation.int().tolist(),
            'validation':summary(x_val,y_val),'test':summary(x_test,y_test),
            'caveat':'Only state storage quantized. No RTL synthesis, FPGA timing or energy measurement.'}
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    Path(output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,default=Path('weights/digits_bilinear.pt'))
    p.add_argument('--output',type=Path,default=Path('build/precision.json'))
    p.add_argument('--budget',type=int,default=64)
    p.add_argument('--maximum',type=int,default=6)
    p.add_argument('--stability-weight',type=float,default=0.10)
    a=p.parse_args();run(a.checkpoint,a.output,a.budget,a.maximum,a.stability_weight)
