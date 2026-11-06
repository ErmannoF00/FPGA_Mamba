"""Consume outputs emitted by the RTL scan and evaluate trained classifiers.

The SystemVerilog testbench must already have compared every cycle against its
independent integer reference. This tool verifies output ordering, injects the
observed signed scan outputs into the PyTorch post-scan layers, and reports
sample-level predictions. The front end and head are not FPGA implementations.
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from .network import DIM, STATE, TIME_STEPS, SelectiveClassifier, digits_data
from .scan_reference import run_hybrid


def parse_scan_trace(path, sample_count):
    lines = Path(path).read_text().splitlines()
    expected = sample_count * TIME_STEPS
    if len(lines) != expected:
        raise ValueError(f"Expected {expected} consumed scan outputs, got {len(lines)}")
    outputs = torch.empty((sample_count,TIME_STEPS,DIM),dtype=torch.int64)
    for ordinal,line in enumerate(lines):
        token=line.strip()
        if len(token)!=DIM*4 or any(c not in '0123456789abcdefABCDEF' for c in token):
            raise ValueError(f"Bad 128-bit output word on trace line {ordinal+1}")
        # Lane 0 occupies the least-significant 16 bits of the RTL bus.
        for d in range(DIM):
            word=int(token[4*(DIM-1-d):4*(DIM-d)],16)
            outputs[ordinal//TIME_STEPS,ordinal%TIME_STEPS,d]=word-(65536 if word & 32768 else 0)
    return outputs


def evaluate(checkpoint,bits_file,trace_path,samples,output):
    torch.set_num_threads(2)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model=SelectiveClassifier(saved['scheme'],saved['qat'])
    model.load_state_dict(saved['weights'])
    model.eval()
    bits=torch.tensor(json.loads(Path(bits_file).read_text())['bits_per_state'],dtype=torch.int64)
    if tuple(bits.shape)!=(DIM,STATE):
        raise ValueError('bits_per_state shape mismatch')
    _,_,(images,labels)=list(digits_data())
    if samples<1 or samples>len(images):
        raise ValueError('samples outside held-out partition')
    observed=parse_scan_trace(trace_path,samples)
    with torch.no_grad():
        host_logits=model(images[:samples])
        hybrid_logits=run_hybrid(model,images[:samples],bits)
        rtl_logits=run_hybrid(model,images[:samples],bits,rtl_outputs=observed)
    host_pred=host_logits.argmax(dim=1)
    hybrid_pred=hybrid_logits.argmax(dim=1)
    rtl_pred=rtl_logits.argmax(dim=1)
    delta=(rtl_logits-hybrid_logits).abs()
    if not torch.equal(rtl_pred,hybrid_pred) or torch.any(delta!=0):
        raise AssertionError('RTL-injected logits differ from integer software reference')
    def accuracy(pred):
        return float((pred==labels[:samples]).float().mean())
    result={
        'scope':'RTL-computed recurrent scan + PyTorch projections, gate, output projection and head',
        'samples':samples,'rtl_scan_transfers':samples*TIME_STEPS,
        'checkpoint_sha256':hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        'trace_sha256':hashlib.sha256(Path(trace_path).read_bytes()).hexdigest(),
        'float_model_accuracy':accuracy(host_pred),
        'integer_software_accuracy':accuracy(hybrid_pred),
        'rtl_injected_accuracy':accuracy(rtl_pred),
        'rtl_vs_integer_max_logit_abs_error':float(delta.max()),
        'rtl_vs_float_max_logit_abs_error':float((rtl_logits-host_logits).abs().max()),
        'rtl_vs_integer_prediction_changes':int((rtl_pred!=hybrid_pred).sum()),
        'rtl_vs_float_prediction_changes':int((rtl_pred!=host_pred).sum()),
        'per_sample':[{'sample':i,'label':int(labels[i]),
                       'float_prediction':int(host_pred[i]),
                       'integer_prediction':int(hybrid_pred[i]),
                       'rtl_prediction':int(rtl_pred[i])} for i in range(samples)]
    }
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='per_sample'},indent=2))
    print(f'Sample-level predictions written to {output}')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,default=Path('weights/digits_bilinear.pt'))
    p.add_argument('--bits',type=Path,default=Path('results/precision64.json'))
    p.add_argument('--trace',type=Path,default=Path('build/precision_scan_outputs.txt'))
    p.add_argument('--samples',type=int,default=50)
    p.add_argument('--output',type=Path,default=Path('build/rtl_classifier.json'))
    args=p.parse_args()
    evaluate(args.checkpoint,args.bits,args.trace,args.samples,args.output)
