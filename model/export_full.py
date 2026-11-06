"""Export a trained checkpoint and held-out images to synchronous RTL ROMs."""
import argparse
import json
from pathlib import Path
import torch
from .full_integer import prepare, forward, quant, pack_image, pack_scores


def export(checkpoint, bits, out, samples=50):
    from .network import SelectiveClassifier, digits_data
    p = prepare(checkpoint, bits)
    out = Path(out)
    out.mkdir(parents=True,exist_ok=True)
    def mem(name, arr):
        flattened=[]
        def walk(x):
            if isinstance(x, list):
                for y in x: walk(y)
            else: flattened.append(int(x))
        walk(arr)
        (out/(name+'.mem')).write_text(''.join(f'{v&0xffff:04x}\n' for v in flattened))
    for key, val in p.items():
        if key.endswith(('_w','_b')) or key in ('conv_w','conv_b','skip'):
            mem(key,val)
    mem('alpha', [[entry['alpha'][d][n] for n in range(4)]
                  for entry in p['table'] for d in range(8)])
    mem('gain', [[entry['gain'][d][n] for n in range(4)]
                 for entry in p['table'] for d in range(8)])
    precision=sum(int(bit)<<(4*i) for i,bit in enumerate(sum(p['bits'],[])))
    (out/'precision_config.svh').write_text(
        '`ifndef PRECISION_CONFIG_SVH\n`define PRECISION_CONFIG_SVH\n'
        f'`define DIGITS_STATE_FRAC_MAP 128\'h{precision:032x}\n`endif\n')
    _,_,(images, labels)=list(digits_data())
    ckpt=torch.load(str(checkpoint),map_location='cpu',weights_only=True)
    model=SelectiveClassifier(ckpt['scheme'],ckpt['qat'])
    model.load_state_dict(ckpt['weights']);model.eval()
    rows=[]; agreement=0; correct=0; py_correct=0
    with torch.no_grad():
        host=model(images[:samples]).argmax(1)
    for i in range(samples):
        img=[[quant(float(v)) for v in row] for row in images[i]]
        scores=forward(img,p)
        pred=max(range(10),key=lambda j:scores[j])
        agreement+=int(pred==int(host[i])); correct+=int(pred==int(labels[i]));py_correct+=int(int(host[i])==int(labels[i]))
        rows.append(f'{pack_image(img)} {pack_scores(scores)} {int(labels[i])} {pred}\n')
    (out/'images_expected.txt').write_text(''.join(rows))
    meta={'samples':samples,'integer_accuracy':correct/samples,
          'pytorch_accuracy':py_correct/samples,'prediction_agreement':agreement/samples,
          'rtl_scope':'input image to 10 logits; Python integer oracle, pending RTL execution',
          'format':p['format']}
    (out/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta,indent=2))
    return meta

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',default='weights/digits_bilinear.pt')
    ap.add_argument('--bits',default='results/precision64.json')
    ap.add_argument('--out',default='build/full_rom')
    ap.add_argument('--samples',type=int,default=50)
    a=ap.parse_args()
    export(a.checkpoint,a.bits,a.out,a.samples)
