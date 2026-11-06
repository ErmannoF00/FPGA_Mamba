"""Evaluate a serialized Mamba-derived classifier on held-out digits."""
import argparse
import json
from pathlib import Path
import torch
from .network import SelectiveClassifier, digits_data
from .train import evaluate


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,default=Path('weights/digits_bilinear.pt'))
    args=parser.parse_args()
    raw=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    model=SelectiveClassifier(raw['scheme'],raw['qat'])
    model.load_state_dict(raw['weights'])
    model.eval()
    _,val,test=list(digits_data())
    accuracy, loss=evaluate(model,test)
    report={'model':raw['scheme'],'qat':raw['qat'],
            'checkpoint':str(args.checkpoint),'test_accuracy':accuracy,
            'test_cross_entropy':loss,'test_samples':len(test[1])}
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
