"""Train and export a small real-dataset Mamba-derived image classifier."""
import argparse
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from .network import SelectiveClassifier, digits_data


def evaluate(model, data, batch_size=256):
    model.eval()
    x, y = data
    correct = 0
    total_loss = 0.
    with torch.no_grad():
        for first in range(0, len(x), batch_size):
            scores = model(x[first:first+batch_size])
            correct += (scores.argmax(1) == y[first:first+batch_size]).sum().item()
            total_loss += F.cross_entropy(scores, y[first:first+batch_size], reduction='sum').item()
    return correct/len(y), total_loss/len(y)


def train(scheme='bilinear', epochs=30, seed=19, directory=Path('build/model'), qat=True):
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    train_set, val_set, test_set = digits_data()
    model = SelectiveClassifier(scheme=scheme, qat=qat)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.004, weight_decay=1e-4)
    best, best_state = -1., None
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(train_set[0]), generator=generator)
        for ids in order.split(128):
            optimizer.zero_grad(set_to_none=True)
            scores = model(train_set[0][ids])
            loss = F.cross_entropy(scores, train_set[1][ids])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        val_acc, _ = evaluate(model, val_set)
        if val_acc > best:
            best = val_acc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch % 5 == 0 or epoch+1 == epochs:
            print(f'epoch={epoch+1:03} validation_accuracy={val_acc:.4f} best={best:.4f}', flush=True)
    model.load_state_dict(best_state)
    test_acc, test_loss = evaluate(model, test_set)
    torch.save({'weights':model.state_dict(),'scheme':scheme,'qat':qat,
                'seed':seed,'epochs':epochs}, directory/'model.pt')
    results = {'dataset':'sklearn.datasets.load_digits', 'source_samples':1797,
               'train':len(train_set[1]), 'validation':len(val_set[1]), 'test':len(test_set[1]),
               'seed':seed, 'best_validation_accuracy':best, 'test_accuracy':test_acc,
               'test_cross_entropy':test_loss, 'scheme':scheme, 'qat':qat,
               'precision':'Q6.10 signed 16-bit'}
    (directory/'metrics.json').write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps(results, indent=2),flush=True)
    return model,results


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--epochs',type=int,default=30)
    parser.add_argument('--seed',type=int,default=19)
    parser.add_argument('--scheme',choices=('bilinear','zoh'),default='bilinear')
    parser.add_argument('--output',type=Path,default=Path('build/model'))
    parser.add_argument('--no-qat',action='store_true')
    a=parser.parse_args()
    train(a.scheme,a.epochs,a.seed,a.output,not a.no_qat)
