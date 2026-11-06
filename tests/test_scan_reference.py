import unittest
from pathlib import Path
import torch
from model.network import DIM,STATE,SelectiveClassifier,digits_data
from model.scan_reference import round_shift,advance,run_hybrid

class ScanReferenceTests(unittest.TestCase):
    def test_nearest_even_signed(self):
        values=torch.tensor([-1536,-512,512,1536,2560],dtype=torch.int64)
        self.assertEqual(round_shift(values,10).tolist(),[-2,0,0,2,2])

    def test_exact_state_step(self):
        h=torch.zeros((1,DIM,STATE),dtype=torch.int64)
        a=torch.full_like(h,1024)
        e=torch.zeros_like(h)
        e[:,0,0]=512
        c=torch.zeros((1,STATE),dtype=torch.int64)
        c[:,0]=1024
        u=torch.zeros((1,DIM),dtype=torch.int64)
        d=torch.zeros((DIM,),dtype=torch.int64)
        bits=torch.full((DIM,STATE),10,dtype=torch.int64)
        state,y=advance(h,a,e,c,d,u,bits)
        self.assertEqual(int(state[0,0,0]),512)
        self.assertEqual(int(y[0,0]),512)

    def test_full_precision_hybrid_equals_pytorch(self):
        ck=torch.load('weights/digits_bilinear.pt',map_location='cpu',weights_only=True)
        model=SelectiveClassifier(ck['scheme'],ck['qat'])
        model.load_state_dict(ck['weights']);model.eval()
        _,_,(inputs,_)=list(digits_data())
        inputs=inputs[:12]
        with torch.no_grad():
            ref=model(inputs)
            replay=run_hybrid(model,inputs,torch.full((DIM,STATE),10,dtype=torch.int64))
        self.assertTrue(torch.equal(ref,replay), f'max error {(ref-replay).abs().max().item()}')

if __name__=='__main__':
    unittest.main()
