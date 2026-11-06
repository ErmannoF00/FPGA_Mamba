import tempfile
import unittest
from pathlib import Path
import torch
from model.network import DIM,STATE,TIME_STEPS,SelectiveClassifier,digits_data
from model.scan_reference import run_hybrid
from model.rtl_classifier import parse_scan_trace


class RtlClassifierTests(unittest.TestCase):
    def test_packed_signed_lanes(self):
        vals=[-32768,-1,0,1,32767,-2048,42,-42]
        packed=''.join(f'{v&65535:04x}' for v in reversed(vals))
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'trace.txt'
            p.write_text('\n'.join([packed]*TIME_STEPS)+'\n')
            data=parse_scan_trace(p,1)
            self.assertEqual(data.shape,(1,TIME_STEPS,DIM))
            self.assertEqual(data[0,0].tolist(),vals)

    def test_malformed_or_incomplete_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'trace.txt'
            p.write_text('0000\n')
            with self.assertRaises(ValueError):parse_scan_trace(p,1)
            p.write_text(('not_hex'+'0'*25+'\n')*TIME_STEPS)
            with self.assertRaises(ValueError):parse_scan_trace(p,1)

    def test_replay_matches_python_and_rejects_corruption(self):
        torch.set_num_threads(2)
        model=SelectiveClassifier('bilinear',qat=True).eval()
        _,_,(images,_)=list(digits_data())
        batch=2
        bits=torch.full((DIM,STATE),3,dtype=torch.int64)
        collected={}
        with torch.no_grad():
            gold=run_hybrid(model,images[:batch],bits,
                capture=lambda s,t,u,a,e,c,skip,y:collected.__setitem__((s,t),y.clone()))
            replay=torch.stack([torch.stack([collected[s,t] for t in range(TIME_STEPS)])
                                for s in range(batch)])
            reconstructed=run_hybrid(model,images[:batch],bits,rtl_outputs=replay)
            self.assertTrue(torch.equal(gold,reconstructed))
            corrupted=replay.clone()
            corrupted[1,4,2]+=1
            with self.assertRaisesRegex(ValueError,'sample=1, token=4, channel=2'):
                run_hybrid(model,images[:batch],bits,rtl_outputs=corrupted)

if __name__=='__main__':unittest.main()
