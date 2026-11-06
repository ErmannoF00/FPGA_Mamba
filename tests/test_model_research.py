import unittest
import torch
from model.network import DIM,STATE,SelectiveClassifier,digits_data
from model.precision import allocate

class ModelTests(unittest.TestCase):
    def test_model_shape(self):
        m=SelectiveClassifier('bilinear',qat=True)
        out=m(torch.zeros((2,8,8)))
        self.assertEqual(tuple(out.shape),(2,10))
        self.assertTrue(torch.isfinite(out).all().item())

    def test_bilinear_stability(self):
        m=SelectiveClassifier('bilinear',qat=False)
        with torch.no_grad():
            dt=torch.full((3,DIM),0.5)
            alpha,gain=m.dynamics(dt)
            self.assertTrue((alpha.abs() < 1).all().item())
            self.assertTrue((gain > 0).all().item())

    def test_precision_budget(self):
        cost=torch.rand((DIM,STATE,5),generator=torch.Generator().manual_seed(9))
        bits=allocate(cost,[0,1,2,3,4],64)
        self.assertEqual(tuple(bits.shape),(DIM,STATE))
        self.assertEqual(int(bits.sum().item()),64)
        self.assertGreaterEqual(int(bits.min().item()),0)
        self.assertLessEqual(int(bits.max().item()),4)

    def test_precision_clamp_shape(self):
        m=SelectiveClassifier('bilinear',qat=True)
        with self.assertRaisesRegex(ValueError,'state_bits'):
            m(torch.zeros((1,8,8)),state_bits=torch.zeros(5))

    def test_real_dataset_size(self):
        train,val,test=list(digits_data())
        self.assertEqual(len(train[0])+len(val[0])+len(test[0]),1797)
        self.assertEqual(len(test[0]),360)

if __name__=='__main__': unittest.main()
