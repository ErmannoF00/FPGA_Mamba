import unittest
from model.full_integer import rounded_div, mulq, clip16, linear, forward, prepare, quant

class IntegerTests(unittest.TestCase):
    def test_ties_even(self):
        for val, den, expect in [(5,2,2),(7,2,4),(-5,2,-2),(-7,2,-4),(1,2,0),(-1,2,0)]:
            self.assertEqual(rounded_div(val,den),expect)
    def test_rounding_and_saturation(self):
        self.assertEqual(mulq(32767,32767),32767)
        self.assertEqual(mulq(-32768,32767),-32768)
        self.assertEqual(clip16(50000),32767)
    def test_linear(self):
        self.assertEqual(linear([1024,-512],[[1024,1024]], [0]),[512])
    def test_actual_checkpoint_first_samples(self):
        from pathlib import Path
        from model.network import digits_data
        root = Path(__file__).resolve().parents[1]
        cfg = prepare(root/'weights/digits_bilinear.pt',root/'results/precision64.json')
        _,_,(images,labels) = list(digits_data())
        seen = 0
        for index in range(16):
            image = [[quant(float(p)) for p in row] for row in images[index]]
            scores = forward(image,cfg)
            self.assertEqual(len(scores),10)
            pred = max(range(10),key=lambda i:scores[i])
            seen += int(pred==int(labels[index]))
        self.assertGreaterEqual(seen,10)

    def test_shape(self):
        with self.assertRaises(ValueError):forward([[0]*8],{})

if __name__=='__main__':unittest.main()
