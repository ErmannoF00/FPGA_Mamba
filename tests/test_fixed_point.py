"""Reference arithmetic, reset and backpressure contracts."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from fixed_point import ScanModel, pack, quantize, saturate, unpack


class FixedPointTests(unittest.TestCase):
    def test_signed_pack_roundtrip(self):
        values = [-32768, -1, 0, 32767]
        self.assertEqual(unpack(pack(values)), values)

    def test_saturation(self):
        self.assertEqual(saturate(99999), 32767)
        self.assertEqual(saturate(-99999), -32768)

    def test_hold_and_reset(self):
        m = ScanModel()
        z, one, two = [0] * 4, [4096] * 4, [2048] * 4
        m.tick(rst_n=True, in_valid=True, out_ready=False,
               sequence_start=True, x=one, a=two, b=two, c=two)
        state = m.state.copy()
        result = m.output.copy()
        obs = m.tick(rst_n=True, in_valid=True, out_ready=False,
                     sequence_start=False, x=one, a=one, b=one, c=one)
        self.assertEqual(obs[1:], (True, result))
        self.assertEqual(m.state, state)
        m.tick(rst_n=False, in_valid=False, out_ready=False,
               sequence_start=False, x=z, a=z, b=z, c=z)
        self.assertEqual(m.state, z)
        self.assertFalse(m.out_valid)

    def test_nontrivial_recurrence(self):
        m = ScanModel()
        one = [4096] * 4
        half = [2048] * 4
        zero = [0] * 4
        for t in range(3):
            m.tick(rst_n=True, in_valid=True, out_ready=True,
                   sequence_start=(t == 0), x=one, a=half, b=half, c=one)
        self.assertEqual(m.state, [3584] * 4)  # 0.875 in Q4.12
        self.assertEqual(m.output, [5632] * 4)  # h + 0.5*x

    def test_negative_fraction_rounds_down(self):
        # Arithmetic right shift must floor negatives, not truncate to zero.
        m = ScanModel(lanes=1)
        m.tick(rst_n=True, in_valid=True, out_ready=True,
               sequence_start=True, x=[1], a=[0], b=[-1], c=[4096])
        self.assertEqual(m.state, [-1])
        self.assertEqual(m.output, [-1])

    def test_eight_lane_reset_and_pack(self):
        m = ScanModel(lanes=8)
        values = [-32768, -2048, -1, 0, 1, 2048, 32767, 99]
        self.assertEqual(unpack(pack(values), lanes=8), values)
        self.assertEqual(m.state, [0] * 8)
        m.tick(rst_n=True, in_valid=True, out_ready=True,
               sequence_start=True, x=values, a=[0]*8,
               b=[4096]*8, c=[4096]*8)
        self.assertEqual(m.state, values)

    def test_backpressure_preserves_pending_output(self):
        m = ScanModel(lanes=1)
        m.tick(rst_n=True, in_valid=True, out_ready=False,
               sequence_start=True, x=[4096], a=[0], b=[4096], c=[4096])
        first_y = m.output.copy()
        for _ in range(80):
            ready, valid, data = m.tick(
                rst_n=True, in_valid=True, out_ready=False,
                sequence_start=True, x=[-4096], a=[4096], b=[4096], c=[4096])
            self.assertFalse(ready)
            self.assertTrue(valid)
            self.assertEqual(data, first_y)
        self.assertEqual(m.state, [4096])


if __name__ == "__main__":
    unittest.main()
