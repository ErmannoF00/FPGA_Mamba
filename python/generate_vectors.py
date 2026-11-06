"""Deterministic cycle-level equivalence vectors for selective_scan.sv.

The vector file contains input pins and EXPECTED PRE-RISING-EDGE outputs.
One row represents a full transfer opportunity (including bubbles and stalls).
"""
import argparse
import random
from pathlib import Path

from fixed_point import ScanModel, pack, quantize


def generate(path: Path, cycles: int, seed: int, lanes: int = 4) -> dict:
    if lanes < 1 or cycles < 300:
        raise ValueError("Require >=1 lane and >=300 cycles for coverage")
    rng = random.Random(seed)
    model = ScanModel(lanes=lanes)
    counts = {k: 0 for k in ("accepted", "stalled", "bubbles", "resets", "sequence_starts", "saturating_inputs")}
    with path.open("w", encoding="ascii") as stream:
        for cycle in range(cycles):
            # Exercise reset in flight, long forced stalls, and restart on release.
            rst_n = cycle not in (0, 1, 2, 257, 258, cycles - 90)
            valid = cycle < 20 or (rng.randrange(6) != 0)
            ready = False if cycle % 113 in range(20, 45) else rng.randrange(5) != 0
            seq = cycle in (3, 259, cycles - 89) or rng.randrange(37) == 0
            # Force all lanes to cover saturating positive/negative values.
            if cycle % 101 in (1, 2):
                x = [32767 if cycle % 101 == 1 else -32768] * lanes
                a = [32767] * lanes
                b = [32767 if cycle % 101 == 1 else -32768] * lanes
                c = [32767] * lanes
                counts["saturating_inputs"] += 1
            else:
                x = [quantize(rng.uniform(-3.5, 3.5)) for _ in range(lanes)]
                a = [quantize(rng.uniform(0.0, 1.0)) for _ in range(lanes)]
                b = [quantize(rng.uniform(-1.5, 1.5)) for _ in range(lanes)]
                c = [quantize(rng.uniform(-1.5, 1.5)) for _ in range(lanes)]
            in_ready, out_valid, result = model.tick(
                rst_n=rst_n, in_valid=valid, out_ready=ready,
                sequence_start=seq, x=x, a=a, b=b, c=c)
            counts["resets"] += int(not rst_n)
            counts["stalled"] += int(rst_n and valid and not in_ready)
            counts["accepted"] += int(rst_n and valid and in_ready)
            counts["bubbles"] += int(rst_n and not valid)
            counts["sequence_starts"] += int(rst_n and valid and in_ready and seq)
            digits = (lanes * 16 + 3) // 4
            fields = [int(rst_n), int(valid), int(ready), int(seq),
                      *(f"{pack(vec):0{digits}x}" for vec in (x, a, b, c)),
                      int(in_ready), int(out_valid), f"{pack(result):0{digits}x}"]
            stream.write(" ".join(map(str, fields)) + "\n")
    if any(counts[k] == 0 for k in ("accepted", "stalled", "bubbles", "resets", "sequence_starts", "saturating_inputs")):
        raise AssertionError(f"Insufficient stimulus coverage: {counts}")
    print(f"vectors: cycles={cycles}, seed={seed}, lanes={lanes}, coverage={counts}")
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("build/scan_vectors.txt"))
    parser.add_argument("--cycles", type=int, default=2400)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--lanes", type=int, default=4)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    generate(args.output, args.cycles, args.seed, args.lanes)
