"""Bit-accurate software reference for the RTL recurrence and its handshake."""

from dataclasses import dataclass, field

WIDTH = 16
FRAC = 12
LANES = 4
D_Q = 1 << (FRAC - 1)
LIMIT = 1 << (WIDTH - 1)
MASK = (1 << WIDTH) - 1


def saturate(value: int) -> int:
    return max(-LIMIT, min(LIMIT - 1, value))


def quantize(value: float) -> int:
    return saturate(round(value * (1 << FRAC)))


def pack(values: list[int]) -> int:
    return sum((value & MASK) << (WIDTH * i) for i, value in enumerate(values))


def unpack(packed: int, lanes: int = LANES) -> list[int]:
    values = [(packed >> (WIDTH * i)) & MASK for i in range(lanes)]
    return [value - (1 << WIDTH) if value & LIMIT else value for value in values]


@dataclass
class ScanModel:
    lanes: int = LANES
    state: list[int] = field(default_factory=lambda: [0] * LANES)
    output: list[int] = field(default_factory=lambda: [0] * LANES)
    out_valid: bool = False

    def __post_init__(self) -> None:
        if self.lanes <= 0:
            raise ValueError("lanes must be positive")
        self.state = [0] * self.lanes
        self.output = [0] * self.lanes

    def tick(self, *, rst_n: bool, in_valid: bool, out_ready: bool,
             sequence_start: bool, x: list[int], a: list[int],
             b: list[int], c: list[int]) -> tuple[bool, bool, list[int]]:
        """Return pre-edge (in_ready, out_valid, output), then clock the edge."""
        assert all(len(v) == self.lanes for v in (x, a, b, c))
        in_ready = not self.out_valid or out_ready
        observation = (in_ready, self.out_valid, self.output.copy())
        if not rst_n:
            # Asynchronous reset is visible before the next rising edge.
            observation = (True, False, [0] * self.lanes)
            self.state = [0] * self.lanes
            self.output = [0] * self.lanes
            self.out_valid = False
        elif in_ready:
            self.out_valid = in_valid
            if in_valid:
                prev = [0] * self.lanes if sequence_start else self.state
                next_state = [saturate((ai * hi + bi * xi) >> FRAC)
                              for ai, hi, bi, xi in zip(a, prev, b, x)]
                self.output = [saturate((ci * hi + D_Q * xi) >> FRAC)
                               for ci, hi, xi in zip(c, next_state, x)]
                self.state = next_state
        return observation
