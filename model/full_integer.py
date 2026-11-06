"""Explicit signed fixed-point inference for the trained selective classifier.

This model defines the hardware arithmetic. It never calls the neural network's
forward() method and uses only exported integer coefficients after loading.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
from typing import Callable

Q = 10
S = 1 << Q
DIM = 8
STATES = 4
STEPS = 8


def rounded_div(value: int, divisor: int) -> int:
    """Round rational value/divisor to nearest integer, ties to even."""
    assert divisor > 0
    floor, remainder = divmod(int(value), int(divisor))
    return floor + int(2 * remainder > divisor or
                       (2 * remainder == divisor and floor % 2 != 0))


def clip16(v: int) -> int:
    return min(32767, max(-32768, int(v)))


def mulq(a: int, b: int) -> int:
    return clip16(rounded_div(int(a) * int(b), S))


def linear(x: list[int], weights: list[list[int]], bias: list[int]) -> list[int]:
    return [clip16(rounded_div(sum(int(w) * int(v) for w, v in zip(row, x)), S) + b)
            for row, b in zip(weights, bias)]


def quant(v: float) -> int:
    return clip16(round(float(v) * S))


def prepare(checkpoint: str | Path, bits_path: str | Path):
    import torch
    ckpt = torch.load(str(checkpoint), map_location='cpu', weights_only=True)
    if ckpt['scheme'] != 'bilinear':
        raise ValueError('The complete RTL prototype requires the bilinear checkpoint')
    weights = ckpt['weights']
    def t(name):
        return weights[name].detach().cpu().numpy()
    def q_matrix(name):
        return [[quant(a) for a in row] for row in t(name)]
    def q_vector(name):
        return [quant(v) for v in t(name)]
    params = {}
    for key in ('in_proj','gate_proj','b_proj','c_proj','dt_proj','out_proj','head'):
        params[key + '_w'] = q_matrix(key+'.weight')
        params[key + '_b'] = q_vector(key+'.bias')
    params['conv_w'] = q_matrix('conv_weight')
    params['conv_b'] = q_vector('conv_bias')
    params['skip'] = q_vector('skip')
    logits = t('a_logits')
    table = []
    for level in (0.125, 0.25, 0.5, 1.0):
        alphas, gains = [], []
        for d in range(DIM):
            aa, gg = [], []
            for n in range(STATES):
                a = -(0.05 + 1.45/(1.0+math.exp(-float(logits[d,n]))))
                den = 1 - level*a/2
                aa.append(quant((1 + level*a/2)/den))
                gg.append(quant(level/den))
            alphas.append(aa)
            gains.append(gg)
        table.append({'alpha': alphas, 'gain': gains})
    params['table'] = table
    bits = json.loads(Path(bits_path).read_text())['bits_per_state']
    if len(bits) != 8 or any(len(row) != 4 or any(not 0 <= p <= 10 for p in row) for row in bits):
        raise ValueError('Bad state bit map')
    params['bits'] = bits
    params['format'] = 'signed Q6.10; nearest-even; saturating 16-bit; 32 states'
    return params


def forward(image, params, trace: Callable | None = None):
    """Takes an 8x8 image in signed Q6.10; returns 10 signed Q6.10 logits."""
    if len(image) != STEPS or any(len(row) != DIM for row in image):
        raise ValueError('Input must contain eight 8-pixel tokens')
    history = [[0]*DIM for _ in range(2)]
    state = [[0]*STATES for _ in range(DIM)]
    tokens = []
    for t in range(STEPS):
        x = [int(v) for v in image[t]]
        projected = linear(x, params['in_proj_w'], params['in_proj_b'])
        gated = linear(x, params['gate_proj_w'], params['gate_proj_b'])
        taps = [projected] + history
        conv = [clip16(sum(mulq(taps[k][d], params['conv_w'][k][d])
                           for k in range(3)) + params['conv_b'][d]) for d in range(DIM)]
        u = [max(0, v) for v in conv]
        b = linear(u, params['b_proj_w'], params['b_proj_b'])
        c = linear(u, params['c_proj_w'], params['c_proj_b'])
        dt = linear(u, params['dt_proj_w'], params['dt_proj_b'])
        dt_idx = [0 if v <= -512 else 1 if v <= 0 else 2 if v <= 512 else 3
                  for v in dt]
        output = []
        for d in range(DIM):
            accum = 0
            for n in range(STATES):
                a = params['table'][dt_idx[d]]['alpha'][d][n]
                gain = params['table'][dt_idx[d]]['gain'][d][n]
                e = mulq(mulq(gain, b[n]), u[d])
                p = params['bits'][d][n]
                updated = rounded_div(a * state[d][n], S) + rounded_div(e, 1 << (Q-p))
                lo, hi = -(1 << (5+p)), (1 << (5+p))-1
                state[d][n] = min(hi, max(lo, updated))
                full = state[d][n] << (Q-p)
                accum += rounded_div(c[n] * full, S)
            accum += rounded_div(params['skip'][d] * u[d], S)
            output.append(clip16(accum))
        gate = [min(S, max(0, rounded_div(v+3*S, 6))) for v in gated]
        gated_output = [mulq(y,g) for y,g in zip(output,gate)]
        result = linear(gated_output, params['out_proj_w'], params['out_proj_b'])
        token = [clip16(result[d] + projected[d]) for d in range(DIM)]
        tokens.extend(token)
        if trace is not None:
            trace(t, {'projected': projected,'gated': gated,'u':u,
                      'b':b,'c':c,'dt':dt,'idx':dt_idx,'y':output,
                      'gate':gate,'token':token})
        history = [projected] + history[:1]
    return linear(tokens, params['head_w'], params['head_b'])


def pack_image(image):
    return ''.join(f'{int(v)&0xffff:04x}' for row in reversed(image)
                   for v in reversed(row))


def pack_scores(scores):
    return ''.join(f'{int(v)&0xffff:04x}' for v in reversed(scores))
