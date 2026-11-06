"""Bit-exact Python recurrence for the precision_scan SystemVerilog module.

The projection, convolution, gate, and classification head are evaluated in
PyTorch. Only the recurrent state/output computation is reproduced in integer
arithmetic. This permits a genuine trained-model RTL scan replay without
misrepresenting it as full on-FPGA inference.
"""
from pathlib import Path
import torch
from torch.nn import functional as F
from .network import DIM, STATE, KERNEL, TIME_STEPS, LEVELS, SCALE


def round_shift(value, shift):
    if shift == 0:
        return value
    divisor=1<<shift
    floor=torch.div(value,divisor,rounding_mode='floor')
    rem=value-floor*divisor
    up=(rem>divisor//2) | ((rem==divisor//2)&((floor&1)!=0))
    return floor+up.to(torch.int64)


def qint(x):
    return torch.round(x*SCALE).to(torch.int64).clamp(-32768,32767)


def clip(x,bits):
    return x.clamp(-(1<<(bits-1)), (1<<(bits-1))-1)


def advance(state, alpha_q, excitation_q, c_q, skip_q, u_q, bits):
    """State integer is encoded with a per-lane fractional bit count."""
    bits=torch.as_tensor(bits,dtype=torch.int64)
    assert tuple(bits.shape)==(DIM,STATE)
    next_s=torch.zeros_like(state)
    full=torch.zeros_like(state)
    for d in range(DIM):
        for n in range(STATE):
            p=int(bits[d,n])
            candidate=round_shift(alpha_q[:,d,n]*state[:,d,n],10)
            candidate+=round_shift(excitation_q[:,d,n],10-p)
            next_s[:,d,n]=clip(candidate,6+p)
            full[:,d,n]=next_s[:,d,n] << (10-p)
    y=torch.zeros((state.shape[0],DIM),dtype=torch.int64)
    for d in range(DIM):
        for n in range(STATE):
            y[:,d]+=round_shift(c_q[:,n]*full[:,d,n],10)
        y[:,d]+=round_shift(skip_q[d]*u_q[:,d],10)
    return next_s,clip(y,16)


def run_hybrid(model, images, state_bits, capture=None, rtl_outputs=None):
    model.eval()
    if images.ndim!=3 or tuple(images.shape[1:])!=(TIME_STEPS,DIM):
        raise ValueError('images must be (batch,8,8)')
    batch=len(images)
    if rtl_outputs is not None:
        rtl_outputs=torch.as_tensor(rtl_outputs,dtype=torch.int64)
        if tuple(rtl_outputs.shape)!=(batch,TIME_STEPS,DIM):
            raise ValueError("rtl_outputs must have shape (batch,8,8)")
    if batch==0: raise ValueError('empty batch')
    bit_matrix=torch.as_tensor(state_bits,dtype=torch.int64)
    if tuple(bit_matrix.shape)!=(DIM,STATE) or bool(((bit_matrix<0)|(bit_matrix>10)).any()):
        raise ValueError('state precision must be (8,4) in range [0,10]')
    state=torch.zeros((batch,DIM,STATE),dtype=torch.int64)
    history=[images.new_zeros((batch,DIM)) for _ in range(KERNEL-1)]
    tokens=[]
    levels=images.new_tensor(LEVELS)
    thresholds=images.new_tensor((-0.5,0,0.5))
    for t in range(TIME_STEPS):
        x=model.q(images[:,t,:])
        projected=model.q(model.in_proj(x))
        gated=model.q(model.gate_proj(x))
        taps=[projected]+history
        conv=model.q(sum(model.q(taps[k]*model.q(model.conv_weight[k])) for k in range(KERNEL))+model.q(model.conv_bias))
        u=model.q(F.relu(conv))
        b=model.q(model.b_proj(u))
        c=model.q(model.c_proj(u))
        dt_logits=model.q(model.dt_proj(u))
        idx=torch.bucketize(dt_logits.detach(),thresholds)
        dt=levels[idx]+0.05*(dt_logits-dt_logits.detach())
        alpha,gain=model.dynamics(dt)
        excitation=model.q(model.q(gain*b.unsqueeze(1))*u.unsqueeze(-1))
        uq=qint(u);aq=qint(alpha);eq=qint(excitation);cq=qint(c)
        sq=qint(model.q(model.skip))
        state,yq=advance(state,aq,eq,cq,sq,uq,bit_matrix)
        if capture is not None:
            for sample in range(batch):
                capture(sample,t,uq[sample],aq[sample],eq[sample],cq[sample],sq,yq[sample])
        if rtl_outputs is not None:
            observed=rtl_outputs[:,t,:]
            mismatch=(observed!=yq).nonzero(as_tuple=False)
            if mismatch.numel():
                sample,channel=mismatch[0].tolist()
                raise ValueError(
                    f"RTL numerical mismatch: sample={sample}, token={t}, "
                    f"channel={channel}, rtl={int(observed[sample,channel])}, "
                    f"reference={int(yq[sample,channel])}"
                )
            # Use the values produced by the RTL simulator in the
            # original trained output projection and classification head.
            yq=observed
        y=yq.to(u.dtype)/SCALE
        gate=model.q(torch.clamp((gated+3.)/6.,0,1))
        result=model.q(model.out_proj(model.q(y*gate)))
        tokens.append(model.q(result+projected))
        history=[projected]+history[:-1]
    return model.head(torch.cat(tokens,dim=-1))
