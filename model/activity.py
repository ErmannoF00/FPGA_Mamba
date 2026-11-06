"""Measure conservative, exact state-hold opportunities in the trained scan."""
import torch
from torch.nn import functional as F
from .network import DIM, STATE, KERNEL, TIME_STEPS, SCALE, LEVELS


def _pow2_ceiling_int(values: torch.Tensor) -> torch.Tensor:
    # int64 vectorized upper power-of-two bound, zero maps to zero.
    x = values.abs().to(torch.int64)
    y = torch.zeros_like(x)
    for b in range(16):
        power = 1 << b
        y = torch.where((x > (power >> 1)) & (x <= power), power, y)
    return y


def analyze(model, inputs: torch.Tensor) -> dict:
    model.eval()
    with torch.no_grad():
        batch=inputs.shape[0]
        h=inputs.new_zeros((batch,DIM,STATE))
        history=[inputs.new_zeros((batch,DIM)) for _ in range(KERNEL-1)]
        thresholds=inputs.new_tensor((-0.5,0,0.5));levels=inputs.new_tensor(LEVELS)
        total=ex_exact=ex_safe=ex_power=0
        mismatches=0
        state_nonzero=0
        for t in range(TIME_STEPS):
            x=model.q(inputs[:,t,:])
            projected=model.q(model.in_proj(x))
            taps=[projected]+history
            conv=model.q(sum(model.q(taps[k]*model.q(model.conv_weight[k])) for k in range(KERNEL))+model.q(model.conv_bias))
            u=model.q(F.relu(conv))
            b=model.q(model.b_proj(u))
            dt_logits=model.q(model.dt_proj(u))
            idx=torch.bucketize(dt_logits.detach(),thresholds)
            dt=levels[idx]+0.05*(dt_logits-dt_logits.detach())
            alpha,gain=model.dynamics(dt)
            drive=model.q(gain*b.unsqueeze(1))
            state_part=model.q(alpha*h)
            input_part=model.q(drive*u.unsqueeze(-1))
            h_new=model.q(state_part+input_part)
            a_q=torch.round(alpha*SCALE).to(torch.int64)
            h_q=torch.round(h*SCALE).to(torch.int64)
            drive_q=torch.round(drive*SCALE).to(torch.int64)
            u_q=torch.round(u*SCALE).to(torch.int64).unsqueeze(-1)
            exact=(torch.round(h_new*SCALE).to(torch.int64)==h_q)
            bound_full=(a_q-SCALE).abs()*h_q.abs()+drive_q.abs()*u_q.abs()
            cert_exact=(bound_full < (SCALE//2))
            # Upper bound where each variable's magnitude is rounded up to pow2.
            bound_shift=(a_q-SCALE).abs()*_pow2_ceiling_int(h_q) + drive_q.abs()*_pow2_ceiling_int(u_q)
            cert_shift=(bound_shift < (SCALE//2))
            mismatches+=int((cert_exact & ~exact).sum().item())
            mismatches+=int((cert_shift & ~exact).sum().item())
            total+=exact.numel(); ex_exact+=int(exact.sum().item())
            ex_safe+=int(cert_exact.sum().item());ex_power+=int(cert_shift.sum().item())
            state_nonzero+=int((h_q!=0).sum().item())
            h=h_new;history=[projected]+history[:-1]
        return {'transitions':total,'exact_noops':ex_exact,'exact_noop_pct':100*ex_exact/total,
            'certified_full_product_noops':ex_safe,'certified_shift_noops':ex_power,
            'certified_shift_pct':100*ex_power/total,
            'nonzero_states_before_update':state_nonzero, 'false_certifications':mismatches}
