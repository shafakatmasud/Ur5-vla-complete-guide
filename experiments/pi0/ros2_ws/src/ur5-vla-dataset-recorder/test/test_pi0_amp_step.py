"""Numerical recovery must never apply non-finite gradients to parameters."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / 'scripts'))
from pi0_amp_step import optimizer_step


def setup_model(enabled=True):
    torch = pytest.importorskip('torch')
    class Model(torch.nn.Linear):
        def clip_grad_norm_(self, maximum):
            return torch.nn.utils.clip_grad_norm_(self.parameters(), maximum)
    model = Model(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    scaler = torch.amp.GradScaler('cpu', enabled=enabled, init_scale=128., growth_interval=2)
    return torch, model, optimizer, scaler


def test_overflow_preserves_weights_backs_off_and_next_finite_update_works():
    torch, model, optimizer, scaler = setup_model()
    before = {k:v.clone() for k,v in model.state_dict().items()}
    loss = model(torch.ones(1,2)).square().sum()
    scaler.scale(loss).backward()
    next(model.parameters()).grad.fill_(float('inf'))
    report = optimizer_step(model, optimizer, scaler)
    assert report['optimizer_step_skipped'] and report['grad_norm'] is None
    assert scaler.get_scale() == 64. and scaler.state_dict()['_growth_tracker'] == 0
    assert not optimizer.state
    for k,v in model.state_dict().items():
        torch.testing.assert_close(v,before[k],rtol=0,atol=0)
    scaler.scale(model(torch.ones(1,2)).square().sum()).backward()
    report = optimizer_step(model, optimizer, scaler)
    assert not report['optimizer_step_skipped']
    assert optimizer.state
    assert any(not torch.equal(v,before[k]) for k,v in model.state_dict().items())


def test_nonfinite_float32_gradient_still_stops_training():
    torch, model, optimizer, scaler = setup_model(enabled=False)
    model(torch.ones(1,2)).square().sum().backward()
    next(model.parameters()).grad.fill_(float('nan'))
    with pytest.raises(RuntimeError,match='without AMP'):
        optimizer_step(model, optimizer, scaler)


def test_finite_gradients_with_overflowed_global_norm_never_reach_optimizer():
    torch, model, optimizer, scaler = setup_model()
    before = {k:v.clone() for k,v in model.state_dict().items()}
    scaler.scale(model(torch.ones(1,2)).square().sum()).backward()
    for p in model.parameters():
        p.grad.fill_(1e30)
    report = optimizer_step(model, optimizer, scaler)
    assert report['optimizer_step_skipped']
    assert not optimizer.state
    for k,v in model.state_dict().items():
        torch.testing.assert_close(v,before[k],rtol=0,atol=0)
