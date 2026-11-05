"""Apply a finite Pi0 optimizer update, or discard an AMP overflow safely."""
import math


def optimizer_step(model, optimizer, scaler):
    import torch

    scale_before = float(scaler.get_scale())
    scaler.unscale_(optimizer)
    # FSDP returns a global norm: every rank makes the same skip decision.
    grad = model.clip_grad_norm_(1.)
    skipped = not bool(torch.isfinite(grad).item())
    if skipped:
        if not scaler.is_enabled():
            raise RuntimeError('Non-finite Pi0 gradient norm without AMP scaling')
        scale_after = scale_before * scaler.get_backoff_factor()
        if not math.isfinite(scale_after) or scale_after < 1e-8:
            raise RuntimeError('Pi0 gradient overflow persists at an unusable loss scale')
        # Do not call optimizer.step for any non-finite global norm, including
        # norm-reduction overflow. Explicit update clears this iteration's AMP
        # state without risking an update to weights or optimizer moments.
        scaler.update(new_scale=scale_after)
        state = scaler.state_dict()
        state['_growth_tracker'] = 0
        scaler.load_state_dict(state)
        optimizer.zero_grad(set_to_none=True)
    else:
        scaler.step(optimizer)
        scaler.update()
        scale_after = float(scaler.get_scale())
    return {
        'grad_norm': None if skipped else float(grad),
        'optimizer_step_skipped': skipped,
        'loss_scale_before': scale_before,
        'loss_scale': scale_after,
    }
