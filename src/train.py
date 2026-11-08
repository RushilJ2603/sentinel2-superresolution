# Training loop, and the gradient probe that analysis A1 is built on.
#
# A1 is the reason this file is interesting. Normally you sum the two loss
# terms into one scalar and call backward() once. Here, every probe_every
# steps, the terms are differentiated SEPARATELY and the angle between the
# two gradient vectors is recorded:
#
#   cos ~ 1   the aux term pushes the weights exactly where L1 already pushes
#   cos ~ 0   it pulls sideways, carrying something L1 doesn't have
#   cos < 0   the two are fighting each other
#
# That turns "does the extra loss term actually do anything" from an
# argument into a measurement. The probe runs in fp32 on its own forward
# pass, so autocast rounding can't contaminate it and the optimiser step
# is left completely alone.
#
# The probe now covers R3 as well as R2, which is the more interesting case:
# SAM measured 0.81 (nearly redundant with L1, as the analysis predicted),
# and the consistency term should measure LOWER because it is the one term
# in the project that looks at the INPUT rather than the target.

import copy
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from model import SRNet, count_params


def _log(*a):
    print(*a, flush=True)          # unbuffered: a 14-minute run needs to be watchable


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def _flat_grad(loss, params, retain):
    # d(loss)/d(params), every parameter flattened into one long vector
    g = torch.autograd.grad(loss, params, retain_graph=retain,
                            allow_unused=True)
    return torch.cat([(gi if gi is not None else torch.zeros_like(p)).reshape(-1)
                      for gi, p in zip(g, params)])


def _f32(out):
    """Cast model output to fp32 before the loss.

    Matters most for R4: the Laplace NLL evaluates exp(-log_b) with log_b
    near -5.3, so b ~ 0.005 and the residual gets multiplied by ~200. In
    bfloat16 that carries roughly 0.4% relative error, which is fine for a
    gradient but not for a run whose entire purpose is CALIBRATION. The
    forward pass still runs in bf16; only the loss arithmetic is promoted.
    """
    if isinstance(out, (tuple, list)):
        return tuple(o.float() for o in out)
    return out.float()


def grad_cosine(model, spec, lr, hr):
    # None for runs with no auxiliary term to compare against
    if spec.aux_name is None:
        return None
    params = [p for p in model.parameters() if p.requires_grad]
    model.zero_grad(set_to_none=True)
    main, aux = spec.parts(_f32(model(lr)), hr, lr)
    g_main = _flat_grad(main, params, True)
    g_aux = _flat_grad(aux, params, False)
    cos = F.cosine_similarity(g_main[None], g_aux[None], dim=1).item()
    model.zero_grad(set_to_none=True)
    return cos


def train_one(spec, train_loader, val_loader, device, steps=12000, lr0=1e-4,
              seed=0, blocks=16, channels=64, probe_every=100, val_every=500,
              amp=True, log=_log, arch=None, residual=False):
    """`arch` selects a comparison architecture from baselines.py.

    None keeps the project's own SRNet. The baselines train under exactly
    this function, so they get the same optimiser, schedule, step count,
    precision and seeds: an under-tuned baseline is not a baseline.
    """
    set_seed(seed)
    if arch:
        from baselines import build
        assert not spec.needs_sigma, \
            "the uncertainty head is defined on SRNet only"
        model = build(arch).to(device)
    else:
        model = SRNet(in_ch=4, c=channels, n_blocks=blocks,
                      predict_sigma=spec.needs_sigma,
                      global_residual=residual).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps,
                                                       eta_min=1e-6)
    hist = {"step": [], "train": [], "val_step": [], "val_l1": [],
            "probe_step": [], "grad_cos": [], "aux": spec.aux_name,
            "params": count_params(model), "kind": spec.kind,
            "arch": arch or ("srnet_res" if residual else "srnet"),
            "label": spec.label()}
    best = {"val_l1": math.inf, "state": None, "step": -1}

    it = iter(train_loader)
    run, nrun, t0 = 0.0, 0, time.time()
    for step in range(1, steps + 1):
        try:
            lo, hi = next(it)
        except StopIteration:                      # restart the epoch
            it = iter(train_loader)
            lo, hi = next(it)
        lo = lo.to(device, non_blocking=True)
        hi = hi.to(device, non_blocking=True)

        model.train()
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            out = model(lo)
        loss = spec.total(_f32(out), hi, lo)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run += loss.item()
        nrun += 1

        if step % probe_every == 0:
            hist["step"].append(step)
            hist["train"].append(run / max(nrun, 1))
            run, nrun = 0.0, 0
            c = grad_cosine(model, spec, lo, hi)   # fp32, separate forward
            if c is not None:
                hist["probe_step"].append(step)
                hist["grad_cos"].append(c)

        if step % val_every == 0 or step == steps:
            model.eval()
            tot, n = 0.0, 0
            with torch.no_grad():
                for vlo, vhi in val_loader:
                    vlo, vhi = vlo.to(device), vhi.to(device)
                    # Early stopping is selected on L1 of the MEAN for every
                    # run, including R4. Selecting R4 on its own NLL would
                    # compare runs on different scales and let the
                    # uncertainty head buy a better checkpoint by widening
                    # its intervals rather than by predicting better.
                    pred = spec.mean_of(model(vlo)).float()
                    tot += F.l1_loss(pred.clamp(0, 1), vhi).item() * vlo.shape[0]
                    n += vlo.shape[0]
            v = tot / max(n, 1)
            hist["val_step"].append(step)
            hist["val_l1"].append(v)
            tag = ""
            if v < best["val_l1"]:                 # early stopping: keep best
                best.update(val_l1=v, step=step,
                            state=copy.deepcopy(model.state_dict()))
                tag = "  *best"
            gc = (" gcos %.3f" % hist["grad_cos"][-1]) if hist["grad_cos"] else ""
            log("   step %5d/%d  train %.5f  val_l1 %.5f%s  %.0fs%s"
                % (step, steps, hist["train"][-1] if hist["train"] else float("nan"),
                   v, gc, time.time() - t0, tag))

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    hist["best_val_l1"] = best["val_l1"]
    hist["best_step"] = best["step"]
    hist["seconds"] = time.time() - t0
    return model, hist
