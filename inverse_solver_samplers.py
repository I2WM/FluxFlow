"""Training-free inverse-problem samplers for conditional OT-CFM models.

All samplers reuse a trained velocity checkpoint under the convention that
noise is at time zero and the clean image is at time one.
"""

import torch

from infer import apply_AT, downsample_blur


def _schedule(value, i, n, kind="linear_decay"):
    if kind == "constant":
        return value
    if kind == "linear_decay":
        return value * (1.0 - i / n)
    if kind == "cosine":
        return value * 0.5 * (1.0 + torch.cos(
            torch.tensor(torch.pi * i / n)).item())
    raise ValueError(kind)


def _A(x, kernel, scale):
    return downsample_blur(x, kernel, scale)


def _AT(y, kernel, scale, hr_size):
    return apply_AT(y, kernel, scale, adjoint_mode="nearest",
                    hr_size=hr_size, apply_scale=True)


def _batch_dot(x, y):
    return (x * y).flatten(1).sum(1)


def _cg(matvec, b, x0=None, max_iter=12, tol=1e-5):
    """Batched conjugate gradients for symmetric positive-definite systems."""
    x = torch.zeros_like(b) if x0 is None else x0.clone()
    r = b - matvec(x)
    p = r.clone()
    rr = _batch_dot(r, r)
    rr0 = rr.clamp_min(1e-30)
    for _ in range(max_iter):
        ap = matvec(p)
        alpha = rr / _batch_dot(p, ap).clamp_min(1e-30)
        alpha_v = alpha[:, None, None, None]
        x = x + alpha_v * p
        r = r - alpha_v * ap
        rr_new = _batch_dot(r, r)
        if torch.sqrt(rr_new / rr0).max().item() < tol:
            break
        beta = rr_new / rr.clamp_min(1e-30)
        p = r + beta[:, None, None, None] * p
        rr = rr_new
    return x


def regularized_pinv_residual(residual_lr, kernel, scale, hr_size,
                              reg=1e-2, cg_iters=12):
    """A^T (A A^T + reg I)^-1 residual, a stable approximation to A^+r."""
    def aat(z):
        return _A(_AT(z, kernel, scale, hr_size), kernel, scale) + reg * z

    z = _cg(aat, residual_lr, max_iter=cg_iters)
    return _AT(z, kernel, scale, hr_size)


def _endpoint(model, x, t, condition):
    """OT-CFM clean destination E[x_1|x_t] for this repo's time direction."""
    v = model(x, t, condition)
    t4 = t[:, None, None, None]
    return x + (1.0 - t4) * v, v


def sample_dps_style(model, desi, num_steps, device, *, hst_size,
                     psf_kernel, scale, guide_scale=0.5,
                     guide_schedule="linear_decay", **_):
    """DPS likelihood gradient through the FM clean-destination estimate."""
    b = desi.shape[0]
    x = torch.randn(b, 1, *hst_size, device=device)
    dt = 1.0 / num_steps
    for i in range(num_steps):
        t = torch.full((b,), i * dt, device=device)
        x = x.detach().requires_grad_(True)
        x1_hat, v = _endpoint(model, x, t, desi)
        residual = _A(x1_hat, psf_kernel, scale) - desi
        loss = 0.5 * residual.square().flatten(1).sum(1).mean()
        grad = torch.autograd.grad(loss, x)[0]
        eta = _schedule(guide_scale, i, num_steps, guide_schedule)
        x = (x + dt * v - eta * grad).detach()
    return x


def sample_pigdm_style(model, desi, num_steps, device, *, hst_size,
                       psf_kernel, scale, guide_scale=0.5,
                       guide_schedule="linear_decay", pinv_reg=1e-2,
                       cg_iters=12, **_):
    """PiGDM-style Jacobian guidance using a regularized full-operator A+."""
    b = desi.shape[0]
    x = torch.randn(b, 1, *hst_size, device=device)
    dt = 1.0 / num_steps
    for i in range(num_steps):
        t = torch.full((b,), i * dt, device=device)
        x = x.detach().requires_grad_(True)
        x1_hat, v = _endpoint(model, x, t, desi)
        residual = _A(x1_hat, psf_kernel, scale) - desi
        with torch.no_grad():
            pinv_r = regularized_pinv_residual(
                residual.detach(), psf_kernel, scale, hst_size,
                reg=pinv_reg, cg_iters=cg_iters)
        # J_{x1_hat}(x)^T A^+ residual without differentiating through CG.
        surrogate = (x1_hat * pinv_r).flatten(1).sum(1).mean()
        grad = torch.autograd.grad(surrogate, x)[0]
        eta = _schedule(guide_scale, i, num_steps, guide_schedule)
        x = (x + dt * v - eta * grad).detach()
    return x


@torch.no_grad()
def sample_flowdps(model, desi, num_steps, device, *, hst_size,
                   psf_kernel, scale, guide_scale=1.0,
                   pinv_reg=1e-2, cg_iters=12,
                   stochasticity=1.0, **_):
    """FlowDPS decomposition adapted to pixel-space conditional OT-CFM."""
    b = desi.shape[0]
    x = torch.randn(b, 1, *hst_size, device=device)
    dt = 1.0 / num_steps
    for i in range(num_steps):
        t_scalar = i * dt
        t_next = (i + 1) * dt
        t = torch.full((b,), t_scalar, device=device)
        x1_hat, v = _endpoint(model, x, t, desi)
        x0_hat = x - t_scalar * v

        residual = _A(x1_hat, psf_kernel, scale) - desi
        dc = regularized_pinv_residual(
            residual, psf_kernel, scale, hst_size,
            reg=pinv_reg, cg_iters=cg_iters)
        # Original FlowDPS interpolation weight is the current noise level.
        gamma = 1.0 - t_scalar
        x1_refined = x1_hat - guide_scale * gamma * dc

        # Stochastic reprojection of the noise component.  At late time the
        # newly drawn noise receives more weight, as in FlowDPS Algorithm 1.
        mix = min(max(stochasticity * t_next, 0.0), 1.0)
        noise_component = ((1.0 - mix) ** 0.5 * x0_hat
                           + mix ** 0.5 * torch.randn_like(x0_hat))
        x = (1.0 - t_next) * noise_component + t_next * x1_refined
    return x


@torch.no_grad()
def sample_flower(model, desi, num_steps, device, *, hst_size,
                  psf_kernel, scale, noise_std=0.05, cg_iters=12,
                  flower_gamma=0.0, **_):
    """Flower Algorithm 1 with CG proximal refinement (conditional adaptation)."""
    b = desi.shape[0]
    x = torch.randn(b, 1, *hst_size, device=device)
    dt = 1.0 / num_steps
    sigma2 = max(noise_std ** 2, 1e-8)
    for i in range(num_steps):
        t_scalar = i * dt
        t_next = (i + 1) * dt
        t = torch.full((b,), t_scalar, device=device)
        x1_hat, _ = _endpoint(model, x, t, desi)
        nu = (1.0 - t_scalar) / (
            t_scalar ** 2 + (1.0 - t_scalar) ** 2) ** 0.5
        weight = (nu ** 2) / sigma2

        def lhs(z):
            return z + weight * _AT(
                _A(z, psf_kernel, scale), psf_kernel, scale, hst_size)

        rhs = x1_hat + weight * _AT(
            desi, psf_kernel, scale, hst_size)
        mu = _cg(lhs, rhs, x0=x1_hat, max_iter=cg_iters)

        if flower_gamma:
            # Diagonal approximation to Flower's optional covariance sample.
            kappa_scale = (1.0 / (nu ** -2 + sigma2 ** -1)) ** 0.5
            mu = mu + flower_gamma * kappa_scale * torch.randn_like(mu)

        # Flower draws a fresh source sample at every re-projection.
        eps = torch.randn_like(x)
        x = (1.0 - t_next) * eps + t_next * mu
    return x


SAMPLERS = {
    "dps_style": sample_dps_style,
    "pigdm_style": sample_pigdm_style,
    "flowdps": sample_flowdps,
    "flower": sample_flower,
}
