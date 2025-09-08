import torch


def underdamped_oscillator(m: float, mu: float, ke: float, x0: float, v0: float, t: float) -> torch.Tensor:
    """Compute the exact solution of an underdamped harmonic oscillator.

    Solves the differential equation for an underdamped mass-spring-damper
    system with given initial position and velocity.

    Args:
        m (float): Mass of the body.
        mu (float): Friction (damping) coefficient.
        ke (float): Elastic (spring) coefficient.
        x0 (float): Initial displacement.
        v0 (float): Initial velocity.
        t (float): Time vector (torch.Tensor or float).

    Returns:
        torch.Tensor: Displacement of the oscillator at times `t`.

    Raises:
        ValueError: If the system is not underdamped (i.e., delta >= omega0).

    """
    t_mu = torch.tensor(mu)
    t_m = torch.tensor(m)
    t_ke = torch.tensor(ke)
    t_x0 = torch.tensor(x0)
    t_v0 = torch.tensor(v0)

    delta = t_mu / (2 * t_m)
    omega0 = torch.sqrt(t_ke / t_m)

    print("Underdamped oscillator motion parameters:")
    print(f"delta = {delta}, omega0 = {omega0}")

    if delta >= omega0:
        err_str = f"System is not underdamped: delta={delta.item():.4f}, omega0={omega0.item():.4f}."
        raise ValueError(err_str)

    omega = torch.sqrt(omega0**2 - delta**2)

    if t_x0 != torch.tensor(0):
        phi = torch.atan(- (t_v0 + delta * t_x0) / (omega * t_x0))
        a = t_x0 / 2*torch.cos(phi)
    elif t_v0 != torch.tensor(0):
        phi = torch.pi / torch.tensor(2)
        a = - t_v0 / (2 * omega)
    else:
        phi = torch.tensor(0)
        a = torch.tensor(0)

    return torch.exp(-delta * t) * 2 * a * torch.cos(omega * t + phi)
