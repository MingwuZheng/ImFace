import torch

@torch.jit.script
def skew(w):
    """Build a skew matrix ("cross product matrix") for vector w.
    Modern Robotics Eqn 3.30.
    Args:
      w: (B,N,3) A 3-vector
    Returns:
      W: (B,N,3,3) A skew matrix such that W @ v == w x v
    """
    B, N, _ = w.size()
    W = torch.zeros(B, N, 3, 3).float().to(w.device)
    W[:, :, 0, 1] = -w[:, :, 2]
    W[:, :, 0, 2] = w[:, :, 1]
    W[:, :, 1, 0] = w[:, :, 2]
    W[:, :, 1, 2] = -w[:, :, 0]
    W[:, :, 2, 0] = -w[:, :, 1]
    W[:, :, 2, 1] = w[:, :, 0]
    return W


def rp_to_se3(R: torch.tensor, p: torch.tensor) -> torch.tensor:
    """Rotation and translation to homogeneous transform.
    Args:
      R: (B,N,3,3) An orthonormal rotation matrix.
      p: (B,N,3) A 3-vector representing an offset.
    Returns:
      X: (B,N,4,4) The homogeneous transformation matrix described by rotating by R
        and translating by p.
    """
    B, N, _ = p.size()
    p = torch.reshape(p, (B, N, 3, 1))
    h = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).to(R.device)  # (1,4)
    return torch.cat([torch.cat([R, p], dim=-1), h[None, None, ...].repeat(B, N, 1, 1)], dim=-2)


def exp_so3(w: torch.tensor, theta: torch.tensor) -> torch.tensor:
    """Exponential map from Lie algebra so3 to Lie group SO3.
    Modern Robotics Eqn 3.51, a.k.a. Rodrigues' formula.
    Args:
      w: (B,N,3) An axis of rotation.
      theta: (B,N) An angle of rotation.
    Returns:
      R: (B,N,3,3) An orthonormal rotation matrix representing a rotation of
        magnitude theta about axis w.
    """
    B, N, _ = w.size()
    theta = theta[..., None, None].repeat(1, 1, 3, 3)
    W = skew(w)
    # WxW = torch.einsum('bnhi,bniw->bnhw', W, W)
    return torch.eye(3)[None, None, ...].repeat(B, N, 1, 1).to(w.device) + torch.sin(theta) * W + (
            1.0 - torch.cos(theta)) * W @ W


def exp_se3(S: torch.tensor, theta: torch.tensor) -> torch.tensor:
    """Exponential map from Lie algebra so3 to Lie group SO3.
    Modern Robotics Eqn 3.88.
    Args:
      S: (B,N,6) A screw axis of motion.
      theta: (B,N) Magnitude of motion.
    Returns:
      a_X_b: (B,N,4,4) The homogeneous transformation matrix attained by integrating
        motion of magnitude theta about S for one second.
    """
    B, N, _ = S.size()
    w, v = torch.split(S, 3, dim=-1)
    W = skew(w)
    R = exp_so3(w, theta)

    # assert torch.isnan(W).sum() == 0, print(W)
    # assert torch.isnan(R).sum() == 0, print(R)

    theta = theta[..., None, None].repeat(1, 1, 3, 3)

    # assert torch.isnan(theta).sum() == 0, print(theta)

    p = (theta * (torch.eye(3)[None, None, ...].repeat(B, N, 1, 1).to(S.device)) + (1.0 - torch.cos(theta)) * W +
         (theta - torch.sin(theta)) * W @ W) @ v[..., None]
    return rp_to_se3(R, p.squeeze(-1))


def to_homogenous(v):
    return torch.cat([v, torch.ones_like(v[..., :1]).to(v.device)], dim=-1)


def from_homogenous(v):
    return v[..., :3] / v[..., -1:]
