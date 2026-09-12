"""Quaternion <-> 6D continuous rotation representation (Zhou et al. 2019, "On the
Continuity of Rotation Representations in Neural Networks", https://arxiv.org/abs/1812.07035),
for pi05_extended_deltarot6d (see notes/training_runs.md). icl-dataset stores
quaternions scalar-first (qw, qx, qy, qz) -- every function here uses that convention,
NOT scipy.spatial.transform.Rotation's scalar-last (x, y, z, w).

The 6D representation is the first two ROWS of the rotation matrix, flattened (6
numbers instead of quaternion's 4 or Euler's 3) -- redundant but, unlike quaternions
(double cover: q and -q represent the same rotation) or Euler angles (gimbal lock),
continuous everywhere on SO(3). That matters for a flow-matching/regression loss:
a discontinuous target representation can make two visually-adjacent rotations map to
distant target vectors, producing large spurious loss spikes.

All functions are vectorized over leading batch/time dims -- operate on (..., 4) or
(..., 6) or (..., 3, 3) arrays.
"""

import numpy as np


def quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    """(..., 4) scalar-first quaternion -> (..., 3, 3) rotation matrix."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    r00 = 1 - 2 * (y**2 + z**2)
    r01 = 2 * (x * y - w * z)
    r02 = 2 * (x * z + w * y)
    r10 = 2 * (x * y + w * z)
    r11 = 1 - 2 * (x**2 + z**2)
    r12 = 2 * (y * z - w * x)
    r20 = 2 * (x * z - w * y)
    r21 = 2 * (y * z + w * x)
    r22 = 1 - 2 * (x**2 + y**2)
    return np.stack(
        [
            np.stack([r00, r01, r02], axis=-1),
            np.stack([r10, r11, r12], axis=-1),
            np.stack([r20, r21, r22], axis=-1),
        ],
        axis=-2,
    )


def rotmat_to_quat_wxyz(rotmat: np.ndarray) -> np.ndarray:
    """(..., 3, 3) rotation matrix -> (..., 4) scalar-first quaternion.

    Shepperd's method (numerically stable across all rotations, unlike the naive
    trace-based formula which divides by ~0 near a 180-degree rotation).
    """
    rotmat = np.asarray(rotmat, dtype=np.float64)
    m00, m01, m02 = rotmat[..., 0, 0], rotmat[..., 0, 1], rotmat[..., 0, 2]
    m10, m11, m12 = rotmat[..., 1, 0], rotmat[..., 1, 1], rotmat[..., 1, 2]
    m20, m21, m22 = rotmat[..., 2, 0], rotmat[..., 2, 1], rotmat[..., 2, 2]
    trace = m00 + m11 + m22

    def _safe_sqrt(x):
        return np.sqrt(np.clip(x, 0.0, None))

    # Candidate quaternion for each of the 4 numerically-stable cases; select per-element.
    t0 = _safe_sqrt(1 + trace) * 2  # 4w
    q0 = np.stack([t0 / 4, (m21 - m12) / t0, (m02 - m20) / t0, (m10 - m01) / t0], axis=-1)

    t1 = _safe_sqrt(1 + m00 - m11 - m22) * 2  # 4x
    q1 = np.stack([(m21 - m12) / t1, t1 / 4, (m01 + m10) / t1, (m02 + m20) / t1], axis=-1)

    t2 = _safe_sqrt(1 - m00 + m11 - m22) * 2  # 4y
    q2 = np.stack([(m02 - m20) / t2, (m01 + m10) / t2, t2 / 4, (m12 + m21) / t2], axis=-1)

    t3 = _safe_sqrt(1 - m00 - m11 + m22) * 2  # 4z
    q3 = np.stack([(m10 - m01) / t3, (m02 + m20) / t3, (m12 + m21) / t3, t3 / 4], axis=-1)

    case = np.argmax(np.stack([trace, m00, m11, m22], axis=-1), axis=-1)
    q = np.select(
        [case[..., None] == i for i in range(4)],
        [q0, q1, q2, q3],
    )
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def rotmat_to_rot6d(rotmat: np.ndarray) -> np.ndarray:
    """(..., 3, 3) rotation matrix -> (..., 6): first two rows, flattened."""
    rotmat = np.asarray(rotmat)
    return rotmat[..., :2, :].reshape(*rotmat.shape[:-2], 6)


def rot6d_to_rotmat(rot6d: np.ndarray) -> np.ndarray:
    """(..., 6) -> (..., 3, 3) via Gram-Schmidt (Zhou et al., Sec 3.3)."""
    rot6d = np.asarray(rot6d, dtype=np.float64)
    a1, a2 = rot6d[..., :3], rot6d[..., 3:]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    a2_proj = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 - a2_proj
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-2)


def quat_wxyz_to_rot6d(q: np.ndarray) -> np.ndarray:
    return rotmat_to_rot6d(quat_wxyz_to_rotmat(q))


def rot6d_to_quat_wxyz(rot6d: np.ndarray) -> np.ndarray:
    return rotmat_to_quat_wxyz(rot6d_to_rotmat(rot6d))
