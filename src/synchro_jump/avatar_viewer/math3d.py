"""Minimal quaternion helpers used by the avatar retargeting layer."""

from __future__ import annotations

import math

import numpy as np


def as_numpy_vector(values) -> np.ndarray:
    """Return one float vector with a stable `(n,)` shape."""

    return np.asarray(values, dtype=float).reshape((-1,))


def normalize_quaternion(quaternion_xyzw) -> np.ndarray:
    """Return one unit quaternion in `xyzw` convention."""

    quaternion = as_numpy_vector(quaternion_xyzw)
    if quaternion.shape != (4,):
        raise ValueError("Quaternion must contain exactly 4 components in xyzw convention")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError("Quaternion norm is too close to zero")
    return quaternion / norm


def quaternion_identity() -> np.ndarray:
    """Return the identity quaternion in `xyzw` convention."""

    return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)


def quaternion_conjugate(quaternion_xyzw) -> np.ndarray:
    """Return the conjugate of one quaternion."""

    x, y, z, w = normalize_quaternion(quaternion_xyzw)
    return np.array([-x, -y, -z, w], dtype=float)


def quaternion_inverse(quaternion_xyzw) -> np.ndarray:
    """Return the inverse of one unit quaternion."""

    return quaternion_conjugate(quaternion_xyzw)


def quaternion_multiply(left_xyzw, right_xyzw) -> np.ndarray:
    """Return the Hamilton product `left * right` in `xyzw` convention."""

    x1, y1, z1, w1 = normalize_quaternion(left_xyzw)
    x2, y2, z2, w2 = normalize_quaternion(right_xyzw)
    return normalize_quaternion(
        np.array(
            [
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            ],
            dtype=float,
        )
    )


def quaternion_from_axis_angle(axis_xyz, angle_rad: float) -> np.ndarray:
    """Return one quaternion rotating by `angle_rad` around `axis_xyz`."""

    axis = as_numpy_vector(axis_xyz)
    if axis.shape != (3,):
        raise ValueError("Axis must contain exactly 3 components")
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1e-12:
        raise ValueError("Axis norm is too close to zero")
    unit_axis = axis / axis_norm
    half_angle = 0.5 * float(angle_rad)
    return normalize_quaternion(
        np.concatenate((unit_axis * math.sin(half_angle), np.array([math.cos(half_angle)], dtype=float)))
    )


def quaternion_from_euler_xyz(angles_rad_xyz) -> np.ndarray:
    """Return one quaternion from intrinsic XYZ Euler rotations."""

    rx, ry, rz = as_numpy_vector(angles_rad_xyz)
    qx = quaternion_from_axis_angle((1.0, 0.0, 0.0), rx)
    qy = quaternion_from_axis_angle((0.0, 1.0, 0.0), ry)
    qz = quaternion_from_axis_angle((0.0, 0.0, 1.0), rz)
    return quaternion_multiply(quaternion_multiply(qx, qy), qz)


def quaternion_from_euler_deg_xyz(angles_deg_xyz) -> np.ndarray:
    """Return one quaternion from intrinsic XYZ Euler rotations in degrees."""

    return quaternion_from_euler_xyz(np.deg2rad(as_numpy_vector(angles_deg_xyz)))


def apply_frame_correction(rotation_xyzw, correction_xyzw) -> np.ndarray:
    """Express one local rotation in another frame basis.

    The `correction_xyzw` quaternion represents the basis change between the
    biomechanical frame and the target rig frame for one bone. The returned
    quaternion is `correction * rotation * correction^-1`.
    """

    correction = normalize_quaternion(correction_xyzw)
    return quaternion_multiply(
        quaternion_multiply(correction, rotation_xyzw),
        quaternion_inverse(correction),
    )


def local_from_global(parent_global_xyzw, child_global_xyzw) -> np.ndarray:
    """Return one child local rotation from global parent/child quaternions."""

    return quaternion_multiply(quaternion_inverse(parent_global_xyzw), child_global_xyzw)
