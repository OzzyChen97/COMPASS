"""HumanML3D 263D feature extraction from 22-joint global positions.

The implementation follows the HumanML3D/TOMATO preprocessing convention used
by the evaluator and T2M-GPT representation.
"""

import numpy as np
import torch

from utils.paramUtil import t2m_kinematic_chain, t2m_raw_offsets
from utils.quaternion import (
    qbetween_np,
    qfix,
    qinv_np,
    qmul_np,
    qrot_np,
    quaternion_to_cont6d_np,
)
from utils.skeleton import Skeleton


FACE_JOINTS = [2, 1, 17, 16]
FEET_RIGHT = [8, 11]
FEET_LEFT = [7, 10]
LOWER_LEG_JOINTS = [5, 8]


def target_offsets(reference_joints: np.ndarray) -> torch.Tensor:
    skeleton = Skeleton(torch.from_numpy(t2m_raw_offsets), t2m_kinematic_chain, "cpu")
    return skeleton.get_offsets_joints(
        torch.from_numpy(np.asarray(reference_joints)[0]).float()
    )


def _normalize(vector: np.ndarray, axis=-1) -> np.ndarray:
    norm = np.linalg.norm(vector, axis=axis, keepdims=True)
    return vector / np.maximum(norm, 1e-8)


def uniform_skeleton(positions: np.ndarray, offsets: torch.Tensor) -> np.ndarray:
    skeleton = Skeleton(torch.from_numpy(t2m_raw_offsets), t2m_kinematic_chain, "cpu")
    source_offsets = skeleton.get_offsets_joints(
        torch.from_numpy(positions[0]).float()
    ).numpy()
    target = offsets.numpy()
    source_leg = sum(np.abs(source_offsets[index]).max() for index in LOWER_LEG_JOINTS)
    target_leg = sum(np.abs(target[index]).max() for index in LOWER_LEG_JOINTS)
    if source_leg <= 1e-8:
        raise ValueError("Degenerate source skeleton leg length")
    root = positions[:, 0] * (target_leg / source_leg)
    quaternions = skeleton.inverse_kinematics_np(positions, FACE_JOINTS)
    skeleton.set_offset(offsets)
    return skeleton.forward_kinematics_np(quaternions, root)


def process_positions(
    positions: np.ndarray,
    offsets: torch.Tensor,
    feet_threshold: float = 0.002,
) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.float32)
    if positions.ndim != 3 or positions.shape[1:] != (22, 3):
        raise ValueError(f"Expected [T, 22, 3] joints, got {positions.shape}")
    if len(positions) < 2 or not np.isfinite(positions).all():
        raise ValueError("Motion must contain at least two finite frames")

    positions = uniform_skeleton(positions, offsets)
    positions[:, :, 1] -= positions[:, :, 1].min()

    root_initial = positions[0]
    positions -= root_initial[0] * np.array([1.0, 0.0, 1.0])
    across = (
        root_initial[FACE_JOINTS[0]] - root_initial[FACE_JOINTS[1]]
        + root_initial[FACE_JOINTS[2]] - root_initial[FACE_JOINTS[3]]
    )
    forward = _normalize(np.cross(np.array([[0.0, 1.0, 0.0]]), across))
    root_initial_quaternion = qbetween_np(forward, np.array([[0.0, 0.0, 1.0]]))
    initial_rotation = np.ones(positions.shape[:-1] + (4,)) * root_initial_quaternion
    positions = qrot_np(initial_rotation, positions)
    global_positions = positions.copy()

    def foot_contact(indices):
        velocity = np.square(positions[1:, indices] - positions[:-1, indices]).sum(axis=-1)
        return (velocity < feet_threshold).astype(np.float32)

    feet_left = foot_contact(FEET_LEFT)
    feet_right = foot_contact(FEET_RIGHT)

    skeleton = Skeleton(torch.from_numpy(t2m_raw_offsets), t2m_kinematic_chain, "cpu")
    quaternions = skeleton.inverse_kinematics_np(
        positions, FACE_JOINTS, smooth_forward=True
    )
    quaternions = qfix(quaternions)
    continuous_6d = quaternion_to_cont6d_np(quaternions)
    root_rotation = quaternions[:, 0].copy()

    velocity = positions[1:, 0] - positions[:-1, 0]
    velocity = qrot_np(root_rotation[1:], velocity)
    root_velocity = qmul_np(root_rotation[1:], qinv_np(root_rotation[:-1]))

    local_positions = positions.copy()
    local_positions[..., 0] -= local_positions[:, 0:1, 0]
    local_positions[..., 2] -= local_positions[:, 0:1, 2]
    local_positions = qrot_np(
        np.repeat(root_rotation[:, None], local_positions.shape[1], axis=1),
        local_positions,
    )

    root_height = local_positions[:, 0, 1:2]
    root_data = np.concatenate(
        [np.arcsin(np.clip(root_velocity[:, 2:3], -1.0, 1.0)),
         velocity[:, [0, 2]], root_height[:-1]],
        axis=-1,
    )
    ric = local_positions[:, 1:].reshape(len(local_positions), -1)
    rotations = continuous_6d[:, 1:].reshape(len(continuous_6d), -1)
    local_velocity = qrot_np(
        np.repeat(root_rotation[:-1, None], global_positions.shape[1], axis=1),
        global_positions[1:] - global_positions[:-1],
    ).reshape(len(global_positions) - 1, -1)

    features = np.concatenate(
        [root_data, ric[:-1], rotations[:-1], local_velocity, feet_left, feet_right],
        axis=-1,
    ).astype(np.float32)
    if features.shape[1] != 263 or not np.isfinite(features).all():
        raise ValueError(f"Invalid HumanML3D features: {features.shape}")
    return features


def humanml_mean_std(motions, feat_bias: float = 5.0):
    total = 0
    total_sum = np.zeros(263, dtype=np.float64)
    total_square = np.zeros(263, dtype=np.float64)
    for motion in motions:
        values = np.asarray(motion, dtype=np.float64)
        total += len(values)
        total_sum += values.sum(axis=0)
        total_square += np.square(values).sum(axis=0)
    if total == 0:
        raise ValueError("Cannot compute statistics from an empty motion collection")
    mean = total_sum / total
    variance = np.maximum(total_square / total - np.square(mean), 0.0)
    raw_std = np.sqrt(variance)
    std = raw_std.copy()

    groups = [
        (0, 1, feat_bias),
        (1, 3, feat_bias),
        (3, 4, feat_bias),
        (4, 67, 1.0),
        (67, 193, 1.0),
        (193, 259, 1.0),
        (259, 263, feat_bias),
    ]
    for start, end, divisor in groups:
        value = max(float(raw_std[start:end].mean()) / divisor, 1e-8)
        std[start:end] = value
    return mean.astype(np.float32), std.astype(np.float32)
