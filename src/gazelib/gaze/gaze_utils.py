import os
import numpy as np
import imageio
import cv2
import h5py
import math
import torch

def pitchyaw_to_vector(pitchyaws):
    r"""Convert given yaw (:math:`\theta`) and pitch (:math:`\phi`) angles to unit gaze vectors.

    Args:
        pitchyaws: Input array of yaw and pitch angles, either numpy array or tensor.

    Returns:
        Output array of shape (n x 3) with 3D vectors per row, of the same type as the input.
    """
    if isinstance(pitchyaws, np.ndarray):
        return pitchyaw_to_vector_numpy(pitchyaws)
    elif isinstance(pitchyaws, torch.Tensor):
        return pitchyaw_to_vector_torch(pitchyaws)
    else:
        raise ValueError("Unsupported input type. Only numpy arrays and torch tensors are supported.")

def pitchyaw_to_vector_numpy(pitchyaws):
    n = pitchyaws.shape[0]
    sin = np.sin(pitchyaws)
    cos = np.cos(pitchyaws)
    out = np.empty((n, 3))
    out[:, 0] = np.multiply(cos[:, 0], sin[:, 1])
    out[:, 1] = sin[:, 0]
    out[:, 2] = np.multiply(cos[:, 0], cos[:, 1])
    return out

def pitchyaw_to_vector_torch(pitchyaws):
    n = pitchyaws.size()[0]
    sin = torch.sin(pitchyaws)
    cos = torch.cos(pitchyaws)
    out = torch.empty((n, 3), device=pitchyaws.device)
    out[:, 0] = torch.mul(cos[:, 0], sin[:, 1])
    out[:, 1] = sin[:, 0]
    out[:, 2] = torch.mul(cos[:, 0], cos[:, 1])
    return out

def vector_to_pitchyaw(vectors):
    """Convert given gaze vectors to pitch (theta) and yaw (phi) angles.

    Args:
        vectors: Input array of gaze vectors, either numpy array or tensor.

    Returns:
        Output array of shape (n x 2) with pitch and yaw angles, of the same type as the input.
    """
    if isinstance(vectors, np.ndarray):
        return vector_to_pitchyaw_numpy(vectors)
    elif isinstance(vectors, torch.Tensor):
        return vector_to_pitchyaw_torch(vectors)
    else:
        raise ValueError("Unsupported input type. Only numpy arrays and torch tensors are supported.")

def vector_to_pitchyaw_numpy(vectors):
    n = vectors.shape[0]
    vectors = vectors / np.linalg.norm(vectors, axis=1).reshape(n, 1)
    out = np.empty((n, 2))
    out[:, 0] = np.arcsin(vectors[:, 1])  # theta
    out[:, 1] = np.arctan2(vectors[:, 0], vectors[:, 2])  # phi
    return out

def vector_to_pitchyaw_torch(vectors):
    n = vectors.size()[0]
    vectors = vectors / torch.norm(vectors, dim=1).reshape(n, 1)
    out = torch.empty((n, 2), device=vectors.device)
    out[:, 0] = torch.asin(vectors[:, 1])  # theta
    out[:, 1] = torch.atan2(vectors[:, 0], vectors[:, 2])  # phi
    return out


def angular_error(a, b):
    """Calculate angular error (via cosine similarity)."""
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return angular_error_numpy(a, b)
    elif isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return angular_error_torch(a, b)
    else:
        raise ValueError("Input type mismatch. Both inputs should be either numpy arrays or torch tensors.")

def angular_error_numpy(a, b):
    """Calculate angular error for numpy arrays."""
    a = pitchyaw_to_vector(a) if a.shape[1] == 2 else a
    b = pitchyaw_to_vector(b) if b.shape[1] == 2 else b

    ab = np.sum(np.multiply(a, b), axis=1)
    a_norm = np.linalg.norm(a, axis=1)
    b_norm = np.linalg.norm(b, axis=1)

    # Avoid zero-values (to avoid NaNs)
    a_norm = np.clip(a_norm, a_min=1e-7, a_max=None)
    b_norm = np.clip(b_norm, a_min=1e-7, a_max=None)

    similarity = np.divide(ab, np.multiply(a_norm, b_norm))

    return np.arccos(similarity) * 180.0 / np.pi

def angular_error_torch(a, b):
    """Calculate angular error for torch tensors."""
    a = pitchyaw_to_vector(a) if a.size()[1] == 2 else a
    b = pitchyaw_to_vector(b) if b.size()[1] == 2 else b

    ab = torch.sum(a * b, dim=1)
    a_norm = torch.norm(a, dim=1)
    b_norm = torch.norm(b, dim=1)

    # Avoid zero-values (to avoid NaNs)
    a_norm = torch.clamp(a_norm, min=1e-7)
    b_norm = torch.clamp(b_norm, min=1e-7)

    similarity = ab / (a_norm * b_norm)

    return torch.acos(similarity) * 180.0 / np.pi






def cos_similarity(a, b):
    """Calculate angular error (via cosine similarity)."""
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return cos_similarity_numpy(a, b)
    elif isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return cos_similarity_torch(a, b)
    else:
        raise ValueError("Input type mismatch. Both inputs should be either numpy arrays or torch tensors.")

def cos_similarity_numpy(a, b):
    """Calculate angular error for numpy arrays."""
    a = pitchyaw_to_vector(a) if a.shape[1] == 2 else a
    b = pitchyaw_to_vector(b) if b.shape[1] == 2 else b

    ab = np.sum(np.multiply(a, b), axis=1)
    a_norm = np.linalg.norm(a, axis=1)
    b_norm = np.linalg.norm(b, axis=1)
    # Avoid zero-values (to avoid NaNs)
    a_norm = np.clip(a_norm, a_min=1e-7, a_max=None)
    b_norm = np.clip(b_norm, a_min=1e-7, a_max=None)
    similarity = np.divide(ab, np.multiply(a_norm, b_norm))
    similarity = np.clip(similarity, min=0., max=1.)
    return similarity

def cos_similarity_torch(a, b):
    """Calculate angular error for torch tensors."""
    a = pitchyaw_to_vector(a) if a.size()[1] == 2 else a
    b = pitchyaw_to_vector(b) if b.size()[1] == 2 else b

    ab = torch.sum(a * b, dim=1)
    a_norm = torch.norm(a, dim=1)
    b_norm = torch.norm(b, dim=1)

    # Avoid zero-values (to avoid NaNs)
    a_norm = torch.clamp(a_norm, min=1e-7)
    b_norm = torch.clamp(b_norm, min=1e-7)

    similarity = ab / (a_norm * b_norm)
    similarity = torch.clamp(similarity, min=0., max=1.)
    return similarity

