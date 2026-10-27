import numpy as np

# Notation: T_a_b is a 4x4 rigid transform mapping homogeneous points from frame b
# into frame a:  p_a = T_a_b @ p_b.  Frames:
#   velo_i : velodyne sensor frame at scan i (x fwd, y left, z up; ground ~ -1.73 m)
#   cam_i  : left gray camera (cam0) frame at scan i (x right, y down, z fwd)
#   world  : velodyne frame at scan 0 (velo_0)


# Tr from calib.txt is T_cam_velo (velodyne -> camera, same rig, any scan i):
#   p_cam_i = Tr @ p_velo_i
# poses.txt row i is T_cam0_cami (camera i -> camera 0):
#   p_cam_0 = P_i @ p_cam_i
# Chain velo_i -> cam_i -> cam_0 -> velo_0:
#   p_velo_0 = Tr^-1 @ P_i @ Tr @ p_velo_i
# so T_world_velo_i = Tr^-1 @ P_i @ Tr, with world := velo_0.
def velo_poses_from_cam_poses(cam_poses: np.ndarray, tr_cam_velo: np.ndarray) -> np.ndarray:
    if cam_poses.ndim != 3 or cam_poses.shape[1:] != (4, 4):
        raise ValueError(f"cam_poses must be (M, 4, 4), got {cam_poses.shape}")
    if tr_cam_velo.shape != (4, 4):
        raise ValueError(f"tr_cam_velo must be (4, 4), got {tr_cam_velo.shape}")
    tr_velo_cam = np.linalg.inv(tr_cam_velo)
    # einsum: out_i = Tr^-1 @ P_i @ Tr for every i
    return np.einsum("ab,ibc,cd->iad", tr_velo_cam, cam_poses, tr_cam_velo)


# Apply rigid transform to (N, 3+) points: [x' y' z']^T = R @ [x y z]^T + t,
# where R = T[:3, :3], t = T[:3, 3]. Extra columns (e.g. reflectance) pass through.
def apply_transform(points: np.ndarray, t_dst_src: np.ndarray) -> np.ndarray:
    if t_dst_src.shape != (4, 4):
        raise ValueError(f"transform must be (4, 4), got {t_dst_src.shape}")
    out = np.array(points, dtype=np.float64, copy=True)
    out[:, :3] = points[:, :3] @ t_dst_src[:3, :3].T + t_dst_src[:3, 3]
    return out


# p_world = T_world_velo_i @ p_velo_i  (velodyne scan i -> world = velo_0)
def velo_to_world(points_velo: np.ndarray, t_world_velo: np.ndarray) -> np.ndarray:
    return apply_transform(points_velo, t_world_velo)


# p_cam_i = Tr @ p_velo_i  (Tr = T_cam_velo from calib.txt)
def velo_to_cam(points_velo: np.ndarray, tr_cam_velo: np.ndarray) -> np.ndarray:
    return apply_transform(points_velo, tr_cam_velo)
