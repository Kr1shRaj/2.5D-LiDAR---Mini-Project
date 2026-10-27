import numpy as np
import pytest

from src.io.kitti_loader import load_calib, load_labels, load_poses, load_scan
from src.preprocess.filters import ego_vehicle_mask, preprocess_mask, range_gate_mask
from src.preprocess.transforms import velo_poses_from_cam_poses, velo_to_world


# random rigid 4x4 from seeded rng
def random_rigid(rng: np.random.Generator) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    t = np.eye(4)
    t[:3, :3] = q
    t[:3, 3] = rng.normal(size=3)
    return t


def test_scan_and_labels_roundtrip(tmp_path) -> None:
    pts = np.array([[1.0, 2.0, -1.73, 0.5], [3.0, -4.0, 0.2, 0.9]], dtype=np.float32)
    pts.tofile(tmp_path / "000000.bin")
    raw = np.array([(7 << 16) | 40, (0 << 16) | 252], dtype=np.uint32)
    raw.tofile(tmp_path / "000000.label")

    scan = load_scan(tmp_path / "000000.bin")
    sem, inst = load_labels(tmp_path / "000000.label", n_points=len(scan))
    assert np.array_equal(scan, pts)
    assert sem.tolist() == [40, 252]
    assert inst.tolist() == [7, 0]


def test_label_count_mismatch_rejected(tmp_path) -> None:
    np.zeros(3, dtype=np.uint32).tofile(tmp_path / "a.label")
    with pytest.raises(ValueError):
        load_labels(tmp_path / "a.label", n_points=5)


def test_calib_and_poses_parse(tmp_path) -> None:
    row = " ".join(str(v) for v in np.arange(12, dtype=float))
    (tmp_path / "calib.txt").write_text(f"P0: {row}\nTr: {row}\n")
    (tmp_path / "poses.txt").write_text(f"{row}\n{row}\n")
    calib = load_calib(tmp_path / "calib.txt")
    poses = load_poses(tmp_path / "poses.txt")
    assert calib["Tr"].shape == (4, 4)
    assert calib["Tr"][1, 3] == 7.0
    assert np.array_equal(calib["Tr"][3], [0, 0, 0, 1])
    assert poses.shape == (2, 4, 4)
    assert poses[0, 2, 0] == 8.0


def test_velo_pose_chain_recovers_motion() -> None:
    rng = np.random.default_rng(seed=1)
    tr_cam_velo = random_rigid(rng)
    true_world_velo = np.stack([np.eye(4), random_rigid(rng), random_rigid(rng)])
    # build cam poses the way kitti stores them: P_i = Tr @ T_v0_vi @ Tr^-1
    cam_poses = np.einsum("ab,ibc,cd->iad", tr_cam_velo, true_world_velo, np.linalg.inv(tr_cam_velo))
    got = velo_poses_from_cam_poses(cam_poses, tr_cam_velo)
    assert np.allclose(got, true_world_velo)

    # same physical point seen from two scans lands at one world spot
    p_world = np.array([4.0, 1.0, -1.73, 1.0])
    for t in got:
        p_velo = (np.linalg.inv(t) @ p_world)[None, :3]
        assert np.allclose(velo_to_world(p_velo, t), p_world[None, :3])


def test_range_gate_and_ego_box() -> None:
    pts = np.array(
        [
            [0.5, 0.0, -1.0],  # ego roof / too close
            [5.0, 0.0, -1.73],  # keep
            [60.0, 0.0, -1.73],  # too far
            [np.nan, 1.0, 1.0],  # broken
        ]
    )
    assert range_gate_mask(pts[:3], 2.0, 50.0).tolist() == [False, True, False]
    box = (-2.0, 2.0, -1.0, 1.0, -1.8, 0.0)
    assert ego_vehicle_mask(pts[:3], box).tolist() == [False, True, True]
    assert preprocess_mask(pts, 2.0, 50.0, box).tolist() == [False, True, False, False]


def test_ego_box_respects_z() -> None:
    box = (-1.2, 1.6, -0.9, 0.9, -1.8, 0.0)
    pts = np.array([[0.5, 0.0, -1.0], [0.5, 0.0, 0.5], [0.5, 0.0, -2.0]])  # body, above roof, below road
    assert ego_vehicle_mask(pts, box).tolist() == [False, True, True]
