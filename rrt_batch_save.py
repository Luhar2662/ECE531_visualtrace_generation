# rrt_interface.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List, Literal, Dict


from PIL import Image

import numpy as np
import open3d as o3d

# Optional: visualization of image/depth
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

from PIL import Image, ImageDraw
import torch
from transformers import GLPNImageProcessor, GLPNForDepthEstimation
from pathlib import Path
import hashlib
import cv2 as cv

# =============================
# Depth -> RGBD -> Point cloud
# =============================
# def _resize_image(image: np.ndarray) -> np.ndarray:
#     image = cv.resize(image, tuple([256,256]), interpolation=cv.INTER_AREA)
#     return image

def resize_to_multiple_of_32(img: Image.Image, target_h: int = 480) -> Image.Image:
    new_height = target_h if img.height > target_h else img.height
    new_height -= (new_height % 32)

    new_width = int(new_height * img.width / img.height)
    diff = new_width % 32
    new_width = new_width - diff if diff < 16 else new_width + 32 - diff

    return img.resize((new_width, new_height))


def predict_depth_glpn(image: Image.Image) -> np.ndarray:
    """
    Returns predicted depth as a float array (H,W). Units are arbitrary (relative).
    """
    feature_extractor = GLPNImageProcessor.from_pretrained("vinvino02/glpn-nyu")
    model = GLPNForDepthEstimation.from_pretrained("vinvino02/glpn-nyu")
    model.eval()

    inputs = feature_extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth  # (1, 1, h, w)

    depth = predicted_depth.squeeze().cpu().numpy()
    return depth



# def get_cached_depth_for_path(
#     image_path: str = "pick_up_the_blue_can_obj_-0.1_-0.05_001_001.png",
#     cache_dir: str = ".depth_cache",
#     compute_fn=None,
#     resize_fn=None,
# ) -> np.ndarray:
#     assert compute_fn is not None

#     base_dir = Path(__file__).resolve().parent
#     img_path = (base_dir / image_path).resolve()

#     st = img_path.stat()
#     key = f"{img_path}__{st.st_mtime_ns}__{st.st_size}"
#     fname = hashlib.sha256(key.encode("utf-8")).hexdigest() + ".npy"

#     cache_path = (base_dir / cache_dir).resolve()
#     cache_path.mkdir(parents=True, exist_ok=True)
#     f = cache_path / fname

#     # Helpful debug
#     print("Cache file:", f)

#     if f.exists():
#         return np.load(f)

#     img = Image.open(img_path).convert("RGB")
#     if resize_fn is not None:
#         img = resize_fn(img)

#     depth = compute_fn(img).astype(np.float32)
#     np.save(f, depth)
#     return depth


def pixel_to_ray_dir_cam(u, v, fx, fy, cx, cy):
    d = np.array([(u - cx)/fx, (v - cy)/fy, 1.0], dtype=np.float64)
    return d / np.linalg.norm(d)



def make_rgbd_from_image_and_depth(
    image_pil: Image.Image,
    depth_float: np.ndarray,
    pad: int = 16,
) -> Tuple[np.ndarray, np.ndarray, Image.Image]:
    """
    Crops borders consistently and returns:
      - color_np: uint8 (H,W,3)
      - depth_m:  float32 (H,W) in *meters-like* scale (not true meters unless you calibrate!)
      - image_cropped_pil: cropped PIL image for reference
    """
    # GLPN depth is relative; scale is arbitrary.
    # We'll keep it as float and treat it as "meters-like" for geometry consistency.
    depth = depth_float.astype(np.float32)

    # crop to match your earlier code
    if pad > 0:
        depth = depth[pad:-pad, pad:-pad]
        image_pil = image_pil.crop((pad, pad, image_pil.width - pad, image_pil.height - pad))

    color_np = np.array(image_pil).astype(np.uint8)
    return color_np, depth, image_pil


def build_point_cloud_from_rgbd(
    color_np: np.ndarray,
    depth_m: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    depth_scale: float = 1.0,
    depth_trunc: float = 10.0,
) -> o3d.geometry.PointCloud:
    """
    Build point cloud from color/depth with Open3D.
    depth_m should be float in meters-like units; depth_scale typically 1.0 for float depth.
    """
    H, W = depth_m.shape[:2]
    assert color_np.shape[0] == H and color_np.shape[1] == W, "Color/depth resolution mismatch"

    color_o3d = o3d.geometry.Image(color_np)
    depth_o3d = o3d.geometry.Image(depth_m.astype(np.float32))

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d,
        depth_o3d,
        depth_scale=depth_scale,   # 1.0 if depth is float meters-like
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=False,
    )

    intr = o3d.camera.PinholeCameraIntrinsic()
    intr.set_intrinsics(W, H, fx, fy, cx, cy)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intr)
    return pcd


# ==========================================
# Pixel + depth -> 3D (consistent frame)
# ==========================================

def pixel_depth_to_3d_resized(
    u: float,
    v: float,
    depth_m: np.ndarray,     # depth for the RESIZED+CROPPED image (same resolution as color_np)
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    neighborhood: int = 2,   # search nearby pixels if invalid (NaN/<=0)
) -> Optional[np.ndarray]:
    """
    Pixel (u,v) must be in the RESIZED+CROPPED image coordinate system.
    Returns point in CAMERA frame.
    """
    H, W = depth_m.shape[:2]
    ui = int(round(u))
    vi = int(round(v))
    ui = max(0, min(W - 1, ui))
    vi = max(0, min(H - 1, vi))

    def valid(z: float) -> bool:
        return np.isfinite(z) and (z > 0.0)

    z = float(depth_m[vi, ui])
    if not valid(z):
        found = False
        for r in range(1, neighborhood + 1):
            for dv in range(-r, r + 1):
                for du in range(-r, r + 1):
                    vv, uu = vi + dv, ui + du
                    if 0 <= vv < H and 0 <= uu < W:
                        z_try = float(depth_m[vv, uu])
                        if valid(z_try):
                            vi, ui, z = vv, uu, z_try
                            found = True
                            break
                if found:
                    break
            if found:
                break
        if not found:
            return None

    X = (ui - cx) * z / fx
    Y = (vi - cy) * z / fy
    Z = z
    return np.array([X, Y, Z], dtype=np.float64)


def map_pixel_orig_to_resized_cropped(
    u0: float,
    v0: float,
    orig_size: Tuple[int, int],         # (W0,H0) original file
    resized_size: Tuple[int, int],      # (Wr,Hr) after resizing BEFORE crop
    pad: int,
) -> Tuple[float, float]:
    """
    Convert a pixel from ORIGINAL image coords to RESIZED+CROPPED coords.
    """
    W0, H0 = orig_size
    Wr, Hr = resized_size

    # scale to resized (before crop)
    u_r = u0 * (Wr / W0)
    v_r = v0 * (Hr / H0)

    # then apply crop offset
    u_c = u_r - pad
    v_c = v_r - pad
    return u_c, v_c



def project_cam_to_cropped_pixels(points_cam, fx, fy, cx, cy):
    P = np.asarray(points_cam, dtype=np.float64)
    X, Y, Z = P[:, 0], P[:, 1], P[:, 2]
    valid = Z > 1e-9

    u = np.full_like(Z, np.nan, dtype=np.float64)
    v = np.full_like(Z, np.nan, dtype=np.float64)
    u[valid] = fx * (X[valid] / Z[valid]) + cx
    v[valid] = fy * (Y[valid] / Z[valid]) + cy
    return np.stack([u, v], axis=1), valid

def cropped_to_original_pixels(pix_cropped, orig_size, resized_size, pad):
    W0, H0 = orig_size
    Wr, Hr = resized_size

    pix = np.asarray(pix_cropped, dtype=np.float64)
    u_r = pix[:, 0] + pad
    v_r = pix[:, 1] + pad

    u0 = u_r * (W0 / Wr)
    v0 = v_r * (H0 / Hr)
    return np.stack([u0, v0], axis=1)

def draw_path_on_image(img_pil, pix_orig, color=(255, 0, 0), width=4):
    img_out = img_pil.copy()
    draw = ImageDraw.Draw(img_out)

    # break the line whenever points are invalid
    seg = []
    for u, v in pix_orig:
        if np.isfinite(u) and np.isfinite(v):
            seg.append((float(u), float(v)))
        else:
            if len(seg) >= 2:
                draw.line(seg, fill=color, width=width)
            seg = []
    if len(seg) >= 2:
        draw.line(seg, fill=color, width=width)
    return img_out


def resample_polyline_max_points(path_xyz: np.ndarray, max_points: int = 5) -> np.ndarray:
    """
    Resample a polyline (N,3) to have at most `max_points` points (>=2),
    evenly spaced by arc length. Preserves endpoints.
    """
    path_xyz = np.asarray(path_xyz, dtype=np.float64)
    n = path_xyz.shape[0]
    if n <= max_points:
        return path_xyz

    # cumulative arc length
    seg = np.linalg.norm(path_xyz[1:] - path_xyz[:-1], axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    if total <= 1e-12:
        # all points identical; just return endpoints
        return path_xyz[[0, -1], :]

    # target arc-length positions
    k = max_points
    s_targets = np.linspace(0.0, total, k)

    # resample by linear interpolation along segments
    out = np.zeros((k, 3), dtype=np.float64)
    j = 0
    for i, st in enumerate(s_targets):
        while j < len(s) - 2 and s[j + 1] < st:
            j += 1
        s0, s1 = s[j], s[j + 1]
        p0, p1 = path_xyz[j], path_xyz[j + 1]
        t = 0.0 if (s1 - s0) < 1e-12 else (st - s0) / (s1 - s0)
        out[i] = (1 - t) * p0 + t * p1

    return out






def apply_T(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    p_h = np.r_[p, 1.0]
    return (T @ p_h)[:3]


# ==========================
# Voxel occupancy for RRT
# ==========================

class VoxelOccupancy:
    def __init__(self, voxel_grid: o3d.geometry.VoxelGrid):
        self.vg = voxel_grid
        self.vs = float(voxel_grid.voxel_size)
        self.occ = {tuple(v.grid_index) for v in voxel_grid.get_voxels()}

    @staticmethod
    def from_point_cloud(pcd: o3d.geometry.PointCloud, voxel_size: float) -> "VoxelOccupancy":
        vg = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=float(voxel_size))
        return VoxelOccupancy(vg)

    def _grid_index(self, p: np.ndarray) -> Tuple[int, int, int]:
        idx = self.vg.get_voxel(p.astype(np.float64))
        return (int(idx[0]), int(idx[1]), int(idx[2]))

    def point_in_collision(self, p: np.ndarray, robot_radius: float = 0.0) -> bool:
        if robot_radius <= 0.0:
            return self._grid_index(p) in self.occ

        r = float(robot_radius)
        vs = self.vs
        c = self._grid_index(p)
        R = int(np.ceil(r / vs))
        r2 = r * r

        for dx in range(-R, R + 1):
            for dy in range(-R, R + 1):
                for dz in range(-R, R + 1):
                    if (dx * vs) ** 2 + (dy * vs) ** 2 + (dz * vs) ** 2 > r2:
                        continue
                    if (c[0] + dx, c[1] + dy, c[2] + dz) in self.occ:
                        return True
        return False

    def segment_in_collision(
        self,
        a: np.ndarray,
        b: np.ndarray,
        robot_radius: float = 0.0,
        step: Optional[float] = None,
    ) -> bool:
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)

        if step is None:
            step = 0.5 * self.vs

        d = b - a
        L = float(np.linalg.norm(d))
        if L == 0.0:
            return self.point_in_collision(a, robot_radius)

        n = int(np.ceil(L / float(step)))
        for i in range(n + 1):
            t = i / n
            p = a + t * d
            if self.point_in_collision(p, robot_radius):
                return True
        return False


# ==========================
# RRT
# ==========================

@dataclass
class RRTParams:
    max_iters: int = 8000
    step_size: float = 0.20
    goal_sample_prob: float = 0.30
    goal_tolerance: float = 0.25
    bounds_min: Tuple[float, float, float] = (-2.0, -2.0, 0.0)
    bounds_max: Tuple[float, float, float] = ( 3.0,  3.0, 1.5)
    collision_step: Optional[float] = None
    rng_seed: Optional[int] = 0


@dataclass
class _Node:
    p: np.ndarray
    parent: int


def _sample_uniform(rng: np.random.Generator, bmin: np.ndarray, bmax: np.ndarray) -> np.ndarray:
    return rng.uniform(bmin, bmax)


def _nearest(nodes: List[_Node], q: np.ndarray) -> int:
    best_i = 0
    best_d2 = float("inf")
    for i, n in enumerate(nodes):
        d2 = float(np.sum((n.p - q) ** 2))
        if d2 < best_d2:
            best_d2 = d2
            best_i = i
    return best_i


def _steer(a: np.ndarray, b: np.ndarray, step_size: float) -> np.ndarray:
    d = b - a
    L = float(np.linalg.norm(d))
    if L <= step_size:
        return b.copy()
    return a + (step_size / L) * d


def _reconstruct_path(nodes: List[_Node], leaf_idx: int) -> np.ndarray:
    path = []
    i = leaf_idx
    while i != -1:
        path.append(nodes[i].p)
        i = nodes[i].parent
    path.reverse()
    return np.vstack(path)


def rrt_plan(
    start: np.ndarray,
    goal: np.ndarray,
    occ: VoxelOccupancy,
    robot_radius: float = 0.0,
    params: Optional[RRTParams] = None,
) -> Optional[np.ndarray]:
    if params is None:
        params = RRTParams()

    rng = np.random.default_rng(params.rng_seed)

    start = np.asarray(start, dtype=np.float64).reshape(3)
    goal  = np.asarray(goal,  dtype=np.float64).reshape(3)

    bmin = np.asarray(params.bounds_min, dtype=np.float64)
    bmax = np.asarray(params.bounds_max, dtype=np.float64)

    # if occ.point_in_collision(start, robot_radius):
    #     print("[RRT] Start in collision.")
    #     return None
    # if occ.point_in_collision(goal, robot_radius):
    #     print("[RRT] Goal in collision.")
    #     return None

    nodes: List[_Node] = [_Node(p=start, parent=-1)]

    for _ in range(params.max_iters):
        q = goal if (rng.uniform() < params.goal_sample_prob) else _sample_uniform(rng, bmin, bmax)

        i_near = _nearest(nodes, q)
        p_near = nodes[i_near].p
        p_new = _steer(p_near, q, params.step_size)

        if occ.segment_in_collision(p_near, p_new, robot_radius, step=params.collision_step):
            continue

        nodes.append(_Node(p=p_new, parent=i_near))

        if np.linalg.norm(p_new - goal) <= params.goal_tolerance:
            if not occ.segment_in_collision(p_new, goal, robot_radius, step=params.collision_step):
                nodes.append(_Node(p=goal.copy(), parent=len(nodes) - 1))
                return _reconstruct_path(nodes, len(nodes) - 1)

    return None





# -----------------------------
# User-tunable constants
# -----------------------------
FX = FY = 500.0
PAD = 0
VOXEL_SIZE = 0.025
ROBOT_RADIUS = 0.01
RNG_SEED = 10

# -----------------------------
# Types
# -----------------------------
FailReason = Literal[
    "OK",
    "PIXEL_OOB_AFTER_PAD",
    "INVALID_DEPTH",
    "START_IN_COLLISION",
    "GOAL_IN_COLLISION",
    "NO_PATH",
]

@dataclass
class TargetResult:
    success: bool
    reason: FailReason
    path_len: Optional[int]

@dataclass
class ImageResult:
    obj_tag: str
    image_path: str
    blue: TargetResult
    coke: TargetResult
    both_success: bool


def _orig256_to_depth_pixel(u0: int, v0: int, pad: int) -> Tuple[int, int]:
    # If you crop by PAD on all sides, depth_m becomes (H-2*PAD, W-2*PAD).
    # So original pixel (u0,v0) maps to cropped coordinates (u0-PAD, v0-PAD).
    return u0 - pad, v0 - pad

def apply_T_inv(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply inverse of 4x4 transform to Nx3 points."""
    Tinv = np.linalg.inv(T)
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts[None, :]
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    Ph = np.hstack([pts, ones])
    Qh = (Tinv @ Ph.T).T
    return Qh[:, :3]

def project_cam_to_depth_pixels(
    pts_cam: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
) -> np.ndarray:
    """
    Project Nx3 CAM points to Nx2 pixel coords in the DEPTH frame (i.e., depth_m coordinates).
    """
    pts_cam = np.asarray(pts_cam, dtype=np.float64)
    Z = pts_cam[:, 2]
    eps = 1e-9
    Z = np.where(Z > eps, Z, np.nan)
    u = fx * (pts_cam[:, 0] / Z) + cx
    v = fy * (pts_cam[:, 1] / Z) + cy
    return np.stack([u, v], axis=1)

def depth_pixels_to_orig256(pix_depth: np.ndarray, pad: int) -> np.ndarray:
    """Convert pixels in depth_m coords back to original 256x256 coords."""
    pix_depth = np.asarray(pix_depth, dtype=np.float64)
    return pix_depth + np.array([pad, pad], dtype=np.float64)


def draw_polyline(img: Image.Image, pix: np.ndarray, color: Tuple[int,int,int], width: int = 3) -> Image.Image:
    """Draw polyline on a PIL image, breaking on NaNs and OOB points."""
    img_out = img.copy()
    draw = ImageDraw.Draw(img_out)
    W, H = img_out.size

    pts = np.asarray(pix, dtype=np.float64)

    cur: List[Tuple[int,int]] = []
    for u, v in pts:
        if np.isnan(u) or np.isnan(v):
            if len(cur) >= 2:
                draw.line(cur, fill=color, width=width)
            cur = []
            continue
        ui = int(round(u))
        vi = int(round(v))
        if 0 <= ui < W and 0 <= vi < H:
            cur.append((ui, vi))
        else:
            if len(cur) >= 2:
                draw.line(cur, fill=color, width=width)
            cur = []
    if len(cur) >= 2:
        draw.line(cur, fill=color, width=width)

    return img_out

def ensure_overlay_dir(rel_dir: str = "overlay") -> Path:
    out_dir = Path(__file__).parent / rel_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

# -----------------------------
# Unprojection (unchanged except we keep (u,v) for projection reuse)
# -----------------------------
def _unproject_pixel_to_world_with_reason(
    pixel_orig_256: Tuple[int, int],
    depth_m: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    voxel_size: float,
    robot_radius: float,
    pad: int,
    T_flip: np.ndarray,
    neighborhood: int = 2,
) -> Tuple[Optional[np.ndarray], Optional[Tuple[int,int]], FailReason]:
    """
    Returns:
      p_world: (3,) or None
      uv_depth: (u,v) in depth_m coords (ints) or None
      reason
    """
    u0, v0 = pixel_orig_256
    u, v = _orig256_to_depth_pixel(u0, v0, pad)

    H, W = depth_m.shape[:2]
    if not (0 <= u < W and 0 <= v < H):
        return None, None, "PIXEL_OOB_AFTER_PAD"

    p_cam = pixel_depth_to_3d_resized(u, v, depth_m, fx, fy, cx, cy, neighborhood=neighborhood)
    if p_cam is None:
        return None, None, "INVALID_DEPTH"

    clearance = float(robot_radius) + 6.0 * float(voxel_size)
    d_cam = pixel_to_ray_dir_cam(u, v, fx, fy, cx, cy)
    p_cam_free = p_cam - clearance * d_cam

    return apply_T(T_flip, p_cam_free), (int(u), int(v)), "OK"

# -----------------------------
# Core evaluation, now returns paths for overlay
# -----------------------------
def evaluate_image_with_reasons_and_paths(
    img_path: str,
    ee_pixel: Tuple[int, int],
    blue_pixel: Tuple[int, int],
    coke_pixel: Tuple[int, int],
    fx: float, fy: float,
    pad: int,
    voxel_size: float,
    robot_radius: float,
    T_flip: np.ndarray,
    rng_seed: int,
) -> Tuple[
    TargetResult, Optional[np.ndarray],
    TargetResult, Optional[np.ndarray],
    float, float
]:
    """
    Returns:
      blue_res, blue_path,
      coke_res, coke_path,
      cx, cy   (intrinsics principal point for depth_m frame)
    """
    img0 = Image.open(img_path).convert("RGB")
    depth_pred = predict_depth_glpn(img0)
    color_np, depth_m, _ = make_rgbd_from_image_and_depth(img0, depth_pred, pad=pad)

    H, W = depth_m.shape[:2]
    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0

    pcd = build_point_cloud_from_rgbd(
        color_np=color_np,
        depth_m=depth_m,
        fx=fx, fy=fy, cx=cx, cy=cy,
        depth_scale=1.0,
        depth_trunc=10.0
    )
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd = pcd.voxel_down_sample(0.03)
    pcd.transform(T_flip)

    occ = VoxelOccupancy.from_point_cloud(pcd, voxel_size=voxel_size)

    # Start (end effector)
    start, _, rs = _unproject_pixel_to_world_with_reason(
        ee_pixel, depth_m, fx, fy, cx, cy, voxel_size, robot_radius, pad, T_flip
    )
    if start is None:
        return (
            TargetResult(False, rs, None), None,
            TargetResult(False, rs, None), None,
            cx, cy
        )

    if occ.point_in_collision(start, robot_radius):
        return (
            TargetResult(False, "START_IN_COLLISION", None), None,
            TargetResult(False, "START_IN_COLLISION", None), None,
            cx, cy
        )

    # Bounds from AABB
    aabb = pcd.get_axis_aligned_bounding_box()
    bmin = aabb.get_min_bound() - 0.25
    bmax = aabb.get_max_bound() + 0.25

    params = RRTParams(
        max_iters=12000,
        step_size=0.20,
        goal_tolerance=0.25,
        bounds_min=tuple(bmin.tolist()),
        bounds_max=tuple(bmax.tolist()),
        rng_seed=rng_seed,
        collision_step=None,
    )

    def plan_to_target(pixel_orig_256: Tuple[int, int]) -> Tuple[TargetResult, Optional[np.ndarray]]:
        goal, _, rg = _unproject_pixel_to_world_with_reason(
            pixel_orig_256, depth_m, fx, fy, cx, cy, voxel_size, robot_radius, pad, T_flip
        )
        if goal is None:
            return TargetResult(False, rg, None), None

        if occ.point_in_collision(goal, robot_radius):
            return TargetResult(False, "GOAL_IN_COLLISION", None), None

        path = rrt_plan(start, goal, occ, robot_radius=robot_radius, params=params)
        if path is None:
            return TargetResult(False, "NO_PATH", None), None

        return TargetResult(True, "OK", int(len(path))), path

    blue_res, blue_path = plan_to_target(blue_pixel)
    coke_res, coke_path = plan_to_target(coke_pixel)
    return blue_res, blue_path, coke_res, coke_path, cx, cy

# -----------------------------
# Batch: print reasons + save overlays
# -----------------------------
def batch_success_report_with_reasons_and_overlays(
    image_files: Dict[str, str],
    can_pixels: Dict[str, Dict[str, Tuple[int, int]]],
    ee_pixel: Tuple[int, int],
    fx: float,
    fy: float,
    pad: int,
    voxel_size: float,
    robot_radius: float,
    T_flip: np.ndarray,
    rng_seed: int,
    overlay_dir: str = "overlay",
) -> List[ImageResult]:
    rows: List[ImageResult] = []
    out_dir = ensure_overlay_dir(overlay_dir)

    for obj_tag, img_path in image_files.items():
        blue_px = can_pixels[obj_tag]["blue"]
        coke_px = can_pixels[obj_tag]["coke"]

        blue_res, blue_path, coke_res, coke_path, cx, cy = evaluate_image_with_reasons_and_paths(
            img_path=img_path,
            ee_pixel=ee_pixel,
            blue_pixel=blue_px,
            coke_pixel=coke_px,
            fx=fx, fy=fy,
            pad=pad,
            voxel_size=voxel_size,
            robot_radius=robot_radius,
            T_flip=T_flip,
            rng_seed=rng_seed,
        )

        row = ImageResult(
            obj_tag=obj_tag,
            image_path=img_path,
            blue=blue_res,
            coke=coke_res,
            both_success=bool(blue_res.success and coke_res.success),
        )
        rows.append(row)

        print(
            f"{obj_tag}: "
            f"BLUE={blue_res.success}({blue_res.reason}, len={blue_res.path_len}) | "
            f"COKE={coke_res.success}({coke_res.reason}, len={coke_res.path_len}) | "
            f"BOTH={row.both_success}"
        )

        # ---------- Save overlays (no show) ----------
        # Always load the ORIGINAL image for drawing (your resized_*.png is already 256x256).
        img0 = Image.open(img_path).convert("RGB")

        any_path = False
        img_overlay = img0.copy()

        # Blue path overlay
        if blue_path is not None:
            any_path = True
            path_vis = resample_polyline_max_points(blue_path, max_points=5)
            path_cam = apply_T_inv(T_flip, path_vis)  # world -> camera
            pix_depth = project_cam_to_depth_pixels(path_cam, fx, fy, cx, cy)  # in depth_m coords
            pix_orig = depth_pixels_to_orig256(pix_depth, pad)               # back to 256x256 coords
            img_b = draw_polyline(img0, pix_orig, color=(0, 120, 255), width=3)
            img_b.save(out_dir / f"{obj_tag}__blue.png")
            img_overlay = draw_polyline(img_overlay, pix_orig, color=(0, 120, 255), width=3)

        # Coke path overlay
        if coke_path is not None:
            any_path = True
            path_vis = resample_polyline_max_points(coke_path, max_points=5)
            path_cam = apply_T_inv(T_flip, path_vis)
            pix_depth = project_cam_to_depth_pixels(path_cam, fx, fy, cx, cy)
            pix_orig = depth_pixels_to_orig256(pix_depth, pad)
            img_c = draw_polyline(img0, pix_orig, color=(255, 60, 60), width=3)
            img_c.save(out_dir / f"{obj_tag}__coke.png")
            img_overlay = draw_polyline(img_overlay, pix_orig, color=(255, 60, 60), width=3)

        # Combined overlay
        if any_path:
            img_overlay.save(out_dir / f"{obj_tag}__overlay.png")

    return rows

# -----------------------------
# Inputs
# -----------------------------
END_EFFECTOR_PIXEL = (234, 71)  # fixed across all runs (256x256 coords)

CAN_PIXELS: Dict[str, Dict[str, Tuple[int, int]]] = {
    "obj_-0.1_-0.05": {"blue": (75, 79), "coke": (21, 142)},
    "obj_-0.1_-0.45": {"blue": (132, 150), "coke": (194, 143)},
    "obj_-0.35_-0.05": {"blue": (101, 106), "coke": (55, 71)},
    "obj_-0.35_0.45": {"blue": (186, 157), "coke": (188, 82)},
    "obj_-0.235_0.42": {"blue": (124, 98), "coke": (179, 118)},
    "obj_-0.22499999999999998_-0.05": {"blue": (91, 59), "coke": (50, 111)},
    "obj_-0.22499999999999998_0.45": {"blue": (123, 100), "coke": (190, 110)},
}

IMAGE_FILES: Dict[str, str] = {
    "obj_-0.1_-0.05": "resized_pick_up_the_blue_can_obj_-0.1_-0.05_001_001.png",
    "obj_-0.1_-0.45": "resized_pick_up_the_blue_can_obj_-0.1_0.45_001_001.png",
    "obj_-0.35_-0.05": "resized_pick_up_the_blue_can_obj_-0.35_-0.05_001_001.png",
    "obj_-0.35_0.45": "resized_pick_up_the_blue_can_obj_-0.35_0.45_001_001.png",
    "obj_-0.235_0.42": "resized_pick_up_the_blue_can_obj_-0.235_0.42_001_001.png",
    "obj_-0.22499999999999998_0.45": "resized_pick_up_the_blue_can_obj_-0.22499999999999998_0.45_001_001.png",
    "obj_-0.22499999999999998_-0.05": "resized_pick_up_the_blue_can_obj_-0.22499999999999998_-0.05_001_001.png",
}

T_FLIP = np.array(
    [[1, 0, 0, 0],
     [0, -1, 0, 0],
     [0, 0, -1, 0],
     [0, 0, 0, 1]], dtype=np.float64
)

if __name__ == "__main__":
    rows = batch_success_report_with_reasons_and_overlays(
        image_files=IMAGE_FILES,
        can_pixels=CAN_PIXELS,
        ee_pixel=END_EFFECTOR_PIXEL,
        fx=FX, fy=FY,
        pad=PAD,
        voxel_size=VOXEL_SIZE,
        robot_radius=ROBOT_RADIUS,
        T_flip=T_FLIP,
        rng_seed=RNG_SEED,
        overlay_dir="overlay",
    )
