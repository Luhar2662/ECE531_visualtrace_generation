# rrt_interface.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

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


# =============================
# Depth -> RGBD -> Point cloud
# =============================

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



def get_cached_depth_for_path(
    image_path: str = "simplerenv_complex.png",
    cache_dir: str = ".depth_cache",
    compute_fn=None,
    resize_fn=None,
) -> np.ndarray:
    assert compute_fn is not None

    base_dir = Path(__file__).resolve().parent
    img_path = (base_dir / image_path).resolve()

    st = img_path.stat()
    key = f"{img_path}__{st.st_mtime_ns}__{st.st_size}"
    fname = hashlib.sha256(key.encode("utf-8")).hexdigest() + ".npy"

    cache_path = (base_dir / cache_dir).resolve()
    cache_path.mkdir(parents=True, exist_ok=True)
    f = cache_path / fname

    # Helpful debug
    print("Cache file:", f)

    if f.exists():
        return np.load(f)

    img = Image.open(img_path).convert("RGB")
    if resize_fn is not None:
        img = resize_fn(img)

    depth = compute_fn(img).astype(np.float32)
    np.save(f, depth)
    return depth


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



def apply_T_inv(T, points_xyz):
    Tinv = np.linalg.inv(T)
    P = np.asarray(points_xyz, dtype=np.float64)
    Ph = np.concatenate([P, np.ones((P.shape[0], 1))], axis=1)  # (N,4)
    Pc = (Tinv @ Ph.T).T[:, :3]
    return Pc

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
    goal_sample_prob: float = 0.15
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


# ==========================
# Main
# ==========================

if __name__ == "__main__":
    # ---- Inputs ----
    img_path = "simplerenv_complex.png"

    # Pixels picked on ORIGINAL image (unresized) coordinates:
    start_pixel_orig = (376, 163)
    goal_pixel_orig  = (288, 176)

    # Synthetic intrinsics for the FINAL (resized+copped) image:
    fx = fy = 500.0
    PAD = 16

    # This flip matches your earlier pcd.transform()
    T_flip = np.array(
        [
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float64,
    )

    # ---- Load image & depth ----
    img0 = Image.open(img_path).convert("RGB")
    orig_size = img0.size  # (W0,H0)

    img_resized = resize_to_multiple_of_32(img0, target_h=480)
    resized_size = img_resized.size  # (Wr,Hr)

    depth_pred = predict_depth_glpn(img_resized)  # (Hr',Wr') but aligned to resized image in practice

    # crop borders (pad=16) consistently
    color_np, depth_m, img_cropped = make_rgbd_from_image_and_depth(img_resized, depth_pred, pad=PAD)

    # visualize
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img_cropped)
    ax[0].axis("off")
    ax[1].imshow(depth_m, cmap="plasma")
    ax[1].axis("off")
    plt.tight_layout()
    plt.pause(2)

    H, W = depth_m.shape[:2]
    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0

    # ---- Build point cloud ----
    pcd = build_point_cloud_from_rgbd(
        color_np=color_np,
        depth_m=depth_m,
        fx=fx, fy=fy, cx=cx, cy=cy,
        depth_scale=1.0,     # depth is float
        depth_trunc=10.0
    )

    # Clean & downsample
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd = pcd.voxel_down_sample(0.03)

    # Apply same transform to pcd and to start/goal points
    pcd.transform(T_flip)

    o3d.io.write_point_cloud("output.ply", pcd)

    # ---- Build voxel occupancy ----
    # 0.05
    voxel_size = 0.025
    occ = VoxelOccupancy.from_point_cloud(pcd, voxel_size=voxel_size)

    # ---- Convert start/goal pixels to 3D (camera frame -> then apply T_flip) ----
    u_s, v_s = map_pixel_orig_to_resized_cropped(
        start_pixel_orig[0], start_pixel_orig[1],
        orig_size=orig_size,
        resized_size=resized_size,
        pad=PAD,
    )
    u_g, v_g = map_pixel_orig_to_resized_cropped(
        goal_pixel_orig[0], goal_pixel_orig[1],
        orig_size=orig_size,
        resized_size=resized_size,
        pad=PAD,
    )
    

    start_cam = pixel_depth_to_3d_resized(u_s, v_s, depth_m, fx, fy, cx, cy, neighborhood=2)
    goal_cam  = pixel_depth_to_3d_resized(u_g, v_g, depth_m, fx, fy, cx, cy, neighborhood=2) 
    clearance = 0.01 + 2 * voxel_size  # robot_radius + 2 voxels (tune)

    # ray dirs in camera frame (use resized+cooked pixel u_s,v_s etc.)
    d_s_cam = pixel_to_ray_dir_cam(u_s, v_s, fx, fy, cx, cy)
    d_g_cam = pixel_to_ray_dir_cam(u_g, v_g, fx, fy, cx, cy)

    # move TOWARD camera in camera frame (subtract)
    start_cam_free = start_cam - clearance * d_s_cam
    goal_cam_free  = goal_cam  - clearance * d_g_cam

    # now apply same transform you used on the point cloud
    start = apply_T(T_flip, start_cam_free)
    goal  = apply_T(T_flip, goal_cam_free)



    if start_cam is None or goal_cam is None:
        raise RuntimeError("Could not unproject start/goal (invalid depth). Try larger neighborhood.")


    print("[Start 3D]", start)
    print("[Goal  3D]", goal)

    # ---- Set bounds automatically from pcd AABB (more robust than hardcoding) ----
    aabb = pcd.get_axis_aligned_bounding_box()
    bmin = aabb.get_min_bound() - 0.25
    bmax = aabb.get_max_bound() + 0.25

    params = RRTParams(
        max_iters=12000,
        step_size=0.20,
        goal_tolerance=0.25,
        bounds_min=tuple(bmin.tolist()),
        bounds_max=tuple(bmax.tolist()),
        rng_seed=0,
        collision_step=None,   # default -> 0.5*voxel_size
    )

    # ---- Plan ----
    path = rrt_plan(start, goal, occ, robot_radius=0.01, params=params)

    # ---- Visualize ----
    geoms = [pcd, occ.vg]

    # show start/goal as spheres
    s_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01).translate(start)
    g_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01).translate(goal)
    s_sphere.paint_uniform_color([0.1, 0.9, 0.1])
    g_sphere.paint_uniform_color([0.9, 0.1, 0.1])
    geoms += [s_sphere, g_sphere]

    if path is None:
        print("No path found.")
        o3d.visualization.draw_geometries(geoms)
    else:
        print("Path length:", len(path))
        pts = o3d.utility.Vector3dVector(path)
        lines = [[i, i + 1] for i in range(len(path) - 1)]
        ls = o3d.geometry.LineSet(points=pts, lines=o3d.utility.Vector2iVector(lines))
        geoms.append(ls)
        o3d.visualization.draw_geometries(geoms)


        # 1) convert path from "planning/world" frame back to camera frame
        path_cam = apply_T_inv(T_flip, path)

        # 2) project to CROPPED pixel coords (cropped image is img_cropped)
        pix_cropped, _ = project_cam_to_cropped_pixels(path_cam, fx, fy, cx, cy)

        # (optional) clip points outside cropped frame to break the polyline cleanly
        Hc, Wc = depth_m.shape[:2]
        out = (
            (pix_cropped[:, 0] < 0) | (pix_cropped[:, 0] >= Wc) |
            (pix_cropped[:, 1] < 0) | (pix_cropped[:, 1] >= Hc)
        )
        pix_cropped[out] = np.array([np.nan, np.nan])

        # 3) map cropped pixels back to ORIGINAL image coords
        pix_orig = cropped_to_original_pixels(pix_cropped, orig_size, resized_size, PAD)

        # 4) draw on original image and save/show
        img_with_path = draw_path_on_image(img0, pix_orig, color=(255, 0, 0), width=4)
        img_with_path.save("rrt_path_overlay.png")
        img_with_path.show()
        print("Saved overlay to rrt_path_overlay.png")