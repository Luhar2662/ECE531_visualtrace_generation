# rrt_interface.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np
import open3d as o3d


# -----------------------------
# Voxel occupancy for collisions
# -----------------------------

class VoxelOccupancy:
    """
    Occupancy map backed by Open3D VoxelGrid + a Python set of occupied grid indices.
    Provides point and segment collision tests with a spherical robot model.
    """
    def __init__(self, voxel_grid: o3d.geometry.VoxelGrid):
        self.vg = voxel_grid
        self.vs = float(voxel_grid.voxel_size)
        self.occ = {tuple(v.grid_index) for v in voxel_grid.get_voxels()}

    @staticmethod
    def from_point_cloud(pcd: o3d.geometry.PointCloud, voxel_size: float) -> "VoxelOccupancy":
        vg = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=float(voxel_size))
        return VoxelOccupancy(vg)

    def _grid_index(self, p: np.ndarray) -> Tuple[int, int, int]:
        # Open3D gives you the voxel index for an arbitrary point (even if unoccupied).
        idx = self.vg.get_voxel(p.astype(np.float64))
        return (int(idx[0]), int(idx[1]), int(idx[2]))

    def point_in_collision(self, p: np.ndarray, robot_radius: float = 0.0) -> bool:
        """
        Collision if the robot sphere at p intersects any occupied voxel.
        This does a conservative inflation by checking neighboring voxels within robot_radius.
        """
        if robot_radius <= 0.0:
            return self._grid_index(p) in self.occ

        r = float(robot_radius)
        vs = self.vs
        c = self._grid_index(p)

        # Number of voxels to search in each axis
        R = int(np.ceil(r / vs))

        # Check neighbors within a sphere in voxel coordinates
        r2 = r * r
        for dx in range(-R, R + 1):
            for dy in range(-R, R + 1):
                for dz in range(-R, R + 1):
                    # Convert voxel-offset to metric distance conservatively
                    # (voxel centers are spaced by vs)
                    if (dx*vs)**2 + (dy*vs)**2 + (dz*vs)**2 > r2:
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
        """
        Collision if any sampled point along segment [a,b] is in collision.
        Use step <= voxel_size for safety.
        """
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


# -----------------------------
# RRT core
# -----------------------------

@dataclass
class RRTParams:
    max_iters: int = 5000
    step_size: float = 0.25             # how far we extend per iteration (meters)
    goal_sample_prob: float = 0.15      # probability of sampling goal directly
    goal_tolerance: float = 0.25        # accept if within this distance (meters)
    bounds_min: Tuple[float, float, float] = (-5.0, -5.0, -1.0)
    bounds_max: Tuple[float, float, float] = ( 5.0,  5.0,  2.0)
    collision_step: Optional[float] = None  # if None: 0.5*voxel_size
    rng_seed: Optional[int] = None


@dataclass
class _Node:
    p: np.ndarray
    parent: int


def _sample_uniform(rng: np.random.Generator, bmin: np.ndarray, bmax: np.ndarray) -> np.ndarray:
    return rng.uniform(bmin, bmax)


def _nearest(nodes: List[_Node], q: np.ndarray) -> int:
    # brute-force nearest neighbor
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
    """
    Returns:
      path: (N,3) array of waypoints from start to goal, or None if failed.
    """
    if params is None:
        params = RRTParams()

    rng = np.random.default_rng(params.rng_seed)

    start = np.asarray(start, dtype=np.float64).reshape(3)
    goal  = np.asarray(goal,  dtype=np.float64).reshape(3)

    bmin = np.asarray(params.bounds_min, dtype=np.float64)
    bmax = np.asarray(params.bounds_max, dtype=np.float64)

    # Basic feasibility
    if occ.point_in_collision(start, robot_radius):
        return None
    if occ.point_in_collision(goal, robot_radius):
        return None

    nodes: List[_Node] = [_Node(p=start, parent=-1)]

    for _ in range(params.max_iters):
        # sample
        if rng.uniform() < params.goal_sample_prob:
            q = goal
        else:
            q = _sample_uniform(rng, bmin, bmax)

        # extend
        i_near = _nearest(nodes, q)
        p_near = nodes[i_near].p
        p_new = _steer(p_near, q, params.step_size)

        # collision checks (edge + point)
        if occ.segment_in_collision(p_near, p_new, robot_radius, step=params.collision_step):
            continue

        nodes.append(_Node(p=p_new, parent=i_near))

        # goal check (connect-to-goal attempt)
        if np.linalg.norm(p_new - goal) <= params.goal_tolerance:
            if not occ.segment_in_collision(p_new, goal, robot_radius, step=params.collision_step):
                nodes.append(_Node(p=goal.copy(), parent=len(nodes) - 1))
                return _reconstruct_path(nodes, len(nodes) - 1)

    return None


# -----------------------------
# Example usage (you adapt)
# -----------------------------
if __name__ == "__main__":
    # Load / create a pcd somehow
    pcd = o3d.io.read_point_cloud("cloud.ply")

    # Clean
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd = pcd.select_by_index(ind)
    pcd = pcd.voxel_down_sample(0.03)

    voxel_size = 0.05
    occ = VoxelOccupancy.from_point_cloud(pcd, voxel_size=voxel_size)

    start = np.array([0.0, 0.0, 0.2])
    goal  = np.array([2.0, 1.0, 0.2])

    params = RRTParams(
        max_iters=8000,
        step_size=0.20,
        goal_tolerance=0.25,
        bounds_min=(-2, -2, 0.0),
        bounds_max=( 3,  3, 1.5),
        rng_seed=0,
    )

    path = rrt_plan(start, goal, occ, robot_radius=0.20, params=params)
    if path is None:
        print("No path found")
    else:
        print("Path length:", len(path))

        # Visualize path as a LineSet
        pts = o3d.utility.Vector3dVector(path)
        lines = [[i, i+1] for i in range(len(path)-1)]
        ls = o3d.geometry.LineSet(points=pts, lines=o3d.utility.Vector2iVector(lines))
        o3d.visualization.draw_geometries([pcd, occ.vg, ls])
