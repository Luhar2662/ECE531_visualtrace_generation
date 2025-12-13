"""
Point cloud from color + depth image

Usage (example):
	python pointcloud.py --color color.png --depth depth.png --output pcd.ply

This script supports reading color/depth images using Open3D directly. Depth images
are expected to be in millimeters (uint16) or meters (float). We use the optional
`--depth_scale` argument to convert depth values to meters for Open3D.

If you don't have a depth image, you can either provide a grayscale image and set
`--generate_depth` to use it as a depth heuristic, or pre-compute depth from stereo
or other sensors and provide that as a depth image.
"""

import argparse
import os
import numpy as np
import open3d as o3d
from PIL import Image


def load_images(color_path: str, depth_path: str = None):
	"""Load color and depth images and return Open3D Image objects.

	Args:
		color_path: path to color image (RGB)
		depth_path: path to depth image (optional)

	Returns:
		(color_o3d, depth_o3d)
	"""
	if not os.path.exists(color_path):
		raise FileNotFoundError(f"Color file not found: {color_path}")
	color_o3d = o3d.io.read_image(color_path)

	if depth_path is None:
		depth_o3d = None
	else:
		if not os.path.exists(depth_path):
			raise FileNotFoundError(f"Depth file not found: {depth_path}")
		depth_o3d = o3d.io.read_image(depth_path)

	return color_o3d, depth_o3d


def analyze_image(o3d_img: o3d.geometry.Image, name: str = "image"):
	arr = np.asarray(o3d_img)
	dtype = arr.dtype
	minv = float(arr.min())
	maxv = float(arr.max())
	mean = float(arr.mean())
	non_zero = float(np.count_nonzero(arr))
	total = float(arr.size)
	nz_fraction = non_zero / total
	print(f"{name} stats: dtype={dtype}, shape={arr.shape}, min={minv}, max={maxv}, mean={mean:.3f}, nonzero_frac={nz_fraction:.3f}")
	return arr


def map_depth_to_meters(depth_o3d: o3d.geometry.Image, depth_max_m: float):
	"""Map input depth pixel values linearly to meters using given max depth.

	If input is uint8/uint16 integer, uses the dtype max as the pixel max.
	Returns a float32 Open3D image with distances in meters.
	"""
	arr = np.asarray(depth_o3d)
	if np.issubdtype(arr.dtype, np.integer):
		max_pixel = float(np.iinfo(arr.dtype).max)
	else:
		max_pixel = float(arr.max()) if arr.max() != 0 else 1.0
	if max_pixel == 0:
		print("Warning: depth image has max pixel value 0; cannot map to meters.")
		max_pixel = 1.0
	depth_m = (arr.astype(np.float32) / max_pixel) * float(depth_max_m)
	return o3d.geometry.Image(depth_m.astype(np.float32))


def _o3d_to_numpy(img: o3d.geometry.Image):
	arr = np.asarray(img)
	# Open3D returns HxW for depth and HxWxC for color
	return arr


def _numpy_to_o3d(arr: np.ndarray):
	return o3d.geometry.Image(arr)


def resize_o3d_image_to(img_o3d: o3d.geometry.Image, target_w: int, target_h: int, is_depth: bool = False, method: str = "nearest"):
	"""Resize an Open3D Image to a target size using PIL resampling.

	Args:
		img_o3d: Open3D Image to resize
		target_w: desired width in pixels
		target_h: desired height in pixels
		is_depth: True if image is a depth map (special handling for dtype)
		method: interpolation method: 'nearest', 'bilinear', 'bicubic'
	Returns:
		resized Open3D Image
	"""
	arr = _o3d_to_numpy(img_o3d)
	# If depth image contains color channels (e.g., 3-channel PNG/JPG) convert to single channel
	if is_depth and arr.ndim == 3:
		# Use the first channel as the depth channel; other methods (e.g., weighted average) could be used
		arr = arr[..., 0]
	# pick resample option
	resample_map = {
		"nearest": Image.NEAREST,
		"bilinear": Image.BILINEAR,
		"bicubic": Image.BICUBIC,
	}
	resample = resample_map.get(method, Image.NEAREST)

	# Depth images are usually uint16 or float; convert to float for resizing to avoid overflow
	if is_depth:
		src_dtype = arr.dtype
		# Convert to float32 for PIL resizing
		arr_f = arr.astype(np.float32)
		# PIL requires 2D arrays for single-channel
		# PIL cannot handle 3-channel float images; ensure 2D float array (mode 'F') for depth
		pil_img = Image.fromarray(arr_f.astype(np.float32))
		pil_img = pil_img.resize((int(target_w), int(target_h)), resample=resample)
		resized_arr = np.asarray(pil_img)
		# convert back while trying to preserve dtype
		if np.issubdtype(src_dtype, np.integer):
			# clip and cast
			resized_arr = np.clip(resized_arr, np.iinfo(src_dtype).min, np.iinfo(src_dtype).max).astype(src_dtype)
		else:
			resized_arr = resized_arr.astype(src_dtype)
	else:
		# color/rgb
		pil_img = Image.fromarray(arr.astype(np.uint8))
		pil_img = pil_img.resize((int(target_w), int(target_h)), resample=resample)
		resized_arr = np.asarray(pil_img).astype(np.uint8)

	return _numpy_to_o3d(resized_arr)


def make_pointcloud(color_o3d, depth_o3d, depth_scale=1000.0, depth_trunc=3.0, fx=None, fy=None, cx=None, cy=None):
	"""Create point cloud from color and depth Open3D images.

	Parameters match the choices in the original code, but we make them explicit and
	configurable.
	"""
	# if depth image is not provided create pointcloud by converting the color image to intensity/approx depth
	if depth_o3d is None:
		raise ValueError("No depth image was provided. Provide a depth image or generate one first")

	# RGBD creation: note that open3d expects depth in meters if you pass a depth image
	# as uint16 (common case), you should pass the `depth_scale` to convert to meters
	rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
		color_o3d, depth_o3d, depth_scale=depth_scale, depth_trunc=depth_trunc, convert_rgb_to_intensity=False
	)

	color_arr = np.asarray(color_o3d)
	height, width = int(color_arr.shape[0]), int(color_arr.shape[1])
	if fx is None:
		fx = 500.0
	if fy is None:
		fy = fx
	if cx is None:
		cx = width / 2.0
	if cy is None:
		cy = height / 2.0

	camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
	camera_intrinsic.set_intrinsics(width, height, fx, fy, cx, cy)

	pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)
	# Flip the 3D point cloud as Open3D's camera coordinate differs from OpenCV's
	if flip:
		pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
	return pcd


def main():
	parser = argparse.ArgumentParser(description="Build a point cloud from color + depth images using Open3D")
	parser.add_argument("--color", required=True, help="Path to color (RGB) image")
	parser.add_argument("--depth", required=False, help="Path to depth image (uint16 PNG or float image). Optional.")
	parser.add_argument("--depth_scale", type=float, default=1000.0, help="Scale factor to convert depth values to meters (default 1000 for mm->m)")
	parser.add_argument("--depth_trunc", type=float, default=3.0, help="Truncate depth values beyond this distance (meters)")
	parser.add_argument("--depth_max_m", type=float, help="If depth image is in integer pixels (8/16-bit) and represents 0..max distance in meters, set this to convert to meters. Overrides --depth_scale and sets depth_scale=1 if used.")
	parser.add_argument("--fx", type=float, help="Camera focal length fx (optional)")
	parser.add_argument("--fy", type=float, help="Camera focal length fy (optional)")
	parser.add_argument("--cx", type=float, help="Camera center cx (optional)")
	parser.add_argument("--cy", type=float, help="Camera center cy (optional)")
	parser.add_argument("--output", help="Path to save the generated point cloud (PLY)")
	parser.add_argument("--visualize", action="store_true", help="Show a visualizer window with the generated point cloud")
	parser.add_argument("--verbose", action="store_true", help="Show image statistics and verbose diagnostic messages")
	parser.add_argument("--no_flip", action="store_true", help="Do not apply default flip transform to generated point cloud (useful for diagnosing blank point clouds)")
	parser.add_argument("--resize", action="store_true", help="Automatically resize images when sizes differ")
	parser.add_argument("--resize_mode", choices=["depth_to_color", "color_to_depth"], default="depth_to_color", help="Direction to resize when widths/heights differ. Default: depth_to_color")
	parser.add_argument("--resize_color_method", choices=["nearest", "bilinear", "bicubic"], default="bilinear", help="Resampling method when resizing color image")
	parser.add_argument("--resize_depth_method", choices=["nearest", "bilinear", "bicubic"], default="nearest", help="Resampling method when resizing depth image (nearest is recommended)")

	args = parser.parse_args()

	color_o3d, depth_o3d = load_images(args.color, args.depth)

	# If resize requested, ensure both images have the same size by resizing depth to color (default)
	if args.resize and depth_o3d is not None:
		color_arr = np.asarray(color_o3d)
		depth_arr = np.asarray(depth_o3d)
		c_h, c_w = color_arr.shape[0], color_arr.shape[1]
		d_h, d_w = depth_arr.shape[0], depth_arr.shape[1]
		if (c_w, c_h) != (d_w, d_h):
			if args.resize_mode == "depth_to_color":
				print(f"Resizing depth image from ({d_w}, {d_h}) to match color image ({c_w}, {c_h}) using {args.resize_depth_method}")
				depth_o3d = resize_o3d_image_to(depth_o3d, c_w, c_h, is_depth=True, method=args.resize_depth_method)
			else:
				print(f"Resizing color image from ({c_w}, {c_h}) to match depth image ({d_w}, {d_h}) using {args.resize_color_method}")
				color_o3d = resize_o3d_image_to(color_o3d, d_w, d_h, is_depth=False, method=args.resize_color_method)
		else:
			print("Color and depth already match sizes; no resize performed.")
	# Analyze image stats to help diagnose empty/blank point clouds
	if args.verbose:
		analyze_image(color_o3d, "color")
		if depth_o3d is not None:
			analyze_image(depth_o3d, "depth (raw)")

	# If the user explicitly maps depth pixel values to meters
	if args.depth_max_m is not None and depth_o3d is not None:
		depth_o3d = map_depth_to_meters(depth_o3d, args.depth_max_m)
		# when mapping, the image is in meters (float), so set depth_scale to 1
		depth_scale = 1.0
	else:
		depth_scale = args.depth_scale

	pcd = make_pointcloud(color_o3d, depth_o3d, depth_scale=depth_scale, depth_trunc=args.depth_trunc, fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy, flip=(not args.no_flip))

	if args.output:
		o3d.io.write_point_cloud(args.output, pcd)
		print(f"Saved point cloud: {args.output}")

	if args.visualize:
		o3d.visualization.draw_geometries([pcd])

	# Additional diagnostics when result is empty
	print(f"Point cloud size: {len(pcd.points)} points")
	if len(pcd.points) == 0:
		print("Warning: Generated point cloud contains 0 points. Common causes: depth image contains zeros, depth_scale incorrect, or depth values are outside `depth_trunc` range.")
		if depth_o3d is not None:
			arr = np.asarray(depth_o3d)
			print(f"Depth array stats after mapping: min={float(arr.min())}, max={float(arr.max())}, dtype={arr.dtype}")
		# Point-cloud diagnostics
		try:
			bounds_min = pcd.get_min_bound()
			bounds_max = pcd.get_max_bound()
			print(f"PCD bounds (min): {bounds_min}, (max): {bounds_max}")
		except Exception as e:
			print(f"Could not compute bounds: {e}")
		return
	# if there are points, display bounding box, and sample points if verbose
	if args.verbose:
		bounds_min = pcd.get_min_bound()
		bounds_max = pcd.get_max_bound()
		print(f"PCD bounds (min): {bounds_min}, (max): {bounds_max}")
		pts = np.asarray(pcd.points)
		print(f"Sample points (first 10): {pts[:10]}" if pts.shape[0] > 0 else "No sample points")
		print("Try: adjusting --depth_scale or provide --depth_max_m to map integer image to meters.")


if __name__ == "__main__":
	main()