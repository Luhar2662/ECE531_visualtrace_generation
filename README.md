# PointCloud from Color + Depth

This repository includes a script `pointcloud.py` that builds a point cloud from a color (RGB) image and a depth image, using Open3D.

## Installing dependencies

Open a terminal and run:

```powershell
pip install open3d numpy pillow
```

## How to use

Basic usage:

```powershell
python pointcloud.py --color path/to/color.png --depth path/to/depth.png --output out.ply --visualize
```

Notes:
- `--depth` is optional but recommended. If your depth image is a 16-bit PNG with raw depth in millimeters, use the default `--depth_scale 1000` to convert to meters.
- If your depth image is already in meters (float), set `--depth_scale 1.0`.
- If your color/depth images differ in size, use `--resize` to automatically resize them so they match. Use `--resize_mode depth_to_color` (default) to resize depth to color, or `--resize_mode color_to_depth` to resize color to match depth.
- You can optionally specify camera intrinsics `--fx`, `--fy`, `--cx`, `--cy` if you know them.

If you need to read depth from a grayscale image, consider pre-processing it into a proper depth map first or provide `--depth` as a grayscale file and set `--depth_scale` accordingly.

## Example

```powershell
python pointcloud.py --color examples/color0.png --depth examples/depth0.png --output result.ply --visualize --resize
```

This will load your color/depth images, create a point cloud, save it to `result.ply` and open a visualization window.

## Troubleshooting
- If your point cloud looks flat or incorrect distances, check `--depth_scale` — a wrong scale turns meters into millimeters or vice-versa.
- If color is not mapped correctly, ensure the color image and depth image have the same width/height or use `--resize` to force resize.
 - If the point cloud is blank (no points) or contains very few points, try the following:
	- Run the script with `--verbose` to print color/depth stats: dtype, min/max/mean, and fraction of non-zero pixels.
	- If the depth image is an 8-bit (JPEG/PNG) image or otherwise not in meters, pass `--depth_max_m` to tell the script the maximum depth represented by the maximum pixel value. For example for a 8-bit depth where 255 corresponds to 3 meters: `--depth_max_m 3.0`.
	- If you know the depth image is in millimeters, keep the default `--depth_scale 1000`.
	- If the depth image is in meters (float), set `--depth_scale 1.0`.
	- Increase `--depth_trunc` if your point cloud disappears because all depths are larger than the truncation range (default 3 meters).
	- If the depth image contains mostly zeros or a small range of values, it may indicate a broken or formatted depth image — check the file in an image viewer or use `--verbose` to inspect values.

## Resampling Methods
- Color image default `--resize_color_method` is `bilinear` (smooth resizing)
- Depth image default `--resize_depth_method` is `nearest` (avoids blending different distances)
