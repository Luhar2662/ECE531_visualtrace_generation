import numpy as np
import open3d as o3d
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from PIL import Image
import torch
from transformers import GLPNImageProcessor, GLPNForDepthEstimation



pcd = o3d.io.read_point_cloud("output.ply")

transformation_matrix = np.array(
    [
        [1, 0, 0, 0],  # X remains X
        [0, -1, 0, 0],  # Flip Y-axis
        [0, 0, -1, 0],  # Flip Z-axis
        [0, 0, 0, 1],
    ]
)


# Apply the transformation matrix to the point clouds
pcd.transform(transformation_matrix)

o3d.visualization.draw_geometries([pcd])