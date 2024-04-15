from math import log10, sqrt
import cv2
import numpy as np
import open3d as o3d
from pathlib import Path
import os

from matplotlib import pyplot as plt


def disparity_to_depth(disparity_image, focal_length, baseline):
    # Convert disparity image to depth map
    disparity_image[disparity_image == 0] = 0.1  # Avoid division by zero
    depth_map = (focal_length * baseline) / disparity_image
    return depth_map


def load_image(file_path, ao=False):
    """
    Load an image from file. Can load in grayscale or ao.
    """
    if ao:
        # Load disparity image (assuming 16-bit PNG)
        image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        image = image / 65535.0  # Adjust scale depending on your disparity computation method

    else:
        # Load disparity image (assuming 16-bit PNG)
        image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        image = image / 65535.0  # Adjust scale depending on your disparity computation method
        # image = median_filter_float32(image, 5) * 255
        image *= 1160
    return image


def display_inlier_outlier(cloud, ind, plot=True):
    #inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    #outlier_cloud.paint_uniform_color([1, 0, 0])
    #inlier_cloud.paint_uniform_color([0.4, 0.4, 0.4])
    if plot:
        o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                          zoom=0.3412,
                                          front=[0.4257, -0.2125, -0.8795],
                                          lookat=[2.6172, 2.0475, 1.532],
                                          up=[-0.0694, -0.9768, 0.2024])
    num_removed = len(outlier_cloud.points)
    return num_removed


def create_point_cloud_from_depth(depth_map, intrinsic_matrix):
    """
    Generate and view point cloud from depth map.
    """
    # Create Open3D depth image from depth map
    depth_image = o3d.geometry.Image(depth_map.astype(np.float32))
    # Create Open3D intrinsic object
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(width=depth_map.shape[1], height=depth_map.shape[0],
                             fx=intrinsic_matrix[0, 0], fy=intrinsic_matrix[1, 1],
                             cx=intrinsic_matrix[0, 2], cy=intrinsic_matrix[1, 2])

    # Generate point cloud from depth image
    point_cloud = o3d.geometry.PointCloud.create_from_depth_image(depth_image, intrinsic,
                                                                  depth_scale=1.0, depth_trunc=1000.0,
                                                                  stride=1)
    return point_cloud


def create_colored_point_cloud(depth_map, color_image, intrinsic_matrix):
    """
    Generate and view colored point cloud from depth map and color image.
    """
    # Create Open3D depth image from depth map
    depth_image = o3d.geometry.Image(depth_map.astype(np.float32))
    # Create Open3D color image from color image
    color_image_o3d = o3d.geometry.Image(color_image.astype(np.uint8))
    # Create Open3D intrinsic object
    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(width=depth_map.shape[1], height=depth_map.shape[0],
                             fx=intrinsic_matrix[0, 0], fy=intrinsic_matrix[1, 1],
                             cx=intrinsic_matrix[0, 2], cy=intrinsic_matrix[1, 2])
    # Generate RGBD image from depth and color images
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_image_o3d, depth_image,
                                                                    depth_scale=1.0, depth_trunc=1000.0,
                                                                    convert_rgb_to_intensity=False)
    # Generate point cloud
    point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
    # Visualize the point cloud
    o3d.visualization.draw_geometries([point_cloud])


gs_model_base_path = r"C:\Users\mgjer\PycharmProjects\GaussianSplattingViewer\out_baseline_05\scene_"
download_dir = "C:\\Users\\mgjer\\Downloads"

original_paths = []
nerf_paths = []

parts = 2
# Loop through all parts
for part in range(1, parts + 1):
    part_dir_name = f"raw_data_v1_part{part}"
    part_dir_path = os.path.join(download_dir, part_dir_name)

    # Check if the directory exists
    if os.path.exists(part_dir_path) and os.path.isdir(part_dir_path):
        # Loop through all subdirectories in the part directory
        for subfolder in os.listdir(part_dir_path):
            subfolder_path = os.path.join(part_dir_path, subfolder)
            # Check if it's a directory
            if os.path.isdir(subfolder_path):
                original_paths.append(subfolder_path)

    # Loop through all parts
for part in range(1, parts + 1):
    part_dir_name = f"stereo_dataset_v1_part{part}"
    part_dir_path = os.path.join(download_dir, part_dir_name)

    # Check if the directory exists
    if os.path.exists(part_dir_path) and os.path.isdir(part_dir_path):
        # Loop through all subdirectories in the part directory
        for subfolder in os.listdir(part_dir_path):
            subfolder_path = os.path.join(part_dir_path, subfolder)
            # Check if it's a directory
            if os.path.isdir(subfolder_path):
                nerf_paths.append(subfolder_path)

removed_3dgs_pts = []
removed_nerf_pts = []
np.random.seed(42)

skip_parts = [7, 19, 30, 37, 21]

percentages = []
# Iterate through all scene directories
for i in range(0, parts * 20):
    # Determine the part number (increment every 20 scenes)
    part_number = i // 20 + 1  # Integer division, starts with part 1 for scenes 0-19
    if i in skip_parts:
        continue
    # Generate folder names
    scene_folder = f"{gs_model_base_path}{i:04d}/depth"
    file_names_original = os.listdir(os.path.join(original_paths[i], "images"))
    file_names_nerf = os.listdir(os.path.join(nerf_paths[i], "Q", "baseline_0.50", "disparity"))

    random_numbers = np.random.choice(range(100), 10, replace=False)

    for x in random_numbers:
        try:
            original_file_name = os.path.join(os.path.join(original_paths[i], "images"), file_names_original[x])

            img_name = str(x) + ".png"
            rendered_file_name = os.path.join(scene_folder, img_name)
            nerf_file_name = os.path.join(os.path.join(nerf_paths[i], "Q", "baseline_0.50", "disparity"),
                                          file_names_nerf[x])

            nerf_AO_file_name = os.path.join(os.path.join(nerf_paths[i],  "Q", "AO"),
                                          file_names_nerf[x])
            focal_length = 3439.3083700227126  # Example focal length in pixels
            baseline = 0.5  # Example baseline in meters
            div = 4
            intrinsic_matrix = np.array([[3439.3083700227126 / div, 0, 2292 / div],  # fx, 0, cx
                                         [0, 3445.0110843463276 / div, 1030 / div],  # 0, fy, cy
                                         [0, 0, 1]])  # Intrinsic matrix of the camera

            disparity_3dgs = load_image(rendered_file_name)

            # Create a mask where the pixels that are not black (i.e., not 0) are set to 255
            black_pixels_mask = cv2.inRange(disparity_3dgs, 0, 0)

            # Count the non-black (non-zero) pixels in the mask
            non_black_count = cv2.countNonZero(black_pixels_mask)

            # The number of black pixels is the total number of pixels minus the non-black ones
            total_pixels = disparity_3dgs.size
            black_count = total_pixels - non_black_count

            percentage = (black_count/total_pixels) * 100
            if percentage < 99.5:
                print(f"skipped scene {i}, image: {x}. Percentage: {percentage:0.2f}%")
                continue

            disparity_nerf = load_image(nerf_file_name)
            disparity_nerf_ao = load_image(nerf_AO_file_name, True)
            mask = (disparity_nerf_ao < 0.5).astype(np.uint8)

            filtered_elements_count = np.count_nonzero(mask)
            total_points = disparity_nerf_ao.size

            # Calculate the percentage of filtered points
            percentage_filtered = (filtered_elements_count / total_points) * 100
            percentages.append(percentage_filtered)
            print(f"filtered elements: {filtered_elements_count}, percentage: {percentage_filtered}%")


            disparity_nerf = cv2.bitwise_and(disparity_nerf, disparity_nerf, mask=mask)

            depth_map_3dgs = disparity_to_depth(disparity_3dgs, focal_length, baseline)
            pc_3dgs = create_point_cloud_from_depth(depth_map_3dgs, intrinsic_matrix)
            depth_map_nerf = disparity_to_depth(disparity_nerf, focal_length, baseline)
            pc_nerf = create_point_cloud_from_depth(depth_map_nerf, intrinsic_matrix)

            plot = False
            cl, ind_3dgs = pc_3dgs.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            removed_3dgs = display_inlier_outlier(pc_3dgs, ind_3dgs, plot=plot)

            cl, ind_nerf = pc_nerf.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            removed_nerf = display_inlier_outlier(pc_nerf, ind_nerf, plot=plot)

            #o3d.visualization.draw_geometries([pc_3dgs, pc_nerf])

            removed_3dgs_pts.append(removed_3dgs)
            removed_nerf_pts.append(removed_nerf)

            value_nerf = 0
            value_3dgs = 0
            if removed_nerf < removed_3dgs:
                winner = "nerf"
            else:
                winner = "3dgs"
            print(
                f"Scene: {i:04d}. Winner is: {winner} num removed nerf: {removed_nerf:.3f} pts, 3dgs: {removed_3dgs:.3f} pts. FileNames: {Path(rendered_file_name).name}, Nerf: {Path(nerf_file_name).name}")

        except Exception as e:
            print(e)


folder = "AO_th_0.5"
if not os.path.exists(folder):
    os.mkdir(folder)


print("Mean percentage filtered", np.mean(np.array(percentages)), np.std(np.array(percentages)))
np.save(f"./{folder}/removed_nerf_pts.npy", np.array(removed_nerf_pts))
np.save(f"./{folder}/removed_3dgs_pts.npy", np.array(removed_3dgs_pts))