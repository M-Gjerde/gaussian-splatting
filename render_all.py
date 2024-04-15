import os
import subprocess
import datetime
# Base paths for the gs_model and colmap_poses directories
import time

gs_model_base_path = r"/home/magnus/phd/3dgs/output/"
raw_data_base_path = r"/home/magnus/Downloads"
sibr_viewer_bin_path = os.path.expanduser('/home/magnus/phd/gaussian-splatting/SIBR_viewers/install/bin')

# Total number of scenes to process
total_scenes = 270

# Iterate through all scene directories
for i in range(0, 270):
    # Determine the part number (increment every 20 scenes)
    part_number = i // 20 + 1  # Integer division, starts with part 1 for scenes 0-19

    try:
        # Generate folder names
        output_path = f"{gs_model_base_path}scene_{i:04d}"

        # SIBR Viewer command
        sibr_viewer_command = f"{os.path.join(sibr_viewer_bin_path, 'SIBR_gaussianViewer_app')} -m {output_path} -n {i}"
        print(f"Executing: {sibr_viewer_command}")
        subprocess.run(sibr_viewer_command, check=True, shell=True)
        break
    except Exception as e:
        print(e)
        continue
