import os
import subprocess
import threading

# Base paths for the gs_model and colmap_poses directories
gs_model_base_path = r"/home/magnus/phd/3dgs/output/"
raw_data_base_path = r"/home/magnus/Downloads"
sibr_viewer_bin_path = os.path.expanduser('/home/magnus/phd/gaussian-splatting/SIBR_viewers/install/bin')

# Total number of scenes to process
total_scenes = 270
# Settings
baselines = [2.5, 5, 10]
num_images_to_generate_per_baseline = 100


def execute_sibr_viewer(start_scene, end_scene):
    for i in range(start_scene, end_scene):
        try:
            # Generate folder names
            output_path = f"{gs_model_base_path}scene_{i:04d}"
            # SIBR Viewer command
            sibr_viewer_command = f'{os.path.join(sibr_viewer_bin_path, "SIBR_gaussianViewer_app")} -m {output_path} -n {i} -r 1 --iteration 30000 -b "{baselines[0]}, {baselines[1]}, {baselines[2]}" -c {num_images_to_generate_per_baseline}'
            print(f"Executing: {sibr_viewer_command}")
            subprocess.run(sibr_viewer_command, check=True, shell=True)
        except Exception as e:
            print(e)


def main():
    threads = []
    num_threads = 7
    scenes_per_thread = total_scenes // num_threads

    for i in range(num_threads):
        start_scene = i * scenes_per_thread
        # Ensure the last thread processes any remaining scenes due to integer division
        end_scene = (i + 1) * scenes_per_thread if i < num_threads - 1 else total_scenes
        thread = threading.Thread(target=execute_sibr_viewer, args=(start_scene, end_scene))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()


if __name__ == "__main__":
    main()
