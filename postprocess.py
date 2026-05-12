"""
对 results/ 目录下的每一个 system 做：
找到 traj/*.traj（每个 traj = 一个吸附构型的弛豫轨迹）
对每个 traj：
取初始结构 & 最终结构
判断是否发生不物理的异常
只保留“正常”的构型
从这些构型中：
读出最终能量
找最低能
把结果存下来（.pkl + .txt）
"""


from tools import DetectTrajAnomaly
from ase.io import read
import glob
import os
import pickle
import csv
from tqdm import tqdm
import argparse
from project_paths import project_path
argparser = argparse.ArgumentParser()

argparser.add_argument("--dir", type=str, default="results/template_run/")
args = argparser.parse_args()


# Recursively find every traj/ under the given dir (including the dir itself)
dir_path = project_path(args.dir)
traj_dirs = []
for root, dirs, files in os.walk(dir_path):
    candidate = os.path.join(root, "traj")
    if os.path.isdir(candidate):
        traj_dirs.append(candidate)

assert len(traj_dirs) > 0, f"No traj folders found under {dir_path}"

summary_data = []

for traj_dir in traj_dirs:
    dir = os.path.dirname(traj_dir)
    
    # Collect all .traj files
    traj_files = glob.glob(os.path.join(traj_dir, "*.traj"))
    valid_energies = {}
    # Iterate over all traj files with a progress bar
    for traj_file in tqdm(traj_files, desc=f"Processing {os.path.basename(dir)}"):
        try:
            traj = read(traj_file, index=":")
            init_image = traj[0]
            fin_image = traj[-1]
            tags = init_image.get_tags()
            # Detect anomalies
            anomaly_detector = DetectTrajAnomaly(init_image, fin_image, tags)
            dissoc = anomaly_detector.is_adsorbate_dissociated()
            desorb = anomaly_detector.is_adsorbate_desorbed()
            recon = anomaly_detector.has_surface_changed()
            # If no anomalies, save the relaxed energy
            if not dissoc and not desorb and not recon:
                config_id = os.path.splitext(os.path.basename(traj_file))[0]
                relaxed_energy = fin_image.get_potential_energy()
                valid_energies[config_id] = relaxed_energy
        except Exception as e:
            print(f"Error processing {traj_file}: {e}")
            continue
    print(f"Number of traj files: {len(traj_files)}")
    print(f"Number of valid energies: {len(valid_energies)}")
    # Save valid energies if any are found
    if valid_energies:
        output_path = os.path.join(dir, "valid_energies.pkl")
        min_energy = min(valid_energies.values())
        min_config = [key for key, value in valid_energies.items() if value == min_energy][0]
        with open(output_path, 'wb') as pickle_file:
            pickle.dump(valid_energies, pickle_file)
        print(f"Minimum energy: {min_energy} for {min_config}")
        # save this as a text file
        with open(os.path.join(dir, "min_energy.txt"), "w") as f:
            f.write(f"Minimum energy: {min_energy}\n")
            f.write(f"Config index: {min_config.split('_')[1]}\n")
        
        summary_data.append([os.path.basename(dir), min_energy, min_config])
        
        #print(f"Valid energies saved to {output_path}")
    else:
        print(f"No valid energies found for {dir}")

if summary_data:
    # Sort summary_data by 'id' (the first column) in ascending order
    # Assuming 'id' is a string starting with numbers like "015_", "016_", etc.
    summary_data.sort(key=lambda x: x[0])

    output_csv_path = os.path.join(args.dir, "min_energies_summary.csv")
    with open(output_csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['id', 'min_energy', 'traj_name'])
        writer.writerows(summary_data)
    print(f"Summary CSV saved to {output_csv_path}")
