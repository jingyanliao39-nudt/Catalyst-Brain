"""
单进程 + 串行 + asyncio

遍历 system_dir/*.yaml
对每个 system：
调用 find_adsorbate_binding_sites
根据 (miller, shift, top) 选 slab
保存：
full_result.json保存 返回的全部 slab 列表（只按 miller 做过初筛）
找出 shift 与 config 里的 shift 在 ±0.01 内且 top 相同的那个 slab，记录其 configs 数量为 num_site
把 results.slabs 替换成仅包含这个匹配的 slab，再序列化成 selected_result.json
只是在“寻找 / 生成吸附位点（吸附构型）
"""


import asyncio
import os
import ast
import glob
import json
import numpy as np
from pathlib import Path
from project_paths import ensure_fairchem_on_path

ensure_fairchem_on_path()

from fairchem.data.oc.core import Slab
from fairchem.demo.ocpapi import find_adsorbate_binding_sites, keep_slabs_with_miller_indices
from utils import * 
import random
import time

init_time = time.time()

config = load_config('config/ocp_demo.yaml')
paths = config['paths']
system_path = paths['system_dir']
system_config_files = glob.glob(system_path + '/*.yaml')

# system_config_files.sort()
# randomly shuffle the list

#system_config_files = random.sample(system_config_files, len(system_config_files)) #全量随机打乱


def dump_pretty_json(path, json_str):
    """Persist JSON with indentation for readability."""
    obj = json.loads(json_str)
    Path(path).write_text(json.dumps(obj, indent=2))


for i, config_file in enumerate(system_config_files): #串行遍历
    config_name = os.path.basename(config_file)
    config_name = config_name.split('.')[0]
    org_config = load_config(config_file)
    config = org_config.copy()
    config['paths'] = paths
    config['config_name'] = config_name
    
    print(f"Processing: {config_name}")
    save_dir = setup_save_path(config, duplicate=False)
    # # if savd_dir already exists, skip this config
    # if os.path.exists(save_dir):
    #     breakpoint()
    #     print(f"Skip: {config_name} already exists")
    #     continue
    # else:
    #     os.makedirs(save_dir, exist_ok=True)
    full_result_path = os.path.join(save_dir, "full_result.json")
    if os.path.exists(full_result_path):
        print(f"Skip: {config_name} – full_result.json already exists")
        continue

    # breakpoint()
    system_info = config['system_info']
    metadata_path = paths['metadata_path']
    try:
        bulk_id, miller, shift, top, ads, bulk_symbol = load_system_info(system_info, metadata_path)
    except:
        print(f"Error: {config_name} is not a valid config file")
        continue
    


    async def main():
        slab_selected = None
        """
        根据 bulk_id 找到对应 bulk
        切 slab（同一个 miller 可能有多个 shift/top）
        对每个 slab：
        用一套“规则/几何策略”在表面上找 候选吸附位点（top/bridge/hollow 等）
        把吸附物 ads 放到这些位点上，生成一组 Adsorbate–Slab configurations
        最终返回 results，里面包含：
        results.slabs: 多个 slab 的结果
        每个 slab 里有多个 configs（每个 config 基本就对应一个“吸附位点/吸附构型”）
        最终写回的 num_site = len(slab_selected.configs) 就是在数“这个 slab 上一共有多少个候选位点/构型”。        
        """
        results = await find_adsorbate_binding_sites(adsorbate=ads, 
                                                     bulk=bulk_id, 
                                                     adslab_filter=keep_slabs_with_miller_indices([miller]))

        for slab in results.slabs:
            slab_top = slab.slab.metadata.top
            slab_shift = slab.slab.metadata.shift
            if np.isclose(slab_shift, shift, atol=0.01) and slab_top == top:
                slab_selected = slab
                org_config['system_info']['num_site'] = len(slab_selected.configs)
                config['system_info']['num_site'] = len(slab_selected.configs)
                break
        


        # assert slab_selected is not None, f"Slab with miller {miller} and shift {shift} not found"
        # Check if slab_selected was found; if not, log and skip this config
        if slab_selected is None:
            print(f"Warning: No matching slab found for miller {miller} and shift {shift} in {config_name}")
            return  # Exit the `main` function for this config
        # breakpoint()
        dump_pretty_json(os.path.join(save_dir, "full_result.json"), results.to_json())
        
        results.slabs = [slab_selected]
        dump_pretty_json(os.path.join(save_dir, "selected_result.json"), results.to_json())
        
        # Update the original yaml with the number of adsorption configurations
        with open(config_file, 'w') as f:
            yaml.dump(org_config, f)
        
        # Save the config file for record
        with open(os.path.join(save_dir, 'config.yaml'), 'w') as f:
            yaml.dump(config, f)

    # Run the async function
    asyncio.run(main())

    # # Pause every 10 iterations
    # if (i + 1) % 10 == 0:
    #     print("Pausing for 5 seconds...")
    time.sleep(3)

fin_time = time.time()
print(f"Total time: {(fin_time - init_time)/60:.2f} minutes for {len(system_config_files)} systems")
print('============ Completed! ============')
