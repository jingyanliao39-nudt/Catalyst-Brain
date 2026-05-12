import os
import glob
import multiprocessing
import torch
from utils import load_config, load_ocp_calculator
from .core_05_engine import singlerun_adsorb_agent_iterative

def run_batch_configs(batch_configs, device_id, worker_idx):
    """
    Worker function to process a batch of configurations on a specific GPU.
    """
    # 1. Set Device
    if torch.cuda.is_available():
        if device_id is not None:
            torch.cuda.set_device(device_id)
        print(f"[Worker {worker_idx}] Started on GPU {torch.cuda.current_device()} with {len(batch_configs)} tasks.")
    
    # 2. Process all configs in this batch using the optimized single-run logic
    #    The single-run logic now handles loading the model ONCE.
    #    However, singlerun_adsorb_agent_iterative currently loads the model inside itself.
    #    We rely on singlerun_adsorb_agent_iterative's internal logic to load/unload.
    #    BUT, to truly optimize, we should load `shared_calc` HERE and pass it down.
    #    Let's modify singlerun_adsorb_agent_iterative slightly to accept an optional calculator.
    
    # For now, relying on singlerun to load/unload is safer *if* it cleans up correctly.
    if batch_configs:
        gnn_model_path = batch_configs[0]['agent_settings']['gnn_model']
        print(f"[Worker {worker_idx}] Loading shared model: {gnn_model_path}")
        shared_calc = load_ocp_calculator(gnn_model_path)
    else:
        shared_calc = None
    
    success_count = 0
    for conf in batch_configs:
        try:
             # Inject the pre-loaded calculator into the config or pass it some other way?
             # Since singlerun signatures are fixed in many places, maybe we attach it to config temporarily.
             # Or we can modify singlerun to look for it.
             conf['_shared_calc_instance'] = shared_calc
             singlerun_adsorb_agent_iterative(conf)
             success_count += 1
        except Exception as e:
            print(f"[Worker {worker_idx}] Error processing {conf.get('config_name', 'unknown')}: {e}")
            import traceback
            traceback.print_exc()

    # cleanup
    print(f"[Worker {worker_idx}] Finished. Success: {success_count}/{len(batch_configs)}")
    if shared_calc is not None:
        del shared_calc
    torch.cuda.empty_cache()


def multirun_adsorb_aigent(setting_config):
    #breakpoint()
    agent_settings = setting_config['agent_settings']
    paths = setting_config['paths']
    num_workers = int(agent_settings.get("num_workers", 16))

    system_path = paths['system_dir']
    system_config_files = glob.glob(system_path + '/*.yaml')
    system_config_files.sort()
    
    configs_to_run = []
    for i, config_file in enumerate(system_config_files):
        config_name = os.path.basename(config_file)
        config_name = config_name.split('.')[0]
        
        config = load_config(config_file)
        config['config_name'] = config_name
        config['agent_settings'] = agent_settings
        config['paths'] = paths
        configs_to_run.append(config)

    total_configs = len(configs_to_run)
    print(f"Starting execution with {num_workers} workers on {total_configs} configs.")
    
    # Split configs into batches for each worker
    if num_workers < 1: num_workers = 1
    
    # If fewer configs than workers, reduce workers
    if total_configs < num_workers:
        num_workers = total_configs
        
    batches = [[] for _ in range(num_workers)]
    for i, conf in enumerate(configs_to_run):
        batches[i % num_workers].append(conf)
    
    processes = []
    num_gpus = torch.cuda.device_count()
    
    try:
        if num_workers > 0:
             multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    for i in range(num_workers):
        if not batches[i]: continue
        
        # Round-robin assign GPU
        device_id = i % num_gpus if num_gpus > 0 else None
        
        p = multiprocessing.Process(
            target=run_batch_configs,
            args=(batches[i], device_id, i)
        )
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()

    print('============ Completed! ============')
