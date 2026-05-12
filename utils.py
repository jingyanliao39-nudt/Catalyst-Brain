import pandas as pd
import os, yaml, pickle, shutil, json, ast
from project_paths import ensure_fairchem_on_path, project_path, resolve_config_paths

ensure_fairchem_on_path()

from fairchem.core.models.model_registry import model_name_to_local_file
from fairchem.core.common.relaxation.ase_utils import OCPCalculator
from ase.optimize import BFGS
from ase.calculators.singlepoint import SinglePointCalculator
import ase.io


def load_config(config_file):
    """Load configuration settings from a YAML file and print them."""
    config_file = project_path(config_file)
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    config = resolve_config_paths(config)
    
    # Print the loaded configuration
    print("Loaded Configuration:")
    print(config_file)
    print(yaml.dump(config, default_flow_style=False))
    
    return config

def setup_save_path(config, duplicate=True):
    """Set up and return a unique save path based on the configuration name."""
    
    # Extract base save directory and configuration name (without file extension)
    # config_name = os.path.splitext(config['config_name'])[0]
    save_dir = config['paths']['save_dir']
    
    # Construct the initial save path
    save_path = os.path.join(save_dir, config['config_name'])
    
    if duplicate:
        # Ensure the save path is unique by appending an index if necessary
        if os.path.exists(save_path):
            i = 1
            while os.path.exists(f"{save_path}_{i}"):
                i += 1
            save_path = f"{save_path}_{i}"

        # Create the directory if it does not exist

    os.makedirs(save_path, exist_ok=True)
    
    return save_path

# def setup_paths(system_info, paths, mode='agent'):
#     """
#     system_info: system information in config yaml
#     paths: paths in config yaml
#     mode: 'agent' or 'ocp'
#     """
#     system_id = system_info.get('system_id', None)
#     if system_id is None:
#         ads = system_info['ads_smiles']
#         bulk_id = system_info['bulk_id']
#         bulk_symbol = system_info['bulk_symbol']
#         miller = str(system_info['miller']).replace(" ", "")
#         shift = str(system_info.get('shift', 'NA'))
#         #system_id = f"{system_info['ads_smiles']}_{system_info['bulk_symbol']}_{system_info['bulk_id']}_{str(system_info['miller'])}_{str(system_info['shift'])}"
        
#     else:
#         metadata_path = paths['metadata_path']
#         info = load_info_from_metadata(system_id, metadata_path)
#         bulk_id, miller, shift, top, ads, bulk_symbol = info
#         #bulk_id, miller, _, _, ads, bulk_symbol = info
#         if mode == 'ocp':
#             shift = 'NA'
#         miller = str(miller).replace(" ", "")
#     save_name = f"{ads}_{bulk_symbol}_{bulk_id}_{miller}_{shift}"
#     save_dir = paths['save_dir']
#     # tag = "llm" if mode == "llm-guided" else "llm_heur"
#     save_path = os.path.join(save_dir, f"{save_name}")
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
#     else:
#         # make copy version
#         i = 1
#         while os.path.exists(f"{save_path}_{i}"):
#             i += 1
#         save_path = f"{save_path}_{i}"
    
#     return save_path
    #return os.path.join(save_dir, f"{system_id}")


def load_system_info(system_info, metadata_path):
    system_id = system_info.get('system_id', None)
    if system_id is None:
        ads = system_info['ads_smiles']
        bulk_id = system_info['bulk_id']
        bulk_symbol = system_info['bulk_symbol']
        miller = str(system_info['miller'])
        shift = system_info['shift']
        top = system_info['top']
    else:
        # metadata_path = paths['metadata_path']
        bulk_id, miller, shift, top, ads, bulk_symbol = load_info_from_metadata(system_id, metadata_path)
    # Normalize adsorbate notation to API-supported format
    # Example: convert "NNH" -> "*N*NH" used by OCP API
    if isinstance(ads, str):
        if '*' not in ads:
            if ads.upper() == 'NNH':
                ads = '*N*NH'
            # Add common single-atom fallbacks
            elif ads.upper() in {'H', 'O', 'N'}:
                ads = f"*{ads.upper()}"
    if not isinstance(miller, tuple):
        miller = ast.literal_eval(miller)
    print(f"bulk_id: {bulk_id}, miller: {miller}, shift: {shift}, top: {top}, ads: {ads}, bulk_symbol: {bulk_symbol}")
    return bulk_id, miller, shift, top, ads, bulk_symbol
 

def load_metadata(metadata_path, system_id):
    """Load metadata or extract system ID list if applicable."""
    if system_id == "all":
        metadata = pd.read_pickle(metadata_path)
        return list(metadata.keys()), metadata
    return [system_id], None

def save_result(result, config, save_dir):
    """Save the result as a pickle file and copy configuration for record."""
    os.makedirs(save_dir, exist_ok=True)
    result_path = os.path.join(save_dir, 'result.pkl')
    with open(result_path, 'wb') as f:
        pickle.dump(result, f)

    # Save it as text file
    try:
        result_path = os.path.join(save_dir, 'result.txt')
        with open(result_path, 'w', encoding='utf-8') as f:
            f.write(str(result))
    except Exception as e:
        print(f"An error occurred while saving the result: {e}")
    # result_path = os.path.join(save_dir, 'result.json')
    # # Save result in JSON format
    # with open(result_path, 'w') as f:
    #     json.dump(result, f)  # indent=4 for pretty printing
    
    
    # save config in the directory
    config_name = config['config_name']
    config_path = os.path.join(save_dir, f'{config_name}.yaml')
    
    # Prune non-serializable objects from config (e.g. OCPCalculator instances)
    # The _shared_calc_instance we injected causes yaml.dump to fail (it's a complex object)
    config_to_save = config.copy()
    if '_shared_calc_instance' in config_to_save:
        del config_to_save['_shared_calc_instance']
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_to_save, f)


    # Copy config file for reproducibility
    # shutil.copy('config/adsorb_aigent.yaml', save_dir)



def derive_input_prompt(system_info, metadata_path):
    system_id = system_info.get("system_id", None)
    # breakpoint()
    if system_id is not None:
        sid_to_details = pd.read_pickle(metadata_path)
        miller = sid_to_details[system_id][1]
        ads = sid_to_details[system_id][4] #.replace("*", "")  
        cat = sid_to_details[system_id][5]
        
    else:
        miller = system_info.get("miller", None)
        ads = system_info.get("ads_smiles", None)
        cat = system_info.get("bulk_symbol", None)

    assert ads is not None and cat is not None and miller is not None, "Missing system information."
    
    prompt = f"The adsorbate is {ads} and the catalyst surface is {cat} {miller}."
    return prompt

def load_text_file(file_path):
    """Loads a text file and returns its content."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        return "File not found."
    except Exception as e:
        return f"An error occurred: {e}"

def load_info_from_metadata(system_id, metadata_path):
    '''
    metadata: sid_to_details dictionary

    need to update the function to return AdsorbateSlabConfig object
    '''
    # breakpoint()
    metadata = pd.read_pickle(metadata_path)
    bulk_id = metadata[system_id][0]
    miller = metadata[system_id][1]
    shift = metadata[system_id][2]
    top = metadata[system_id][3]
    ads = metadata[system_id][4] #.replace("*", "")  
    bulk = metadata[system_id][5]
    # # if metadata[system_id][6] exists
    # if metadata[system_id][6]:
    #     num_site = metadata[system_id][6]
    # else:
    #     num_site = None  # Handle the case where metadata[system_id][6] does not exist
    return bulk_id, miller, shift, top, ads, bulk #, num_site

def load_ocp_calculator(model_name):
    """
    Helper to load the Calculator (OCP, ORB, or GRACE) once and reuse it.
    """
    model_name_str = str(model_name).lower()

    # 1. GRACE Model
    if "grace" in model_name_str:
        try:
            from tensorpotential.calculator import TPCalculator
        except ImportError:
            print("Warning: tensorpotential not found. Cannot load GRACE model.")
            raise

        print(f"[utils] Loading GRACE model from: {model_name}")
        return TPCalculator(model_name)

    # 2. ORB Model
    elif "orb" in model_name_str:
        try:
            import torch
            from orb_models.forcefield import pretrained
            from orb_models.forcefield.calculator import ORBCalculator
        except ImportError:
            print("Warning: orb_models not found. Cannot load ORB model.")
            raise

        print(f"[utils] Loading ORB model from: {model_name}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Using orb_v3_conservative_inf_omat as per user's typical usage
        orbff = pretrained.orb_v3_conservative_inf_omat(
            device=device,
            precision="float32-high",
            weights_path=model_name
        )
        return ORBCalculator(orbff, device=device)

    # 3. DPA Model (DeepMD-kit)
    elif "dpa" in model_name_str:
        try:
            from deepmd.calculator import DP
        except ImportError:
            print("Warning: deepmd-kit not found. Cannot load DPA model.")
            raise
        
        print(f"[utils] Loading DPA model from: {model_name}")
        return DP(model=model_name)
    
    # 5. Default: OCP Model
    else:
        checkpoint_path = model_name_to_local_file(model_name, local_cache='/tmp/fairchem_checkpoints/')
        return OCPCalculator(checkpoint_path=checkpoint_path, cpu=False, seed=42)

def relax_adslab(adslab, model_or_calc, save_path, memory_save=True):
    """
    Runs relaxation.
    model_or_calc: can be a str (path/name) OR an instance of OCPCalculator.
    """
    # Ensure full periodicity (required by OCP/Fairchem)
    adslab.pbc = [True, True, True]

    calc = None
    created_locally = False

    if isinstance(model_or_calc, str):
        # Fallback for legacy calls or quick tests
        checkpoint_path = model_name_to_local_file(model_or_calc, local_cache='/tmp/fairchem_checkpoints/')
        calc = OCPCalculator(checkpoint_path=checkpoint_path, cpu=False, seed=42)
        created_locally = True
    else:
        # Reuse existing calculator
        calc = model_or_calc

    adslab.calc = calc
    opt = BFGS(adslab, trajectory=save_path)
    
    final_energy = None
    final_forces = None

    try:
        opt.run(fmax=0.05, steps=500)
        # Capture properties while calculator is attached
        final_energy = adslab.get_potential_energy()
        final_forces = adslab.get_forces()
    except Exception as e:
        print(f"Relaxation warning/error: {e}")
        # Try to recover last known energy if possible
        if calc.results:
             final_energy = calc.results.get('energy')
             final_forces = calc.results.get('forces')

    finally:
        # If we created it locally, we should destroy it.
        # If passed in, we ONLY detach it from the atoms, but keep the calculator alive.
        adslab.calc = None
        if created_locally:
            del calc
            torch.cuda.empty_cache()
    
    # Attach SinglePointCalculator so downstream code works without the heavy OCP model
    if final_energy is not None and final_forces is not None:
        adslab.calc = SinglePointCalculator(adslab, energy=final_energy, forces=final_forces)

    if memory_save:
        # Check if traj file exists and is not empty before reading
        if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
            try:
                # reduced trajectory only with first and last frame
                traj = ase.io.read(save_path, ':')
                reduced_traj = [traj[0], traj[-1]]
                # remove the existing traj file
                os.remove(save_path)
                # save the reduced trajectory
                ase.io.write(save_path, reduced_traj, format='traj')  # save the reduced trajectory
            except Exception as e:
                print(f"Warning: Failed to process trajectory for memory save: {e}")
        else:
             # If file doesn't exist or is empty, we can't save a reduced traj.
             # If it was empty, we might want to clean it up.
             if os.path.exists(save_path) and os.path.getsize(save_path) == 0:
                 try:
                    os.remove(save_path)
                 except:
                    pass
    return adslab
