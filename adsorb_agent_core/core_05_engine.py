import os
import ast
import torch
import numpy as np
from ase.io import read
from utils import load_config, setup_save_path, derive_input_prompt, load_text_file, load_info_from_metadata, relax_adslab, save_result, load_ocp_calculator
from project_paths import ensure_fairchem_on_path
from .core_02_llm import _init_llm
from .core_03_chains import (
    two_solution_planner,
    info_reasoning_adapter,
    solution_planner,
    
    rollback_reviewer,
    binding_indexer
)
from .core_04_analysis import (
    _solution_signature,
    _multiset_signature,
    _safe_anomaly_flags,
    _summarize_structure,
    _pick_best_candidate,
    _apply_structure_deviation_flags
)

ensure_fairchem_on_path()
from fairchem.data.oc.core import Adsorbate, Bulk, Slab, AdsorbateSlabConfig

def _propose_dual_solutions(planner, observations, reasoning_result, previous_summary, seen_solution_sigs, max_regen_attempts):
    solutions = []
    local_sigs = set()
    regen_attempt = 0
    avoid_preview = list(seen_solution_sigs)
    avoid_configs = "\\n".join([str(s) for s in avoid_preview[-30:]])

    while len(solutions) < 2 and regen_attempt < max_regen_attempts:
        plan = planner.invoke({
            "observations": observations,
            "adapter_solution_reasoning": reasoning_result.adapted_prompts,
            "previous_summary": previous_summary,
            "avoid_configs": avoid_configs,
        })

        for sol in (plan.solutions or []):
            sig = _solution_signature(sol)
            if sig in seen_solution_sigs or sig in local_sigs:
                continue
            solutions.append(sol)
            local_sigs.add(sig)
            if len(solutions) >= 2:
                break

        regen_attempt += 1
        if len(solutions) < 2:
            avoid_configs = "\\n".join([str(s) for s in list(seen_solution_sigs | local_sigs)[-30:]])

    # If less than 2 solutions found, retry with updated avoid list
    if len(solutions) < 2:
        plan = planner.invoke({
            "observations": observations,
            "adapter_solution_reasoning": reasoning_result.adapted_prompts,
            "previous_summary": previous_summary,
            "avoid_configs": avoid_configs,
        })
        for sol in (plan.solutions or [])[:2]:
            if len(solutions) >= 2:
                break
            solutions.append(sol)
    
    return solutions

def singlerun_adsorb_agent_iterative(config):
    # If passed implicitly via config (from run_batch_configs)
    _injected_calc = config.get('_shared_calc_instance', None)

    # Set GPU device for this process if specified (Legacy handling, now handled by run_batch_configs)
    if "device_id" in config and torch.cuda.is_available() and _injected_calc is None:
        torch.cuda.set_device(config["device_id"])
        print(f"[Worker Debug] PID: {os.getpid()} using GPU: {torch.cuda.current_device()} for config: {config['config_name']}")

    system_info = config["system_info"]
    agent_settings = config["agent_settings"]
    paths = config["paths"]

    metadata_path = paths["metadata_path"]
    question_path = paths["question_path"]
    knowledge_path = paths["knowledge_path"]
    bulk_db_path = paths["bulk_db_path"]
    ads_db_path = paths["ads_db_path"]

    llm_model = _init_llm(agent_settings)

    gnn_model = agent_settings["gnn_model"]
    mode = agent_settings["mode"]
    init_multiplier = agent_settings.get("init_multiplier", 1.0)
    random_ratio = agent_settings.get("random_ratio", 0.0)
    max_rounds = int(agent_settings.get("max_rounds", 3))
    rollback_patience = int(agent_settings.get("rollback_patience", 1))
    use_result_llm = bool(agent_settings.get("use_result_llm", False))
    energy_worsen_threshold = float(agent_settings.get("energy_worsen_threshold", 0.5))

    observations = derive_input_prompt(system_info, metadata_path)

    print("observations:", observations)
    

    reasoning_questions = load_text_file(question_path)
    ###注意 这个没用上
    knowledge_statements = load_text_file(knowledge_path)
    num_site = int(system_info["num_site"] * init_multiplier)

    save_dir = setup_save_path(config, duplicate=False)
    # Check if FULLY completed (look for result.pkl)
    result_pkl = os.path.join(save_dir, "result.pkl")
    if os.path.exists(result_pkl):
        print(f"Skip: {config['config_name']} fully completed (result.pkl exists)")
        return None
    
    # Optional: cleanup partial runs if you want to restart fresh
    traj_dir = os.path.join(save_dir, "traj")
    # if os.path.exists(traj_dir):
    #     import shutil
    #     shutil.rmtree(traj_dir) # Uncomment to auto-clean partial runs

    print("Reasoning step...")
    reasoning_adapter = info_reasoning_adapter(model=llm_model)
    reasoning_result = reasoning_adapter.invoke({
        "observations": observations,
        "reasoning": reasoning_questions,
    })

    # Load Model ONCE per worker (or use injected one)
    if _injected_calc:
         shared_calc = _injected_calc
         # print(f"[Worker] Using injected shared GNN model on device {torch.cuda.current_device()}")
    else:
        print(f"[Worker] Loading GNN model from scratch: {gnn_model} on device {torch.cuda.current_device()}")
        shared_calc = load_ocp_calculator(gnn_model)

    # Load base structures once
    print("Loading adslabs...")
    if system_info.get("system_id", None) is not None:
        system_id = system_info.get("system_id", None)
        info = load_info_from_metadata(system_id, metadata_path)
    else:
        info = [
            system_info.get("bulk_id"),
            system_info.get("miller"),
            system_info.get("shift", None),
            system_info.get("top", None),
            system_info.get("ads_smiles"),
            system_info.get("bulk_symbol"),
        ]

    bulk_id, miller, shift, top, ads, bulk_symbol = info
    if not isinstance(miller, tuple):
        miller = ast.literal_eval(miller)

    bulk = Bulk(bulk_src_id_from_db=bulk_id, bulk_db_path=bulk_db_path)
    slabs = Slab.from_bulk_get_specific_millers(bulk=bulk, specific_millers=miller)
    slab = None
    for slab_candidate in slabs:
        if np.isclose(slab_candidate.shift, shift, atol=0.01):
            if top is None or slab_candidate.top == top:
                slab = slab_candidate
                break
    if slab is None:
        print(f"Failed to match slab with requested shift/top. Skipping...")
        return None
    adsorbate = Adsorbate(adsorbate_smiles_from_db=ads, adsorbate_db_path=ads_db_path)

    os.makedirs(traj_dir, exist_ok=True)

    history = []
    rollback_events = []
    best_state = None
    best_round = None
    error_streak = 0
    previous_summary = ""
    # Cache binding indices by (adsorbate_smiles, binding_atom_elements multiset)
    binding_index_cache = {}
    # De-duplicate motif-level LLM solutions across rounds/candidates
    seen_solution_sigs = set() #记录了目前为止（从第0轮到当前轮之前）所有 AI 提议并运行过的吸附位点配置
    max_regen_attempts = int(agent_settings.get("max_regen_attempts", 4))

    for r in range(max_rounds):
        print(f"Iter round {r}...")
        planner = two_solution_planner(model=llm_model)
        global_seen_before = set(seen_solution_sigs) #以前见过的配置“存”在这个变量

        # Collect two NEW (non-repeated) solutions; if LLM repeats, regenerate.
        solutions = _propose_dual_solutions(
            planner, 
            observations, 
            reasoning_result, 
            previous_summary, 
            seen_solution_sigs, 
            max_regen_attempts
        )
        round_solution_sigs = [_solution_signature(sol) for sol in solutions]
        seen_solution_sigs |= set(round_solution_sigs)

        round_candidates = []
        seen_this_round = set()
        
        for k, sol in enumerate(solutions):
            print(f"Candidate {k}: {sol.adsorption_site_type}, {sol.orientation_of_adsorbate}, bind_ads={sol.binding_atoms_in_adsorbate}, bind_surf={sol.binding_atoms_on_surface}, n_bind={sol.number_of_binding_atoms}")
            # If LLM still produced a repeated motif (fallback path), skip compute and force regeneration next round.
            sig = _solution_signature(sol)
            if sig in seen_this_round:
                round_candidates.append({
                    "candidate_id": k,
                    "solution": sol.dict(),
                    "error": "Duplicate solution signature (within round); skipped",
                    "energy": None,
                    "anomaly": {"any": True, "adsorbate_desorbed": False, "adsorbate_dissociated": False, "surface_changed": False, "deviated": True},
                })
                continue
            seen_this_round.add(sig)
            if sig in global_seen_before:
                round_candidates.append({
                    "candidate_id": k,
                    "solution": sol.dict(),
                    "error": "Duplicate solution signature (already tried); skipped",
                    "energy": None,
                    "anomaly": {"any": True, "adsorbate_desorbed": False, "adsorbate_dissociated": False, "surface_changed": False, "deviated": True},
                })
                continue

            if mode == "llm-guided":
                cache_key = (
                    str(ads),_multiset_signature(getattr(sol, "binding_atoms_in_adsorbate", []) or []),)
                if cache_key in binding_index_cache:
                    adsorbate.binding_indices = np.array(binding_index_cache[cache_key], dtype=int)
                else:
                    try:
                        index_adapter = binding_indexer(model=llm_model)
                        
                        index_result = index_adapter.invoke({
                            "text": sol.text,
                            "binding_atoms_in_adsorbate": getattr(sol, "binding_atoms_in_adsorbate", None),
                            "atomic_numbers": adsorbate.atoms.numbers,
                        })
                        adsorbate.binding_indices = np.array(index_result.solution, dtype=int)
                        binding_index_cache[cache_key] = adsorbate.binding_indices.copy()
                    except Exception as e:
                        msg = f"binding_indexer failed: {e}"
                        round_candidates.append({
                            "candidate_id": k,
                            "solution": sol.dict(),
                            "error": msg,
                            "energy": None,
                            "anomaly": {"any": True, "adsorbate_desorbed": False, "adsorbate_dissociated": False, "surface_changed": False, "deviated": True},
                        })
                        continue

            cutoff_multiplier = 1.5
            adslabs = []
            while not adslabs and cutoff_multiplier <= 2:
                try:
                    # OPTIMIZATION: Try more sites per LLM suggestion to find the true basin.
                    # Default increased from 1 to 20 to mimic heuristic coverage for the selected site type.
                    num_sites_effective = int(agent_settings.get("num_sites_per_llm", 20))
                    
                    # Replaced LLM selector with deterministic logic inside AdsorbateSlabConfig
                    adslabs_ = AdsorbateSlabConfig(
                        slab,
                        adsorbate,
                        num_sites=num_sites_effective,
                        mode=mode,
                        site_type=sol.adsorption_site_type,
                        site_atoms=sol.binding_atoms_on_surface,
                        random_ratio=random_ratio,
                        cutoff_multiplier=cutoff_multiplier,
                        orient=sol.orientation_of_adsorbate,
                        site_selector=None,
                        site_atoms_start_radius=agent_settings.get("site_atoms_start_radius", None),
                        site_atoms_radius_step=float(agent_settings.get("site_atoms_radius_step", 0.2)),
                        site_atoms_max_radius=float(agent_settings.get("site_atoms_max_radius", 4.0)),
                    )
                    adslabs = list(adslabs_.atoms_list)
                except Exception:
                    adslabs = []
                cutoff_multiplier += 0.05

            if not adslabs:
                round_candidates.append({
                    "candidate_id": k,
                    "solution": sol.dict(),
                    "error": "No configurations generated",
                    "energy": None,
                    "anomaly": {"any": True, "adsorbate_desorbed": False, "adsorbate_dissociated": False, "surface_changed": False},
                })
                continue

            # ---------------------------------------------------------------------------------
            # OPTIMIZATION: Initial Energy Screening (Smart Start)
            # Evaluate all generated candidates with a single-point energy calculation (cheap).
            # Pick the single best starting structure for full relaxation (expensive).
            # This enables "Basin Hopping" without paying the cost of N relaxations.
            # ---------------------------------------------------------------------------------
            if len(adslabs) > 1:
                try:
                    screened_candidates = []
                    print(f"  > Screening {len(adslabs)} initial structures by single-point energy...")
                    for idx, atoms_cand in enumerate(adslabs):
                        # Ensure PBC is set (required for OCP/Fairchem)
                        atoms_cand.pbc = [True, True, True]
                        # Use the shared calculator for quick energy check
                        atoms_cand.calc = shared_calc
                        try:
                            # Single point energy - fast
                            e_init = atoms_cand.get_potential_energy()
                            screened_candidates.append((e_init, atoms_cand))
                        except Exception as e:
                            print(f"    Screening error on cand {idx}: {e}")
                            pass
                    
                    if screened_candidates:
                        # Sort by lowest energy
                        screened_candidates.sort(key=lambda x: x[0])
                        best_start_e, best_start_atoms = screened_candidates[0]
                        print(f"  > Best initial structure found: E={best_start_e:.4f} eV. Proceeding with this one.")
                        # Replace list with just the best one to reduce relaxation cost to 1
                        adslabs = [best_start_atoms]
                    else:
                        print(f"  > Warning: Screening yielded no valid energies. Fallback: selecting random/first candidate to save compute.")
                        adslabs = adslabs[:1]

                except Exception as e:
                    print(f"  > Screening process failed ({e}). Falling back to relaxing the first candidate.")
                    adslabs = adslabs[:1] # Fallback to first one
            # ---------------------------------------------------------------------------------

            # Optimization: Relax candidates (Screening above limits this to typically 1)
            # This bridges the gap between LLM "reasoning" (high-level) and "brute force" (finding the exact minimum).
            best_sub_result = None

            for sub_idx, adslab_init in enumerate(adslabs):
                init_atoms = adslab_init.copy()
                save_path = os.path.join(traj_dir, f"round_{r}_cand_{k}_sub_{sub_idx}.traj")
                
                try:
                    # Pass the SHARED calculator instance
                    adslab_relaxed = relax_adslab(adslab_init, shared_calc, save_path)
                    
                    if adslab_relaxed.calc is None:
                        continue # Skip failed relaxations
                    
                    energy_val = float(adslab_relaxed.get_potential_energy())
                    
                    # [FIX] Calculate anomalies using the SAVED TRAJECTORY to ensure consistency with postprocess.py
                    # Previously: anomaly_val = _safe_anomaly_flags(init_atoms, adslab_relaxed)
                    # This led to discrepancies where Agent thought structure was valid, but postprocess (reading from disk) saw dissociation.
                    try:
                        traj_check = read(save_path, index=":")
                        anomaly_val = _safe_anomaly_flags(traj_check[0], traj_check[-1])
                    except Exception as e_traj:
                        print(f"Warning: Could not read back traj for anomaly check: {e_traj}. Fallback to memory objects.")
                        anomaly_val = _safe_anomaly_flags(init_atoms, adslab_relaxed)
                    
                    # Store result if it's the first or better than previous
                    # We prefer: No hard anomaly > Lower Energy
                    
                    is_hard_anomaly = anomaly_val.get("any", False)
                    
                    if best_sub_result is None:
                        best_sub_result = (energy_val, adslab_relaxed, init_atoms, save_path, anomaly_val)
                    else:
                        current_best_energy = best_sub_result[0]
                        current_best_anomaly = best_sub_result[4].get("any", False)
                        
                        # Logic:
                        # 1. If currently have hard anomaly but new one doesn't -> take new one
                        # 2. If both have/don't have hard anomaly -> take lower energy
                        if current_best_anomaly and not is_hard_anomaly:
                            best_sub_result = (energy_val, adslab_relaxed, init_atoms, save_path, anomaly_val)
                        elif (current_best_anomaly == is_hard_anomaly) and (energy_val < current_best_energy):
                            best_sub_result = (energy_val, adslab_relaxed, init_atoms, save_path, anomaly_val)
                            
                except Exception as e:
                    print(f"Relaxation failed for sub-candidate {sub_idx}: {e}")
                    continue

            if best_sub_result is None:
                 # All failed
                 round_candidates.append({
                    "candidate_id": k,
                    "solution": sol.dict(),
                    "error": "All sub-candidates failed relaxation",
                    "energy": None,
                    "anomaly": {"any": True},
                })
                 continue
                 
            # Unpack best result
            (energy, adslab_relaxed, init_atoms, save_path, anomaly) = best_sub_result
            
            # Post-processing for the best candidate
            structure_summary = _summarize_structure(adslab_relaxed)
            # Re-check deviation flags for the winner (it might be different than the first one)
            anomaly = _apply_structure_deviation_flags(sol, anomaly, structure_summary)

            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

            round_candidates.append({
                "candidate_id": k,
                "solution": sol.dict(),
                "traj": save_path,
                "energy": energy,
                "anomaly": anomaly,
                "structure": structure_summary,
                "cutoff_multiplier": cutoff_multiplier,
            })

        chosen = _pick_best_candidate(round_candidates)
        round_record = {
            "round": r,
            "candidates": round_candidates,
            "chosen_candidate": chosen["candidate_id"] if chosen else None,
        }
        history.append(round_record)

        if chosen is None or chosen.get("energy") is None:
            error_streak += 1
            # Include concrete failure reasons so the planner avoids infeasible motifs next round.
            fail_bits = []
            for c in round_candidates:
                sol = c.get("solution", {}) or {}
                sig = _solution_signature(type("_Tmp", (), sol)) if isinstance(sol, dict) else None
                err = c.get("error")
                anomaly_any = (c.get("anomaly") or {}).get("any", False)
                fail_bits.append(
                    {
                        "candidate": c.get("candidate_id"),
                        "site_type": sol.get("adsorption_site_type") or sol.get("site_type"),
                        "site_atoms": sol.get("binding_atoms_on_surface") or sol.get("site_atoms"),
                        "orient": sol.get("orientation_of_adsorbate") or sol.get("orient"),
                        "error": err,
                        "anomaly_any": anomaly_any,
                        "sig": sig,
                    }
                )
            previous_summary = f"Round {r}: failed (no valid energies). Details: {fail_bits}"
        else:
            chosen_energy = chosen["energy"]
            chosen_hard_anomaly = chosen.get("anomaly", {}).get("any", False)
            chosen_deviated = chosen.get("anomaly", {}).get("deviated", False)

            # Update global best if it has an energy and no *hard* anomaly.
            # (Deviation does not disqualify a stable relaxed structure.)
            if not chosen_hard_anomaly and (best_state is None or chosen_energy < best_state["energy"]):
                best_state = chosen
                best_round = r

            # If current looks bad, count error streak (hard failures / severe worsening only).
            if chosen_hard_anomaly:
                error_streak += 1
            elif best_state is not None and chosen_energy > best_state["energy"] + energy_worsen_threshold:
                error_streak += 1
            else:
                error_streak = 0

            # Build summary for next round (optionally via LLM)
            dev_note = ""
            if chosen_deviated:
                detail = chosen.get('anomaly', {}).get('deviation_detail', {})
                exp = detail.get('expected_site_type_text', 'unknown')
                obs_ids = detail.get('observed_site_type_ids', [])
                obs_texts = detail.get('observed_site_type_texts', [])
                
                # If the drift resulted in a new global best, frame it as a discovery to exploit.
                if best_round == r:
                    dev_note = f" SUCCESSFUL DRIFT: An unexpected stable configuration was found! (Expected {exp} -> Observed {obs_texts}). The system drifted to a LOWER ENERGY state. Please try to design directly for this new observed structure in the next round!"
                else:
                    dev_note = f" WARNING: SITE DRIFT DETECTED (Expected {exp} -> Observed {obs_texts}). Use this insight to avoid unstable sites."

            previous_summary = (
                f"Round {r} chosen candidate {chosen['candidate_id']}: energy={chosen_energy:.6f} eV.{dev_note} "
                f"Structure Summary: {chosen.get('structure', {})}. "
                f"Anomaly Flags: {chosen.get('anomaly', {})}."
            )

        # Optional LLM-driven rollback decision
        if best_state is not None and error_streak >= rollback_patience:
            do_rollback = True
            rollback_to = best_round
            reason = f"error_streak={error_streak} >= rollback_patience={rollback_patience}"
            if use_result_llm:
                try:
                    # [FIX] Hide energy of anomalous candidates so LLM doesn't hallucinate about "stable" low energy structures.
                    hist_sum = str([
                        {
                            h["round"]: [
                                (
                                    c.get("energy") if not (c.get("anomaly", {}) or {}).get("any", False) else "INVALID_ANOMALY",
                                    (c.get("anomaly", {}) or {}).get("any", False),
                                    (c.get("anomaly", {}) or {}).get("deviated", False),
                                )
                                for c in h["candidates"]
                            ]
                        }
                        for h in history
                    ])
                    rr = rollback_reviewer(model=llm_model)
                    rr_out = rr.invoke({
                        "observations": observations,
                        "history_summary": hist_sum,
                        "current_round_summary": previous_summary,
                    })

                    do_rollback = bool(rr_out.rollback)
                    rollback_to = rr_out.rollback_to_round if rr_out.rollback_to_round is not None else best_round
                    reason = rr_out.reason
                except Exception as e:
                    reason = f"rollback_reviewer failed, fallback deterministic: {e}"

            if do_rollback:
                event = {
                    "round": r,
                    "rollback_to_round": int(rollback_to) if rollback_to is not None else None,
                    "best_round": int(best_round) if best_round is not None else None,
                    "reason": reason,
                    "error_streak": int(error_streak),
                }
                rollback_events.append(event)
                history[-1]["rollback_event"] = event
                print(f"[ROLLBACK] to round {rollback_to} (best_round={best_round}): {reason}")
                
                 # [FIX] Force best_state to match the safe rollback target.
                # If rollback_reviewer chose a specific round (e.g. 4) over the global best (e.g. 2),
                # it means the global best is likely unstable/desorbed. We must 'forget' it.
                if rollback_to is not None and best_round is not None and rollback_to != best_round:
                    target_candidate = None
                    for h_record in history:
                        if h_record["round"] == rollback_to:
                            c_id = h_record.get("chosen_candidate")
                            if c_id is not None:
                                for cand in h_record["candidates"]:
                                    if cand["candidate_id"] == c_id:
                                        target_candidate = cand
                                        break
                            break
                    
                    if target_candidate:
                        print(f"[ROLLBACK FIX] Reverting global best_state from Round {best_round} to Round {rollback_to} (Discarding potentially invalid lower energy state).")
                        best_state = target_candidate
                        best_round = rollback_to

                # Provide the best solution text so LLM knows what to optimize around
                best_sol_text = best_state.get('solution', {}).get('text', 'Unknown configuration')
                best_sol_reasoning = best_state.get('solution', {}).get('reasoning', 'No reasoning provided')
                previous_summary = (
                    f"Rolling back to round {rollback_to} (best_round={best_round}). Reason: {reason}. "
                    f"The BEST structure so far was: {best_sol_text}. "
                    f"The original reasoning for this best structure was: {best_sol_reasoning}. "
                    "Try to refine this configuration or explore variations nearby based on this reasoning."
                )
                
                error_streak = 0
                # Continue next round using best summary; we do not recompute earlier rounds.
                continue

    result_dict = {
        "system": info,
        "history": history,
        "rollback_events": rollback_events,
        "best_round": best_round,
        "best": best_state,
        "config_no_count": sum(len(h["candidates"]) for h in history),
    }
    print("Result:", result_dict)
    save_result(result_dict, config, save_dir)
    
    # Clean up the shared calculator BEFORE EXITING IF WE CREATED IT
    if _injected_calc is None:
        del shared_calc
        torch.cuda.empty_cache()

    return result_dict
