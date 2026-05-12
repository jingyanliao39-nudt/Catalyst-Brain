from collections import Counter
from tools import SiteAnalyzer, DetectTrajAnomaly
from project_paths import ensure_fairchem_on_path

ensure_fairchem_on_path()

def _multiset_signature(items) -> tuple:
    """Stable multiset signature for a list of strings (顺序无关但计数敏感).
    输入：比如 ["O", "O", "C"]（顺序可能乱）
    先 Counter 计数，得到 {"O": 2, "C": 1}
    再把它变成排序后的 tuple：(("C", 1), ("O", 2))
    """

    if not items:
        return tuple()
    c = Counter([str(x).strip() for x in items if str(x).strip()])
    return tuple(sorted(c.items(), key=lambda kv: kv[0]))


def _solution_signature(sol) -> tuple:
    """
    用于solutions去重 LLM 提议的吸附位点配置

    This is intentionally motif-level (site type + binding element multisets + denticity + orientation),
    independent of free-form text.
    """

    site_type = str(getattr(sol, "adsorption_site_type", "") or "").strip().lower()
    orient = str(getattr(sol, "orientation_of_adsorbate", "") or "").strip().lower()
    n_bind = int(getattr(sol, "number_of_binding_atoms", 0) or 0)
    ads_bind = _multiset_signature(getattr(sol, "binding_atoms_in_adsorbate", None) or [])
    surf_bind = _multiset_signature(getattr(sol, "binding_atoms_on_surface", None) or [])
    return (site_type, ads_bind, surf_bind, n_bind, orient)

def _safe_anomaly_flags(init_atoms, final_atoms):
    tags = init_atoms.get_tags()
    flags = {
        "adsorbate_dissociated": False,
        "surface_changed": False,
        "adsorbate_desorbed": False,
    }
    try:
        detector = DetectTrajAnomaly(init_atoms, final_atoms, tags)
        flags["adsorbate_dissociated"] = bool(detector.is_adsorbate_dissociated())
        flags["surface_changed"] = bool(detector.has_surface_changed())
        flags["adsorbate_desorbed"] = bool(detector.is_adsorbate_desorbed())
    except Exception as e:
        flags["error"] = f"DetectTrajAnomaly failed: {e}"
    flags["any"] = any(flags.get(k, False) for k in ["adsorbate_dissociated", "surface_changed", "adsorbate_desorbed"])
    return flags


def _apply_structure_deviation_flags(solution, anomaly_flags, structure_summary):
    """Augment anomaly flags with a 'deviated' bit based on relaxed binding topology.

    This is used to detect cases where the relaxed structure no longer matches the
    intended motif (e.g., asked for bridge=2 but ends up ontop=1 or unbound).
    """

    flags = dict(anomaly_flags or {})
    deviated = False
    reason = ""

    site_type_map = {"ontop": 1, "on-top": 1, "bridge": 2, "hollow": 3}
    site_type_map_rev = {1: "ontop", 2: "bridge", 3: "hollow"}

    expected_site_text = str(getattr(solution, "adsorption_site_type", "") or "").strip().lower()
    expected_site_from_text = site_type_map.get(expected_site_text)

    try:
        expected = int(getattr(solution, "number_of_binding_atoms", 0) or 0)
    except Exception:
        expected = 0

    # expected_site_id: prefer explicit site text when parseable; else fall back to number_of_binding_atoms
    expected_site_id = expected_site_from_text if expected_site_from_text in (1, 2, 3) else (expected if expected in (1, 2, 3) else None)

    if not isinstance(structure_summary, dict) or structure_summary.get("error"):
        deviated = True
        reason = structure_summary.get("error", "structure_summary_error") if isinstance(structure_summary, dict) else "invalid_structure_summary"
    else:
        dentate = int(structure_summary.get("dentate", 0) or 0)
        site_types = structure_summary.get("site_types", []) or []

        if dentate <= 0:
            deviated = True
            reason = "no_surface_binding_detected"
        elif expected_site_id in (1, 2, 3) and site_types and expected_site_id not in site_types:
            deviated = True
            reason = f"expected_site_type_{expected_site_id}_not_in_observed_{site_types}"
        # STRICTER CHECK: Dentate mismatch
        # If expected binding atoms is known (e.g. 1) but observed is very different (e.g. 3), flag it.
        # Allow tolerance of 1 (e.g. 2 vs 3 is okay, but 1 vs 3 is not).
        elif expected > 0 and abs(expected - dentate) >= 2:
            deviated = True
            reason = f"dentate_mismatch_expected_{expected}_observed_{dentate}"

    # Deviation is a *soft* signal: the relaxed structure may be stable and useful
    # even if it does not match the intended motif.
    flags["deviated"] = bool(deviated)
    if deviated:
        flags["deviation_reason"] = reason

        # Add structured detail to make result inspection easier.
        observed_site_ids = []
        observed_site_texts = []
        observed_dentate = None
        observed_surface_binding_elements = []
        observed_adsorbate_binding_elements = []
        observed_center_site_types = []
        try:
            if isinstance(structure_summary, dict) and not structure_summary.get("error"):
                observed_site_ids = list(structure_summary.get("site_types", []) or [])
                observed_site_texts = [site_type_map_rev.get(int(x), str(x)) for x in observed_site_ids]
                observed_dentate = structure_summary.get("dentate", None)
                observed_surface_binding_elements = list(structure_summary.get("surface_binding_elements", []) or [])
                observed_adsorbate_binding_elements = list(structure_summary.get("adsorbate_binding_elements", []) or [])
                observed_center_site_types = list(structure_summary.get("center_site_types", []) or [])
        except Exception:
            pass

        flags["deviation_detail"] = {
            "expected_site_type_text": expected_site_text or None,
            "expected_site_type_id": expected_site_id,
            "expected_number_of_binding_atoms": expected,
            "observed_site_type_ids": observed_site_ids,
            "observed_site_type_texts": observed_site_texts,
            "observed_dentate": observed_dentate,
            "observed_surface_binding_elements": observed_surface_binding_elements,
            "observed_adsorbate_binding_elements": observed_adsorbate_binding_elements,
            "observed_center_site_types": observed_center_site_types,
        }

    # Keep the original hard-anomaly aggregation (dissociation / desorption / surface change).
    # Do NOT include 'deviated' here.
    flags["any"] = any(
        bool(flags.get(k, False))
        for k in ["adsorbate_dissociated", "surface_changed", "adsorbate_desorbed"]
    )
    # Expose a convenience flag for callers.
    flags["soft_any"] = bool(flags.get("deviated", False))
    return flags


def _summarize_structure(atoms):
    try:
        sa = SiteAnalyzer(atoms, cutoff_multiplier=1.25)
        dentate = sa.get_dentate()
        site_types = sa.get_site_types()
        center_site_types = sa.get_center_site_type()
        binding_elems = []
        ads_binding_elems = []
        for b in sa.binding_info:
            binding_elems.extend(b.get("surface_binding_atom_elements", []))
            # Just append the single element string if present
            if b.get("adsorbate_binding_atom_elements"):
                ads_binding_elems.append(b.get("adsorbate_binding_atom_elements"))
        
        binding_elems = sorted(list(set(binding_elems)))
        ads_binding_elems = sorted(list(set(ads_binding_elems)))
        
        return {
            "dentate": dentate,
            "site_types": site_types,
            "center_site_types": center_site_types,
            "surface_binding_elements": binding_elems,
            "adsorbate_binding_elements": ads_binding_elems,
        }
    except Exception as e:
        return {"error": f"SiteAnalyzer failed: {e}"}


def _pick_best_candidate(candidates):
    """Pick best candidate.

    Policy:
    - Filter out *hard* anomalies first (dissociation/desorption/surface change).
    - Among remaining, prefer non-deviated over deviated, then lowest energy.
    - If everything is hard-anomalous, fall back to energy among all candidates.
    """

    def hard_anom(c):
        return bool((c.get("anomaly") or {}).get("any", False))

    def deviated(c):
        return bool((c.get("anomaly") or {}).get("deviated", False))

    valid = [c for c in candidates if not hard_anom(c)]
    pool = valid if valid else candidates
    pool = [c for c in pool if c.get("energy") is not None]
    if not pool:
        return None

    return min(pool, key=lambda c: (1 if deviated(c) else 0, c["energy"]))
