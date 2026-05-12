from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from .core_01_models import (
    AdaptReasoningParser,
    AdaptSolutionParser,
    AdaptCriticParser,
    AdaptIndexParser,
    TwoSolutionsParser,
    RollbackDecisionParser
)

def info_reasoning_adapter(model, parser=AdaptReasoningParser):
    output_parser = PydanticOutputParser(pydantic_object=parser)
    information_gathering_adapt_prompt = PromptTemplate(
        input_variables=["observations", "reasoning"],
        partial_variables={"format_instructions": output_parser.get_format_instructions()},
        template=(
            "You are an expert in catalyst and surface chemistry. Based on the given adsorbate and catalyst surface, "
            "observations: {observations}\n"
            "Your task is to rephrase, rewrite, and reorder each reasoning module to better identify the information "
            "needed to derive the most stable adsorption site and configuration for adsorption energy identification. "
            "Additionally, enhance the reasoning with relevant details to determine which adsorption site and configuration "
            "shows the lowest energy for the given adsorbate and catalytic surface.\n"
            "Reasoning Modules: {reasoning}.\n\n"
            "{format_instructions}"
        )
    )
    return information_gathering_adapt_prompt | model | output_parser

def solution_planner(model, parser=AdaptSolutionParser):
    output_parser = PydanticOutputParser(pydantic_object=parser)
    solution_planner_prompt = PromptTemplate(
        input_variables=["observations", "adapter_solution_reasoning"],
        partial_variables={"format_instructions": output_parser.get_format_instructions()},
        template=(
            "You are an expert in catalyst and surface chemistry.\n"
            "Your task is to find the most stable adsorption configuration of an adsorbate on the catalytic surface, "
            "including adsorption site type (ontop, bridge, hollow), binding atoms in the adsorbate and surface, their numbers, and the orientation of adsorbate (side-on, end-on, etc). "
            "Given the system: {observations}, you must operationalize "
            "the reasoning modules {adapter_solution_reasoning} to derive the most stable configuration for adsorption energy identification.\n"
            "You need to provide the most stable adsorption site & configuration with the adsorption site type, binding atoms "
            "in the adsorbate and surface, the number of those binding atoms, and the connection of those binding atoms.\n"
            "NOTE: The adsorption site can be surrounded by atoms of the same element or a combination of different elements. Avoid generating binding surface atoms by merely listing all atom types. \n"
            "Instead, determine the binding surface atoms based on the actual atomic arrangement of the surface.\n" 
            "Ensure the derived configuration is very specific and not semantically repetitive, and provide a rationale.\n"
            "Note: Do not produce invalid content. Do not repeat the same or semantically similar configuration. Stick to the given adsorbate and catalyst surface.\n\n"
            "{format_instructions}"
        )
    )
    return solution_planner_prompt | model | output_parser

def solution_reviewer(model, parser=AdaptSolutionParser):
    output_parser = PydanticOutputParser(pydantic_object=parser)
    solution_planner_prompt = PromptTemplate(
        input_variables=["initial_configuration", "relaxed_configuration", "adapter_solution_reasoning"],
        partial_variables={"format_instructions": output_parser.get_format_instructions()},
        template=(
            "You are an expert in catalysis and surface chemistry.\n"
            "Your task is to update the most stable adsorption configuration of an adsorbate on a catalytic surface.\n"
            "This includes determining the adsorption site type (on-top, bridge, hollow), identifying the binding atoms in both the adsorbate and the surface, specifying the number of binding atoms, and describing the orientation of the adsorbate (side-on, end-on, etc.).\n"
            "I have already obtained a stable relaxed configuration that shows lower energy than the initial guess of the configuration:\n"
            "Initial configuration: {initial_configuration}.\n"
            "Relaxed configuration: {relaxed_configuration}.\n"
            "You must utilize the reasoning modules {adapter_solution_reasoning} to derive a more stable configuration, referring to the initial and relaxed configurations.\n"
            "Note: Do not simply follow the relaxed configuration; instead, critically analyze and reason to derive the most stable configuration.\n"
            "You need to provide the most stable adsorption site and configuration, including the adsorption site type, the binding atoms in the adsorbate and surface, the number of those binding atoms, and the connections between those binding atoms.\n"
            "Ensure the derived configuration is very specific and not semantically repetitive, and provide a rationale.\n"
            "Note: Do not produce invalid content. Do not repeat the same or semantically similar configuration. Stick to the given adsorbate and catalyst surface.\n\n"
            "{format_instructions}"
        )
    )
    return solution_planner_prompt | model | output_parser

def structure_analyzer(model, parser=AdaptSolutionParser):
    output_parser = PydanticOutputParser(pydantic_object=parser)
    solution_planner_prompt = PromptTemplate(
        input_variables=["observations", "binding_information"],
        partial_variables={"format_instructions": output_parser.get_format_instructions()},
        template=(
            "You are an expert in catalysis and surface chemistry.\n"
            "Your task is to convert the given adsorption configuration information into a text description.\n"
            "Given adsorbate-catalyst system: {observations}\n"
            "Binding Information: {binding_information}\n"
            "The binding information is a dictionary containing the binding atoms in the adsorbate and surface, their indices, and the binding positions.\n"
            "Provide a simplified description of the adsorption configuration based on the binding information.\n"
            "Ensure the description is clear and concise.\n"
            "In the output text description, you don't need to include the specific indices.\n\n"
            "{format_instructions}"
        )
    )
    return solution_planner_prompt | model | output_parser

def surface_critic(model, parser=AdaptCriticParser):
    output_parser = PydanticOutputParser(pydantic_object=parser)
    site_type_prompt = PromptTemplate(
        input_variables=["observations", "adsorption_site_type", "binding_atoms_on_surface","knowledge"],
        partial_variables={"format_instructions": output_parser.get_format_instructions()},
        template=(
            "You are an expert in catalyst and surface chemistry.\n"
            "Observations: {observations}\n"
            "Adsorption Site Type: {adsorption_site_type}\n"
            "Binding Atoms on Surface: {binding_atoms_on_surface}\n"
            "Knowledge: {knowledge}\n"
            "Determine whether the site type matches the number of binding surface atoms.\n"
            "If the site type matches, return 1; otherwise, return 0.\n\n"
            "{format_instructions}"
        )
    )
    return site_type_prompt | model | output_parser

def adsorbate_critic(model, parser=AdaptCriticParser):
    output_parser = PydanticOutputParser(pydantic_object=parser)
    orientation_prompt = PromptTemplate(
        input_variables=["observations", "binding_atoms_in_adsorbate",  "orientation_of_adsorbate","knowledge"],
        partial_variables={"format_instructions": output_parser.get_format_instructions()},
        template=(
            "You are an expert in catalyst and surface chemistry.\n"
            "Observation: {observations}\n"
            "Binding Atoms in Adsorbate: {binding_atoms_in_adsorbate}\n"
            "Orientation: {orientation_of_adsorbate}\n"
            "Knowledge: {knowledge}\n"
            "Determine whether the orientation matches the binding atoms in the adsorbate.\n"
            "If the orientation fully matches, return 1; otherwise, return 0.\n\n"
            "{format_instructions}"
        )
    )
    return orientation_prompt | model | output_parser

def binding_indexer(model, parser=AdaptIndexParser):
    output_parser = PydanticOutputParser(pydantic_object=parser)
    prompt_template = PromptTemplate(
        input_variables=["text", "binding_atoms_in_adsorbate", "atomic_numbers"],
        partial_variables={"format_instructions": output_parser.get_format_instructions()},
        template=(
            "You are an expert in catalyst and surface chemistry. Based on the given description of the adsorption configuration: \n"
            "Text description: {text}\n"
            "Binding atoms in adsorbate (elements): {binding_atoms_in_adsorbate}\n"
            "Atomic numbers of atoms in the adsorbate: {atomic_numbers}\n"
            "Your task is to derive the indices of the binding atoms in the adsorbate. "
            "Provide the answers for the following questions (only answers, do not include the questions in the output):\n"
            "1. What are the atom indices of the adsorbate that bind to the site? (Answer: list of indices)\n"
            "Please stick to the provided answer form and keep it concise.\n"
            "Note: The indices should be 0-based.\n\n"
            "{format_instructions}"
        )
    )
    return prompt_template | model | output_parser

def two_solution_planner(model, parser=TwoSolutionsParser):
    output_parser = PydanticOutputParser(pydantic_object=parser)
    prompt_template = PromptTemplate(
        input_variables=["observations", "adapter_solution_reasoning", "previous_summary", "avoid_configs"],
        partial_variables={"format_instructions": output_parser.get_format_instructions()},
        template=(
            "You are an expert in catalyst and surface chemistry.\n"
            "System: {observations}\n"
            "Reasoning Modules: {adapter_solution_reasoning}\n"
            "Previous attempt summary (may be empty): {previous_summary}\n\n"
            "CRITICAL ANALYSIS OF RESULT DEVIATION:\n"
            "- Carefully compare the 'Expected' (proposed) vs 'Observed' (relaxed) configurations in the summary.\n"
            "- A deviation (e.g., 'Expected X -> Observed Y') indicates the proposed site X was not a stable local minimum and drifted to Y.\n"
            "- PAY ATTENTION to the 'Observed' site Y. This is a verified stable site. Analyzing where valid sites accumulate helps you map the stable regions of the surface.\n"
            "- Do NOT simply repeat similar proposals if they consistently drift to other sites or desorb.\n"
            "- If multiple different attempts all collapse to the same Observed site Y, it means Y has a large basin. Try to explore a completely different region to find other minima.\n\n"
            "Already tried configurations to AVOID repeating (canonical signature tuples): {avoid_configs}\n\n"
            "Task: Propose EXACTLY TWO distinct adsorption configurations to try next. Each must specify:\n"
            "- adsorption_site_type: ontop/bridge/hollow (lowercase)\n"
            "- binding_atoms_in_adsorbate (elements)\n"
            "- binding_atoms_on_surface (elements)\n"
            "- number_of_binding_atoms (1/2/3 consistent with site type)\n"
            "- orientation_of_adsorbate (end-on/side-on/etc)\n"
            "- reasoning\n"
            "- text: a compact natural-language description used later for binding index inference\n\n"
            "Constraints:\n"
            "- The two solutions must be meaningfully different (site_type and/or binding atoms and/or orientation).\n"
            "- Do NOT repeat any configuration that matches an 'avoid' signature; propose something new.\n"
            "- Avoid semantically repetitive variants.\n"
            "- Stay consistent with the given adsorbate and surface.\n\n"
            "{format_instructions}"
        ),
    )
    return prompt_template | model | output_parser

def rollback_reviewer(model, parser=RollbackDecisionParser):
    output_parser = PydanticOutputParser(pydantic_object=parser)
    prompt_template = PromptTemplate(
        input_variables=["observations", "history_summary", "current_round_summary"],
        partial_variables={"format_instructions": output_parser.get_format_instructions()},
        template=(
            "You are an expert in catalyst/surface chemistry and robust optimization.\n"
            "System: {observations}\n\n"
            "History summary (energies/anomalies): {history_summary}\n"
            "Current round summary: {current_round_summary}\n\n"
            "Decide whether we should rollback to a prior best round to avoid error propagation.\n"
            "If rollback is recommended, set rollback=true and provide rollback_to_round (0-based).\n"
            "If not, rollback=false. Keep reason concise.\n\n"
            "{format_instructions}"
        ),
    )
    return prompt_template | model | output_parser
