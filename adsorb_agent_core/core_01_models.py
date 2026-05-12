from pydantic import BaseModel, Field
from typing import List, Optional

class AdaptReasoningParser(BaseModel):
    """Information gathering plan"""

    other: Optional[str] = Field(default=None, description="other information about the adsorbate-catalyst system")

    adapted_prompts: List[str] = Field(
        description="Adapted and rephrased prompts to better identify the information required to solve the task"
    )
    preamble: Optional[str] = Field(
        default=None, description="preamble to reasoning modules"
    )  


class AdaptSolutionParser(BaseModel):
    """Information gathering plan"""

    # human_solution: Optional[List[str]] = Field(description="Human help in solving the problem")

    adsorption_site_type: str = Field(description="Type of adsorption site (e.g., ontop, bridge, hollow; in lower case)")
    binding_atoms_in_adsorbate: List[str] = Field(description="Binding atoms in the adsorbate")
    binding_atoms_on_surface: List[str] = Field(description="Binding atoms on the surface")
    number_of_binding_atoms: int = Field(description="Number of binding atoms on the surface")
    orientation_of_adsorbate: str = Field(description="Orientation of the adsorbate (e.g., end-on, side-on)")
    reasoning: str = Field(description="Reasoning for the derived configuration")
    text: str = Field(description="Textual description of the derived configuration")

class AdaptCriticParser(BaseModel):
    """Information gathering plan"""
    human_solution: Optional[List[str]] = Field(default=None, description="Human help in solving the problem")
    solution: int = Field(description="1 if the observation is correct, otherwise 0")

class AdaptIndexParser(BaseModel):
    """Plan for gathering information about binding atom indices."""
    
    human_solution: Optional[List[str]] = Field(default=None, description="Human-provided help in solving the problem")

    solution: List[int] = Field(
        description="Indices of the binding atoms in the adsorbate (0-based indexing)"
    )


class CandidateSolution(BaseModel):
    """A single adsorption-configuration candidate."""

    adsorption_site_type: str = Field(description="Type of adsorption site (e.g., ontop, bridge, hollow; in lower case)")
    binding_atoms_in_adsorbate: List[str] = Field(description="Binding atoms in the adsorbate")
    binding_atoms_on_surface: List[str] = Field(description="Binding atoms on the surface")
    number_of_binding_atoms: int = Field(description="Number of binding atoms on the surface")
    orientation_of_adsorbate: str = Field(description="Orientation of the adsorbate (e.g., end-on, side-on)")
    reasoning: str = Field(description="Reasoning for the derived configuration")
    text: str = Field(description="Textual description of the derived configuration")


class TwoSolutionsParser(BaseModel):
    """Return exactly two distinct candidate adsorption configurations."""

    solutions: List[CandidateSolution] = Field(
        description="A list of exactly two distinct candidate adsorption configurations"
    )


class RollbackDecisionParser(BaseModel):
    """Decide whether to rollback to a prior best round."""

    rollback: bool = Field(description="Whether to rollback to a prior best state")
    rollback_to_round: Optional[int] = Field(
        description="0-based round index to rollback to (if rollback=true)",
        default=None,
    )
    reason: str = Field(description="Concise reason for the decision")
