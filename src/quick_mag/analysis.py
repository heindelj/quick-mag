from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable

from quick_mag.oxidation_states import common_oxidation_states, oxidation_state_score
from quick_mag.radii import find_ionic_radii


@dataclass(frozen=True)
class AveragedCrystalRadius:
    element: str
    oxidation_state: int | None
    crystal_radius: float | None
    match_count: int = 0
    warning: str = ""


@dataclass(frozen=True)
class ElementRadiusSummary:
    element: str
    average_oxidation_state: float
    average_crystal_radius: float | None
    site_count: int
    oxidation_state_breakdown: Dict[int, int]
    missing_oxidation_states: tuple[int, ...] = ()


@dataclass(frozen=True)
class GoldschmidtToleranceFactorResult:
    tolerance_factor: float | None
    a_site: ElementRadiusSummary | None
    b_site: ElementRadiusSummary | None
    x_site: ElementRadiusSummary | None
    warnings: tuple[str, ...] = ()


def compute_goldschmidt_tolerance_factor(
    a_site_radius: float,
    b_site_radius: float,
    x_site_radius: float,
) -> float:
    return (a_site_radius + x_site_radius) / (
        math.sqrt(2.0) * (b_site_radius + x_site_radius)
    )


def average_crystal_radius_for_oxidation_state(
    element: str,
    oxidation_state: int,
) -> AveragedCrystalRadius:
    try:
        matches = tuple(
            record
            for record in find_ionic_radii(element, charge=oxidation_state)
            if record.source == "shannon"
        )
    except KeyError:
        return AveragedCrystalRadius(
            element=element,
            oxidation_state=oxidation_state,
            crystal_radius=None,
            warning=(
                f"No Shannon crystal-radius table exists for element {element!r}. "
                "Goldschmidt analysis is unavailable for this oxidation state."
            ),
        )
    if not matches:
        return AveragedCrystalRadius(
            element=element,
            oxidation_state=oxidation_state,
            crystal_radius=None,
            warning=(
                f"No Shannon crystal-radius entries matched {element}{oxidation_state:+d}. "
                "Using this oxidation state in Goldschmidt analysis is not supported yet."
            ),
        )

    mean_radius = sum(record.crystal_radius for record in matches) / len(matches)
    return AveragedCrystalRadius(
        element=element,
        oxidation_state=oxidation_state,
        crystal_radius=mean_radius,
        match_count=len(matches),
    )


def average_crystal_radius_for_element(element: str) -> AveragedCrystalRadius:
    try:
        matches = tuple(
            record for record in find_ionic_radii(element) if record.source == "shannon"
        )
    except KeyError:
        return AveragedCrystalRadius(
            element=element,
            oxidation_state=None,
            crystal_radius=None,
            warning=(
                f"No Shannon crystal-radius table exists for element {element!r}. "
                "Element-average radius lookup is unavailable."
            ),
        )
    if not matches:
        return AveragedCrystalRadius(
            element=element,
            oxidation_state=None,
            crystal_radius=None,
            warning=(
                f"No Shannon crystal-radius entries matched element {element!r}. "
                "Element-average radius lookup is unavailable."
            ),
        )

    mean_radius = sum(record.crystal_radius for record in matches) / len(matches)
    return AveragedCrystalRadius(
        element=element,
        oxidation_state=None,
        crystal_radius=mean_radius,
        match_count=len(matches),
    )


def ranked_shannon_oxidation_states(element: str) -> tuple[int, ...]:
    try:
        available_states = {
            record.charge
            for record in find_ionic_radii(element)
            if record.source == "shannon"
        }
    except KeyError:
        return ()

    return tuple(
        oxidation_state
        for oxidation_state in common_oxidation_states(element)
        if oxidation_state in available_states
    )


def fallback_crystal_radius_for_oxidation_state(
    element: str,
    oxidation_state: int,
) -> AveragedCrystalRadius:
    exact_radius = average_crystal_radius_for_oxidation_state(element, oxidation_state)
    if exact_radius.crystal_radius is not None:
        return exact_radius

    ranked_states = ranked_shannon_oxidation_states(element)
    if ranked_states:
        fallback_state = min(
            ranked_states,
            key=lambda candidate: (
                abs(candidate - oxidation_state),
                0 if candidate == 0 or (candidate > 0) == (oxidation_state > 0) else 1,
                oxidation_state_score(element, candidate),
                abs(candidate),
                candidate,
            ),
        )
        fallback_radius = average_crystal_radius_for_oxidation_state(
            element, fallback_state
        )
        if fallback_radius.crystal_radius is not None:
            return AveragedCrystalRadius(
                element=element,
                oxidation_state=fallback_state,
                crystal_radius=fallback_radius.crystal_radius,
                match_count=fallback_radius.match_count,
                warning=(
                    exact_radius.warning
                    or (
                        f"Fell back from {element}{oxidation_state:+d} to "
                        f"{element}{fallback_state:+d} for rendering radius selection."
                    )
                ),
            )

    return average_crystal_radius_for_element(element)


def crystal_radius_for_rendering(
    element: str,
    oxidation_state: int | None,
) -> AveragedCrystalRadius:
    if oxidation_state is None:
        return average_crystal_radius_for_element(element)
    return fallback_crystal_radius_for_oxidation_state(element, oxidation_state)


def _iter_nonzero_oxidation_states(
    oxidation_state_breakdown: Dict[int, int],
) -> Iterable[tuple[int, int]]:
    for oxidation_state, site_count in sorted(oxidation_state_breakdown.items()):
        if site_count > 0:
            yield oxidation_state, site_count


def summarize_element_radius(
    element: str,
    oxidation_state_breakdown: Dict[int, int],
) -> tuple[ElementRadiusSummary, list[str]]:
    total_sites = sum(
        site_count for _oxidation_state, site_count in _iter_nonzero_oxidation_states(oxidation_state_breakdown)
    )
    if total_sites <= 0:
        return (
            ElementRadiusSummary(
                element=element,
                average_oxidation_state=0.0,
                average_crystal_radius=None,
                site_count=0,
                oxidation_state_breakdown={},
            ),
            [f"{element} had no nonzero site counts in the oxidation-state breakdown."],
        )

    weighted_oxidation = 0.0
    weighted_radius = 0.0
    matched_sites = 0
    warnings: list[str] = []
    missing_oxidation_states: list[int] = []

    for oxidation_state, site_count in _iter_nonzero_oxidation_states(oxidation_state_breakdown):
        weighted_oxidation += oxidation_state * site_count
        averaged_radius = average_crystal_radius_for_oxidation_state(element, oxidation_state)
        if averaged_radius.crystal_radius is None:
            missing_oxidation_states.append(oxidation_state)
            if averaged_radius.warning:
                warnings.append(averaged_radius.warning)
            continue
        weighted_radius += averaged_radius.crystal_radius * site_count
        matched_sites += site_count

    average_radius = weighted_radius / matched_sites if matched_sites else None
    if matched_sites != total_sites:
        warnings.append(
            f"{element} used only {matched_sites} of {total_sites} sites in the averaged crystal-radius estimate "
            "because some oxidation states are missing from the Shannon table."
        )

    return (
        ElementRadiusSummary(
            element=element,
            average_oxidation_state=weighted_oxidation / total_sites,
            average_crystal_radius=average_radius,
            site_count=total_sites,
            oxidation_state_breakdown=dict(oxidation_state_breakdown),
            missing_oxidation_states=tuple(missing_oxidation_states),
        ),
        warnings,
    )


def estimate_goldschmidt_tolerance_factor(
    oxidation_state_distribution: Dict[str, Dict[int, int]],
) -> GoldschmidtToleranceFactorResult:
    summaries: list[ElementRadiusSummary] = []
    warnings: list[str] = []
    for element, breakdown in sorted(oxidation_state_distribution.items()):
        summary, summary_warnings = summarize_element_radius(element, breakdown)
        summaries.append(summary)
        warnings.extend(summary_warnings)

    anion_candidates = [
        summary
        for summary in summaries
        if any(oxidation_state < 0 and site_count > 0 for oxidation_state, site_count in summary.oxidation_state_breakdown.items())
    ]
    cation_candidates = [
        summary
        for summary in summaries
        if any(oxidation_state > 0 and site_count > 0 for oxidation_state, site_count in summary.oxidation_state_breakdown.items())
    ]

    if len(anion_candidates) != 1:
        warnings.append(
            "Goldschmidt tolerance factor requires a single identifiable X-site anion species."
        )
        return GoldschmidtToleranceFactorResult(
            tolerance_factor=None,
            a_site=None,
            b_site=None,
            x_site=anion_candidates[0] if len(anion_candidates) == 1 else None,
            warnings=tuple(warnings),
        )

    if len(cation_candidates) != 2:
        warnings.append(
            "Goldschmidt tolerance factor requires two identifiable cation species for the A and B sites."
        )
        return GoldschmidtToleranceFactorResult(
            tolerance_factor=None,
            a_site=None,
            b_site=None,
            x_site=anion_candidates[0],
            warnings=tuple(warnings),
        )

    ranked_cations = sorted(
        cation_candidates,
        key=lambda summary: (
            float("-inf") if summary.average_crystal_radius is None else summary.average_crystal_radius,
            summary.element,
        ),
        reverse=True,
    )
    a_site, b_site = ranked_cations[0], ranked_cations[1]
    x_site = anion_candidates[0]

    if (
        a_site.average_crystal_radius is None
        or b_site.average_crystal_radius is None
        or x_site.average_crystal_radius is None
    ):
        warnings.append(
            "Goldschmidt tolerance factor could not be computed because at least one inferred site radius is unavailable."
        )
        return GoldschmidtToleranceFactorResult(
            tolerance_factor=None,
            a_site=a_site,
            b_site=b_site,
            x_site=x_site,
            warnings=tuple(warnings),
        )

    tolerance_factor = compute_goldschmidt_tolerance_factor(
        a_site.average_crystal_radius,
        b_site.average_crystal_radius,
        x_site.average_crystal_radius,
    )
    return GoldschmidtToleranceFactorResult(
        tolerance_factor=tolerance_factor,
        a_site=a_site,
        b_site=b_site,
        x_site=x_site,
        warnings=tuple(warnings),
    )


__all__ = [
    "AveragedCrystalRadius",
    "ElementRadiusSummary",
    "GoldschmidtToleranceFactorResult",
    "average_crystal_radius_for_element",
    "average_crystal_radius_for_oxidation_state",
    "crystal_radius_for_rendering",
    "compute_goldschmidt_tolerance_factor",
    "estimate_goldschmidt_tolerance_factor",
    "fallback_crystal_radius_for_oxidation_state",
    "ranked_shannon_oxidation_states",
    "summarize_element_radius",
]
