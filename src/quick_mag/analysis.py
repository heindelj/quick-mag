from __future__ import annotations

from dataclasses import dataclass

from quick_mag.radii import find_ionic_radii


@dataclass(frozen=True)
class AveragedCrystalRadius:
    element: str
    oxidation_state: int | None
    crystal_radius: float | None
    match_count: int = 0
    warning: str = ""


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
                "No radius is available for this oxidation state."
            ),
        )
    if not matches:
        return AveragedCrystalRadius(
            element=element,
            oxidation_state=oxidation_state,
            crystal_radius=None,
            warning=(
                f"No Shannon crystal-radius entries matched {element}{oxidation_state:+d}."
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


def _shannon_charges(element: str) -> tuple[int, ...]:
    """Oxidation states this element has a Shannon crystal radius for."""
    try:
        return tuple(
            sorted(
                {
                    record.charge
                    for record in find_ionic_radii(element)
                    if record.source == "shannon"
                }
            )
        )
    except KeyError:
        return ()


def fallback_crystal_radius_for_oxidation_state(
    element: str,
    oxidation_state: int,
) -> AveragedCrystalRadius:
    exact_radius = average_crystal_radius_for_oxidation_state(element, oxidation_state)
    if exact_radius.crystal_radius is not None:
        return exact_radius

    # No exact entry: borrow the nearest tabulated charge, preferring one of the
    # same sign, then the one closest to neutral. This only fires for oxidation
    # states Shannon does not tabulate, which are chemically implausible ones.
    available_states = _shannon_charges(element)
    if available_states:
        fallback_state = min(
            available_states,
            key=lambda candidate: (
                abs(candidate - oxidation_state),
                0 if candidate == 0 or (candidate > 0) == (oxidation_state > 0) else 1,
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


__all__ = [
    "AveragedCrystalRadius",
    "average_crystal_radius_for_element",
    "average_crystal_radius_for_oxidation_state",
    "crystal_radius_for_rendering",
    "fallback_crystal_radius_for_oxidation_state",
]
