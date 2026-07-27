from __future__ import annotations

"""Utilities for neutral and ionic electron configurations.

The neutral configurations are sourced from the cited Wikipedia table,
then expanded into explicit subshell occupancies for ionization and spin-counting.
"""

import re
from functools import lru_cache

from quick_mag.electron_configuration_data import (
    ATOMIC_NUMBERS,
    CONCISE_NEUTRAL_ELECTRON_CONFIGURATIONS,
)

ElectronSubshell = tuple[int, str, int]
ElectronConfiguration = tuple[ElectronSubshell, ...]

ORBITAL_CAPACITY: dict[str, int] = {"s": 2, "p": 6, "d": 10, "f": 14}
ORBITAL_ORDER: dict[str, int] = {"s": 0, "p": 1, "d": 2, "f": 3}
AUFBAU_SEQUENCE: tuple[tuple[int, str], ...] = (
    (1, "s"),
    (2, "s"), (2, "p"),
    (3, "s"), (3, "p"), (4, "s"),
    (3, "d"), (4, "p"), (5, "s"),
    (4, "d"), (5, "p"), (6, "s"),
    (4, "f"), (5, "d"), (6, "p"), (7, "s"),
    (5, "f"), (6, "d"), (7, "p"), (8, "s"),
)

_SUBSHELL_TOKEN = re.compile(r"(\d+)([spdf])(\d+)")


def _normalize_element_symbol(element: str) -> str:
    if not element:
        raise ValueError("Element symbol cannot be empty.")
    return element[0].upper() + element[1:].lower()


def _canonicalize_configuration(
    subshell_counts: dict[tuple[int, str], int],
) -> ElectronConfiguration:
    return tuple(
        (n, orbital, electron_count)
        for (n, orbital), electron_count in sorted(
            subshell_counts.items(),
            key=lambda item: (item[0][0], ORBITAL_ORDER[item[0][1]]),
        )
        if electron_count > 0
    )


def _parse_explicit_subshell(token: str) -> ElectronSubshell:
    match = _SUBSHELL_TOKEN.fullmatch(token)
    if match is None:
        raise ValueError(f"Unsupported subshell token: {token!r}")
    n_str, orbital, electron_count_str = match.groups()
    return int(n_str), orbital, int(electron_count_str)


def _build_neutral_electron_configurations() -> dict[str, ElectronConfiguration]:
    neutral_configurations: dict[str, ElectronConfiguration] = {}
    ordered_symbols = sorted(ATOMIC_NUMBERS, key=ATOMIC_NUMBERS.get)
    for symbol in ordered_symbols:
        subshell_counts: dict[tuple[int, str], int] = {}
        for token in CONCISE_NEUTRAL_ELECTRON_CONFIGURATIONS[symbol].split():
            if token.startswith("[") and token.endswith("]"):
                core_symbol = token[1:-1]
                for n, orbital, electron_count in neutral_configurations[core_symbol]:
                    subshell_counts[(n, orbital)] = electron_count
                continue

            n, orbital, electron_count = _parse_explicit_subshell(token)
            subshell_counts[(n, orbital)] = electron_count

        configuration = _canonicalize_configuration(subshell_counts)
        total_electrons = sum(electron_count for _, _, electron_count in configuration)
        if total_electrons != ATOMIC_NUMBERS[symbol]:
            raise ValueError(
                f"Neutral configuration for {symbol} has {total_electrons} electrons, "
                f"expected {ATOMIC_NUMBERS[symbol]}."
            )
        neutral_configurations[symbol] = configuration

    return neutral_configurations


NEUTRAL_ELECTRON_CONFIGURATIONS = _build_neutral_electron_configurations()


@lru_cache(maxsize=None)
def _get_explicit_valence_subshells(element: str) -> tuple[tuple[int, str], ...]:
    symbol = _normalize_element_symbol(element)
    subshells: list[tuple[int, str]] = []
    for token in CONCISE_NEUTRAL_ELECTRON_CONFIGURATIONS[symbol].split():
        if token.startswith("[") and token.endswith("]"):
            continue
        n, orbital, _ = _parse_explicit_subshell(token)
        subshells.append((n, orbital))

    return tuple(
        sorted(
            subshells,
            key=lambda subshell: (subshell[0], ORBITAL_ORDER[subshell[1]]),
            reverse=True,
        )
    )


@lru_cache(maxsize=None)
def get_neutral_electron_configuration(element: str) -> ElectronConfiguration:
    symbol = _normalize_element_symbol(element)
    try:
        return NEUTRAL_ELECTRON_CONFIGURATIONS[symbol]
    except KeyError as exc:
        raise KeyError(f"No neutral electron configuration is stored for {element!r}.") from exc


@lru_cache(maxsize=None)
def get_ionic_electron_configuration(
    element: str,
    oxidation_state: int,
) -> ElectronConfiguration:
    symbol = _normalize_element_symbol(element)
    oxidation_state = int(oxidation_state)
    subshell_counts = {
        (n, orbital): electron_count
        for n, orbital, electron_count in get_neutral_electron_configuration(symbol)
    }

    if oxidation_state > 0:
        electrons_to_remove = oxidation_state
        total_electrons = sum(subshell_counts.values())
        if electrons_to_remove > total_electrons:
            raise ValueError(
                f"Cannot remove {oxidation_state} electrons from neutral {symbol}."
            )

        valence_subshells = _get_explicit_valence_subshells(symbol)
        for subshell in valence_subshells:
            if electrons_to_remove <= 0:
                break
            occupancy = subshell_counts.get(subshell, 0)
            if occupancy <= 0:
                continue
            removable = min(electrons_to_remove, occupancy)
            subshell_counts[subshell] = occupancy - removable
            electrons_to_remove -= removable

        while electrons_to_remove > 0:
            occupied_subshells = [
                key for key, electron_count in subshell_counts.items()
                if electron_count > 0
            ]
            highest_subshell = max(
                occupied_subshells,
                key=lambda key: (key[0], ORBITAL_ORDER[key[1]]),
            )
            removable = min(electrons_to_remove, subshell_counts[highest_subshell])
            subshell_counts[highest_subshell] -= removable
            electrons_to_remove -= removable
    elif oxidation_state < 0:
        electrons_to_add = -oxidation_state
        while electrons_to_add > 0:
            available_subshell = None
            for subshell in AUFBAU_SEQUENCE:
                occupancy = subshell_counts.get(subshell, 0)
                if occupancy < ORBITAL_CAPACITY[subshell[1]]:
                    available_subshell = subshell
                    break

            if available_subshell is None:
                raise ValueError(
                    f"Cannot add {-oxidation_state} electrons to neutral {symbol} "
                    "with the supported subshell model."
                )

            room = ORBITAL_CAPACITY[available_subshell[1]] - subshell_counts.get(available_subshell, 0)
            added = min(electrons_to_add, room)
            subshell_counts[available_subshell] = subshell_counts.get(available_subshell, 0) + added
            electrons_to_add -= added

    return _canonicalize_configuration(subshell_counts)


def count_unpaired_electrons(configuration: ElectronConfiguration) -> int:
    total_unpaired = 0
    for _, orbital, electron_count in configuration:
        capacity = ORBITAL_CAPACITY[orbital]
        half_filled = capacity // 2
        if electron_count <= half_filled:
            total_unpaired += electron_count
        else:
            total_unpaired += capacity - electron_count
    return total_unpaired


@lru_cache(maxsize=None)
def count_unpaired_electrons_for_ion(element: str, oxidation_state: int) -> int:
    return count_unpaired_electrons(
        get_ionic_electron_configuration(element, oxidation_state)
    )
