"""Per-site oxidation states set by hand, layered over the predicted assignment.

The model picks one charge-balanced distribution per structure -- the lowest-energy
one -- and expands it onto the sites. That is a starting point, not an answer: a
cell with a defect, a deliberately mixed-valence one, or a site the geometry-free
energy model simply gets wrong all need a state the enumeration will not produce.
This module holds the edits that say so, and nothing else: it does not decide when
an edit is allowed, only what one *means* once it is made.

An edit has one of two scopes, and which one it gets is decided by what it was made
on rather than by a mode the user has to remember:

**Cell scope** -- an edit made on a *unit cell*. A unit cell has no sites that are
not part of the repeating motif, so naming one names it everywhere: every boundary
image the 3D view draws, and every cell of any supercell built from it afterwards.
Stored against a snapshot of the cell it was authored on and resolved onto a later
structure through the integer supercell matrix relating the two lattices. Going
through an *integer* matrix is what makes the resolution survive a change of
lattice constant -- the metric cancels -- and matching each atom to the *nearest*
reference site of its element, rather than to an exact position, is what makes it
survive a change of tilt angle, where the copies of a site are no longer at
identical fractional coordinates.

**Atom scope** -- an edit made on a *supercell*. Breaking the periodicity is the
entire reason to be working in a supercell, so an edit there is a statement about
one atom and is stored by index. An index stops meaning what it meant the moment
the atom count changes, which is exactly when :meth:`OxidationOverrides.drop_atom_scope`
is called.

Atom scope wins where both cover an atom: it is the more specific statement, and it
is the one made later.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Sequence, Tuple

import numpy as np

from quick_mag.magnetic_moments import OxidationStateAssignment, get_magnetic_moment

# How far the supercell matrix relating two lattices may drift from integer before
# they are taken not to be related by one, as a fraction of the entry itself.
#
# Relative, not absolute, because the drift scales with the entry: the reference
# cell is stored with the lattice constant it had, and a later edit to that constant
# moves an N-cell repeat by N times the fractional change. An absolute bound tight
# enough to be meaningful at N = 1 rejects a 10% edit on a 3x3x3, which is how a
# perfectly good set of edits would silently stop applying halfway through a session.
SUPERCELL_MATRIX_TOLERANCE = 0.2

__all__ = [
    "OxidationOverrides",
    "SUPERCELL_MATRIX_TOLERANCE",
    "assignment_with_overrides",
    "distributions_from_states",
    "reference_site_of_atoms",
    "resolve_overrides",
    "supercell_matrix",
]


@dataclass
class OxidationOverrides:
    """The hand-set oxidation states of one session, in both scopes.

    ``reference_*`` is a snapshot of the unit cell the cell-scope edits were
    authored on: its lattice (rows are the cell vectors, matching
    ``ChemicalStructure.lattice``), one element symbol per site, and the
    fractional coordinates wrapped into ``[0, 1)``. It is captured on the first
    cell-scope edit and re-captured whenever an edit arrives on a cell that is not
    the one already stored.
    """

    reference_lattice: np.ndarray | None = None
    reference_labels: Tuple[str, ...] = ()
    reference_fractions: np.ndarray | None = None
    # Reference-cell site index -> charge. On a unit cell the site index and the
    # atom index are the same thing, which is why an edit can be recorded without
    # resolving anything.
    cell_states: Dict[int, int] = field(default_factory=dict)
    # Analysis-structure atom index -> charge.
    atom_states: Dict[int, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.cell_states) + len(self.atom_states)

    def is_empty(self) -> bool:
        return not self.cell_states and not self.atom_states

    def clear(self) -> None:
        """Forget every edit, and the cell they were authored against."""
        self.cell_states.clear()
        self.atom_states.clear()
        self.reference_lattice = None
        self.reference_labels = ()
        self.reference_fractions = None

    def drop_atom_scope(self) -> None:
        """Forget the per-atom edits, keeping the propagating ones.

        Called when the atom count changes -- a supercell resize, a tiling, a
        vacancy added or removed -- because that is when an atom index stops
        naming the atom it was recorded against. Cell-scope edits are keyed
        geometrically and survive all of it.
        """
        self.atom_states.clear()

    # -- authoring ---------------------------------------------------------

    def capture_reference(self, structure) -> None:
        """Snapshot ``structure`` as the cell that cell-scope edits name sites of."""
        self.reference_lattice = np.array(structure.lattice, dtype=np.float64, copy=True)
        self.reference_labels = tuple(structure.element_symbols())
        self.reference_fractions = np.mod(
            np.asarray(structure.fractional_coords, dtype=np.float64), 1.0
        )

    def reference_matches(self, structure) -> bool:
        """Whether ``structure`` is the cell the stored edits were authored on.

        Composition and site count have to agree exactly; the lattice only has to
        be the same cell rather than the same *shape*, so straining it or nudging a
        tilt does not throw the edits away.
        """
        if self.reference_lattice is None or self.reference_fractions is None:
            return False
        if self.reference_labels != tuple(structure.element_symbols()):
            return False
        matrix = supercell_matrix(structure.lattice, self.reference_lattice)
        return matrix is not None and np.array_equal(matrix, np.eye(3, dtype=int))

    def set(self, structure, atom_index: int, charge: int, *, propagate: bool) -> None:
        """Record ``atom_index`` of ``structure`` as ``charge``.

        ``propagate`` is the caller's answer to "is this a unit cell?"; it is not
        re-derived here, because the answer depends on builder provenance this
        module deliberately knows nothing about.
        """
        atom_index = int(atom_index)
        charge = int(charge)
        if not propagate:
            self.atom_states[atom_index] = charge
            return
        if not self.reference_matches(structure):
            # Edits authored against a different cell cannot be resolved onto this
            # one, and silently keeping them would apply another structure's
            # chemistry here.
            self.cell_states.clear()
            self.capture_reference(structure)
        self.cell_states[atom_index] = charge
        # The propagating statement supersedes any one-atom statement about the
        # same atom; leaving the atom-scope entry would make the edit look like it
        # had not taken.
        self.atom_states.pop(atom_index, None)

    def revert(self, structure, atom_index: int, *, propagate: bool) -> None:
        """Drop the edit covering ``atom_index``, returning it to the model.

        On a supercell only the one-atom edit goes: a propagating edit still
        covering the atom is a statement about the whole cell, and dropping that
        from inside a supercell would silently change every other copy too.
        """
        atom_index = int(atom_index)
        self.atom_states.pop(atom_index, None)
        if propagate:
            self.cell_states.pop(atom_index, None)


def supercell_matrix(lattice, reference_lattice) -> np.ndarray | None:
    """Integer ``M`` with ``lattice == M @ reference_lattice``, or None.

    None when the two are not related by a supercell at all -- a different
    material, or a cell that has been sheared rather than repeated.
    """
    try:
        raw = np.asarray(lattice, dtype=np.float64) @ np.linalg.inv(
            np.asarray(reference_lattice, dtype=np.float64)
        )
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(raw)):
        return None
    rounded = np.round(raw)
    allowed = SUPERCELL_MATRIX_TOLERANCE * np.maximum(np.abs(rounded), 1.0)
    if np.any(np.abs(raw - rounded) > allowed):
        return None
    if abs(float(np.linalg.det(rounded))) < 0.5:
        return None
    return rounded.astype(int)


def reference_site_of_atoms(
    structure,
    overrides: OxidationOverrides,
    repeats: Sequence[int] | None = None,
) -> np.ndarray | None:
    """Reference-cell site each atom of ``structure`` is a copy of; -1 where none.

    ``repeats`` is how many primitive cells ``structure`` spans per axis, when the
    caller knows -- off the builder provenance, which says so exactly. It is taken
    over inferring the same thing from the two lattices, because inference has to
    tell a repeat apart from a change of lattice constant and provenance does not.
    Without it the lattices are all there is, and a cell that is not a supercell of
    the reference resolves to nothing rather than to something wrong.

    Matching is nearest-same-element in the reference cell, under the minimum-image
    convention. Nearest rather than exact because the copies of a site are only at
    identical fractional coordinates in an untilted cell: a tilt pattern with a
    period of more than one cell moves them, by much less than the spacing between
    distinct sites of the same element.
    """
    if overrides.reference_fractions is None or overrides.reference_lattice is None:
        return None
    matrix = None
    if repeats is not None:
        counts = [max(1, int(count)) for count in repeats]
        # Only when the reference is a cell of *this* material. A repeat count says
        # how the structure is built, not what it is built from.
        if set(overrides.reference_labels) >= set(structure.element_symbols()):
            matrix = np.diag(counts)
    if matrix is None:
        matrix = supercell_matrix(structure.lattice, overrides.reference_lattice)
    if matrix is None:
        return None
    reference_lattice = np.asarray(overrides.reference_lattice, dtype=np.float64)
    # cart = frac @ L and L = M @ L_ref, so frac_ref = frac @ M -- integer
    # arithmetic on the fractional coordinates, with the metric nowhere in it.
    fractions = np.mod(
        np.asarray(structure.fractional_coords, dtype=np.float64) @ matrix, 1.0
    )
    reference_fractions = np.mod(
        np.asarray(overrides.reference_fractions, dtype=np.float64), 1.0
    )

    sites_by_element: Dict[str, List[int]] = {}
    for site, element in enumerate(overrides.reference_labels):
        sites_by_element.setdefault(str(element), []).append(site)

    symbols = structure.element_symbols()
    resolved = np.full(len(symbols), -1, dtype=int)
    for atom, element in enumerate(symbols):
        candidates = sites_by_element.get(str(element))
        if not candidates:
            continue
        if len(candidates) == 1:
            resolved[atom] = candidates[0]
            continue
        delta = reference_fractions[candidates] - fractions[atom]
        delta -= np.round(delta)
        distances = np.linalg.norm(delta @ reference_lattice, axis=1)
        resolved[atom] = candidates[int(np.argmin(distances))]
    return resolved


def resolve_overrides(
    structure,
    overrides: OxidationOverrides,
    repeats: Sequence[int] | None = None,
) -> Dict[int, int]:
    """Atom index -> charge for ``structure``, with both scopes resolved onto it."""
    resolved: Dict[int, int] = {}
    if overrides.cell_states:
        sites = reference_site_of_atoms(structure, overrides, repeats)
        if sites is not None:
            for atom, site in enumerate(sites):
                charge = overrides.cell_states.get(int(site))
                if charge is not None:
                    resolved[atom] = int(charge)
    for atom, charge in overrides.atom_states.items():
        if 0 <= int(atom) < structure.atom_count:
            resolved[int(atom)] = int(charge)
    return resolved


def distributions_from_states(
    symbols: Sequence[str], states: Sequence[int]
) -> Dict[str, Dict[int, int]]:
    """``{element: {oxidation state: count}}`` counted straight off the sites."""
    distributions: Dict[str, Dict[int, int]] = {}
    for symbol, charge in zip(symbols, states):
        per_element = distributions.setdefault(str(symbol), {})
        per_element[int(charge)] = per_element.get(int(charge), 0) + 1
    return distributions


def assignment_with_overrides(
    assignment: OxidationStateAssignment,
    structure,
    overrides: OxidationOverrides,
    repeats: Sequence[int] | None = None,
) -> OxidationStateAssignment:
    """``assignment`` with the hand-set states written into it.

    Returned unchanged -- the same object, so callers can compare identities --
    when nothing is overridden or nothing would change. Everything downstream of
    the assignment (render radii, the ion descriptors the exchange matrix is built
    from, the hover tooltip, export) reads it through this, so an edit reaches all
    of them without any of them knowing overrides exist.

    ``total_energy`` is deliberately left alone. It is the model's energy for the
    distribution the model chose, and an edited assignment has no such energy; the
    UI stops showing it once there are edits rather than showing a number that
    describes something else.
    """
    resolved = resolve_overrides(structure, overrides, repeats)
    states = np.asarray(assignment.site_oxidation_states, dtype=int)
    if not resolved or len(states) != structure.atom_count:
        return assignment

    symbols = structure.element_symbols()
    edited_states = states.copy()
    moments = np.asarray(assignment.magnetic_moments, dtype=np.float64).copy()
    changed = False
    for atom, charge in resolved.items():
        if not 0 <= atom < len(edited_states) or int(edited_states[atom]) == charge:
            continue
        edited_states[atom] = charge
        if atom < len(moments):
            try:
                moments[atom] = abs(float(get_magnetic_moment(symbols[atom], charge)))
            except Exception:
                # An ion the electron-configuration tables cannot build carries no
                # formal moment. The state is still the user's to set -- a site
                # with no moment is a perfectly good thing to ask for.
                moments[atom] = 0.0
        changed = True
    if not changed:
        return assignment
    return replace(
        assignment,
        site_oxidation_states=edited_states,
        magnetic_moments=moments,
        distributions=distributions_from_states(symbols, edited_states),
    )
