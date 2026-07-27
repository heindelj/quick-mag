# Perovskite Spin-Structure Classifier

`quick_mag/classify_spin_structure.py` classifies common collinear perovskite B-site
spin patterns from an existing B-site grid.  It expects the caller to provide
the perovskite site roles through `PerovskiteSiteIndexing`, preferably from
`site_indexing_from_perovskite_build(build)` for generated structures or from
the B-site matching workflow for imported structures.

## Descriptors

- `sigma_a`, `sigma_b`, `sigma_c`: average normalized spin dot product to the
  two periodic B-site neighbors along each pseudocubic axis.
- `axis_signs`: snapped signs of each sigma, where `+1` is ferromagnetic,
  `-1` is antiferromagnetic, `0` is mixed, and `None` is unresolved.
- `kappa`: `1 - mean(dot_ij ** 2)` over resolved nearest-neighbor bonds.  This
  is near zero for collinear states and grows for non-collinear states.
- `confidence`: average closeness of resolved sigmas to their snapped targets.
- `notes`: audit flags such as single-layer periodic axes or inactive B-site
  moments.

## Rule Table

| Label | Axis signs |
| --- | --- |
| `F` | `+ + +` |
| `G` | `- - -` |
| `A` | two `+`, one `-` |
| `C` | one `+`, two `-` |
| `E` | in-plane `a-b` plaquette has a `++--` rotation and `c` is single-layer constrained or ferromagnetically stacked |
| `CE` | `E` plus mixed oxidation-state variance when `site_oxidation_states` are supplied |
| `noncollinear` | no canonical match and large `kappa` |
| `canted` | collinear-like but intermediate axis sigmas |
| `unknown` | insufficient or ambiguous data |

The minimal E-type synthetic cell is a `2x2x1` B-site grid.  Because the
single-layer `c` axis is periodic, the classifier records a note that the axis
is constrained by the cell choice.
