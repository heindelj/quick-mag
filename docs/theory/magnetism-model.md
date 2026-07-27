# Magnetism Model

quick_mag predicts collinear magnetic ground states through a short pipeline that turns a
bare geometry into ranked spin configurations. Each stage lives in its own module and is
reused by both the CLI (`quick_mag.magnetic_cli`) and the UI (`quick_mag.quick_mag_ui`).

## 1. Oxidation states

`quick_mag.oxidation_state_energy` ranks charge-balanced oxidation-state assignments by a
physical, geometry-free energy that mimics filling orbitals by their energies:

$$
E(\text{assignment}) = \sum_{\text{cations}} (\text{ionization ladder})
\;-\; \sum_{\text{anions}} (\text{electron-attachment gain})
\;-\; \gamma \, (\text{charge-transfer stabilization}).
$$

The successive-ionization ladder makes higher charges costly, while the
electronegativity-gap term supplies the charge-transfer stabilization that pays for them.
Lower energy means more stable. Charge-balanced enumeration is reused from
`quick_mag.oxidation_state_enumeration`; only the scoring is added here.

## 2. Magnetic moments

`quick_mag.magnetic_moments` and `quick_mag.ion_descriptors` expand a chosen oxidation
distribution onto concrete per-site assignments and derive, for each transition-metal
site, the shell-averaged net unpaired spin per $d$ orbital that feeds the exchange model.

## 3. Exchange couplings — the polarization model

`quick_mag.polarization_model` builds the coupling matrix $J$ from an
**exchange-polarization superexchange** picture. Each metal $i$ polarizes the $p$ channels
of a bridging ligand $L$ with intensity

$$
\mu_{i\to L,p} = \kappa_i \, f_{iL}(r) \sum_a \bar m_{i,a}\, W_{a,p},
$$

where $W$ are squared Slater–Koster $pd$ channel weights (`quick_mag.sk_table`) in a
bridge-adapted orthonormal $p$ frame, $\bar m_{i,a}$ is the shell-averaged net unpaired
spin per $d$ orbital, and $f_{iL}(r) = \exp\!\big(-\alpha_{iL}(r - R_{0,iL})\big)$ is a
metal–ligand damping with $\alpha_{iL} = \sqrt{\alpha_i \alpha_L}$ and $R_{0,iL}$ the sum
of Shannon crystal radii (not fit).

For a collinear configuration the ligand's net per-channel polarization is
$P_p = \sigma_i \mu_{i,p} + \sigma_j \mu_{j,p}$ with $\sigma = \pm 1$. A Pauli cost
$E_A = w_L \sum_p P_p^2$ penalizes shared channels (AFM) and a ligand Hund term
$E_B = -J_H^L \sum_{p\neq p'} P_p P_{p'}$ rewards orthogonal-channel polarization (FM). The
configuration-dependent part per bridge is

$$
J_{\text{bridge}} = 2\,(w_L + J_H^L)\,(\mu_i \cdot \mu_j)
\;-\; 2\,J_H^L \Big(\textstyle\sum_p \mu_{i,p}\Big)\Big(\textstyle\sum_p \mu_{j,p}\Big).
$$

**Sign convention:** $J > 0$ is antiferromagnetic, and the model energy of a collinear
configuration is

$$
E = \tfrac{1}{2}\, \sigma^{\mathsf T} J \, \sigma, \qquad \sigma \in \{+1, 0, -1\}.
$$

Spin magnitude lives *inside* $\mu$ (via $\sum_a \bar m_a$), so the solver runs on unit
$\pm 1$ spins and must not multiply by nominal moments.

## 4. Solving for spin configurations

`quick_mag.spin_solver_np` searches for low-energy collinear configurations of the Ising
form above. For small magnetic sublattices (≤ `--exact-max-sites`, default 16) it
enumerates exactly; otherwise it runs a multi-restart optimizer (`--n-trials`,
`--n-steps`). The canonical G/C/F/A reference orderings (`quick_mag.reference_configs`) are
always scored on the same coupling matrix so their energies are comparable to the search.

## 5. Classifying orderings

`quick_mag.classify_spin_structure` labels a configuration against the canonical
perovskite B-site patterns (F / A / C / G / E). See the
[Spin-Structure Classifier](classify_spin_structure.md) page for the descriptor
definitions and the classification rules.
