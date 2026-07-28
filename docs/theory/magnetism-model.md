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
electronegativity-gap term provides a charge-transfer stabilization.
Charge-balanced state enumeration is implemented in `quick_mag.oxidation_state_enumeration`.

## 2. Magnetic moments

`quick_mag.magnetic_moments` and `quick_mag.ion_descriptors` expand a chosen oxidation
distribution onto concrete per-site assignments and assign high-spin $d$ orbital occupancies for each magnetic site to be used by the exchange model.

### 2.1 Microstates and the occupancy vector

The exchange model needs one number per $d$ orbital: $\bar m_{a}$, the net unpaired spin
on orbital $a$. Hund filling fixes how many orbitals in each shell are empty, half-filled,
or full ($E$/$H$/$F$ in `ion_descriptors.shell_ehf`), but *which* named orbital takes which
role is not fixed. Every distinct assignment is a degenerate microstate
(`enumerate_shell_microstates`). Mn³⁺ ($t_{2g}^3 e_g^1$) has two microstates: the lone $e_g$ electron
can sit in $d_{z^2}$ or in $d_{x^2-y^2}$, at identical energy.

$\bar m$ is the average over those microstates, which for degenerate microstates is just
the mean over each equivalent $t_{2g}$ orbital and each equivalent $e_g$
orbital (`polarization_model.occupancy_vector`). Orbital order is written
$(d_{xy}, d_{xz}, d_{yz} \mid d_{z^2}, d_{x^2-y^2})$:

| ion | filling | microstates (electrons per orbital) | $\bar m$ |
|---|---|---|---|
| Cr³⁺ $d^3$ | $t_{2g}^3 e_g^0$ | `1 1 1 │ 0 0` | `1 1 1 │ 0 0` |
| Mn³⁺ $d^4$ | $t_{2g}^3 e_g^1$ | `1 1 1 │ 1 0`<br>`1 1 1 │ 0 1` | `1 1 1 │ ½ ½` |
| Fe³⁺ $d^5$ | $t_{2g}^3 e_g^2$ | `1 1 1 │ 1 1` | `1 1 1 │ 1 1` |
| Fe²⁺ $d^6$ | $t_{2g}^4 e_g^2$ | `2 1 1 │ 1 1`<br>`1 2 1 │ 1 1`<br>`1 1 2 │ 1 1` | `⅔ ⅔ ⅔ │ 1 1` |
| Ni²⁺ $d^8$ | $t_{2g}^6 e_g^2$ | `2 2 2 │ 1 1` | `0 0 0 │ 1 1` |
| Cu²⁺ $d^9$ | $t_{2g}^6 e_g^3$ | `2 2 2 │ 2 1`<br>`2 2 2 │ 1 2` | `0 0 0 │ ½ ½` |

Only singly occupied orbitals contribute since a doubly occupied and empty orbitals carry no net spin.

**What approximation is this?** $\bar m$ is spherically symmetric within each shell, so the
base exchange terms of §3 carry no orbital-order information at all: Mn³⁺ enters them as
half an electron in each $e_g$ orbital rather than one electron in a specific lobe. The Kugel–Khomskii term in §3.2 attempts to recover the directionality of orbital interactions by reading the occupied $e_g$ lobe off the structure, taking the longest metal–ligand bond as
the orbital director (`eg_orbital_director`). A Jahn-Teller distorted geometry therefore
carries the orbital order that $\bar m$ discards, which is why relaxing a structure before
solving it can change the predicted ordering.

Weighting the microstates non-uniformly is a physically-correct extension which we may explore in the future. This requires having a model of the magnitude of the splitting of degenerate orbitals which can quickly become more expensive than the screening model we aim to provide.

## 3. Exchange couplings

`quick_mag.polarization_model` builds the coupling matrix $J$ from an
**exchange-polarization superexchange** picture. Each metal $i$ polarizes the $p$ channels
of a bridging ligand $L$ with intensity

$$
\mu_{i\to L,p} = \kappa_i \, f_{iL}(r) \sum_a \bar m_{i,a}\, W_{a,p},
$$

where $W$ are squared Slater–Koster $pd$ channel weights (`quick_mag.sk_table`) in a
bridge-adapted orthonormal $p$ frame, $\bar m_{i,a}$ is the microstate-averaged net
unpaired spin per $d$ orbital of [§2.1](#21-microstates-and-the-occupancy-vector), and $f_{iL}(r) = \exp\!\big(-\alpha_{iL}(r - R_{0,iL})\big)$ is a
metal–ligand damping of the interaction with $\alpha_{iL} = \sqrt{\alpha_i \alpha_L}$ and $R_{0,iL}$ the sum
of Shannon crystal radii.

### 3.1 Slater–Koster $pd$ integrals

`quick_mag.sk_table` writes the Slater-Koster angular integrals for interacting $p$ and $d$ orbitals in tensor form. Each real $d$ orbital $a$ is represented by its
unit-normalized symmetric traceless quadrupole tensor $Q_a$ (so $\operatorname{Tr} Q_a = 0$
and $\operatorname{Tr} Q_a^2 = 1$):

$$
Q_{xy} = \tfrac{1}{\sqrt 2}\big(\hat e_x \hat e_y^{\mathsf T} + \hat e_y \hat e_x^{\mathsf T}\big),
\qquad
Q_{z^2} = \tfrac{1}{\sqrt 6}\operatorname{diag}(-1,-1,2),
\qquad
Q_{x^2-y^2} = \tfrac{1}{\sqrt 2}\operatorname{diag}(1,-1,0),
$$

with $Q_{xz}, Q_{yz}$ following the $Q_{xy}$ pattern. For a metal–ligand bond unit vector
$u$ (ligand $\to$ metal), the signed $pd$ amplitude of orbital $a$ splits into
a $\sigma$ part along $u$ and a $\pi$ part perpendicular to it:

$$
A_\sigma(a, u) = \sqrt{\tfrac{3}{2}}\; u^{\mathsf T} Q_a\, u
\qquad\text{(scalar)},
$$

$$
t_\pi(a, u) = \sqrt{2}\,\big[\, Q_a u - (u^{\mathsf T} Q_a u)\, u \,\big]
\qquad\text{(vector, } t_\pi \perp u\text{)}.
$$

Writing $u = (l, m, n)$ in direction cosines, $A_\sigma$ expands to the $(pd\sigma)$
angular factors,

$$
\begin{aligned}
A_\sigma(d_{xy}) &= \sqrt 3\, l m, &
A_\sigma(d_{xz}) &= \sqrt 3\, l n, &
A_\sigma(d_{yz}) &= \sqrt 3\, m n, \\[2pt]
A_\sigma(d_{z^2}) &= n^2 - \tfrac{1}{2}\big(l^2 + m^2\big), &
A_\sigma(d_{x^2-y^2}) &= \tfrac{\sqrt 3}{2}\big(l^2 - m^2\big), &&
\end{aligned}
$$

The normalization is fixed so
that a pure $\sigma$ bond gives unit amplitude: $A_\sigma(d_{z^2}, \hat z) = 1$ and
$|t_\pi(d_{xz}, \hat z)| = 1$.

Two sum rules follow from $\{Q_a\}$ being an orthonormal basis of the symmetric traceless
tensors, and hold for every bond direction:

$$
\sum_a A_\sigma(a,u)^2 = 1,
\qquad
\sum_a \big|t_\pi(a,u)\big|^2 = 2,
$$

i.e. one $\sigma$ channel and two $\pi$ channels per bond, distributed over the $d$ shell.

The polarization model consumes squared weights, because the induced ligand
spin polarization scales as the hopping squared and is therefore blind to orbital sign.
Projecting onto the orthonormal ligand $p$ frame $\{e_1, e_2, e_3\}$ and squaring per
channel ($\sigma$/$\pi$):

$$
W_\sigma[a,p] = \big(A_\sigma(a,u)\,(e_p \cdot u)\big)^2,
\qquad
W_\pi[a,p] = \big(e_p \cdot t_\pi(a,u)\big)^2 .
$$

Summed over channels these give $\sum_p (W_\sigma + W_\pi)[a,p] = A_\sigma^2 + |t_\pi|^2$,
independent of the frame. Each bridge end contracts them with its occupancies to form
intensity vectors $B_{\sigma} = \sum_a \bar m_a W_\sigma[a,\cdot]$ and
$B_{\pi} = \sum_a \bar m_a W_\pi[a,\cdot]$, which combine with a global $\pi/\sigma$
amplitude ratio $\gamma_\pi$ into the polarization actually used:

$$
\mu_i = \kappa_i\, f_{iL}(r)\,\big(B_{\sigma,i} + \gamma_\pi^2\, B_{\pi,i}\big).
$$

Each site's $d$ orbitals are evaluated in its own octahedral frame
(`local_octahedral_frame`), so tilted or rotated octahedra retain their $e_g$/$t_{2g}$
character.

**Where the bond angle enters.** The bridge frame is built with $e_1 = u_i$, so end $i$
places all of its $\sigma$ weight on channel $e_1$. End $j$, at M–L–M angle $\theta$, has
$u_j \cdot e_1 = \cos\theta$ and $u_j \cdot e_2 = \sin\theta$, so its $\sigma$ weight splits
as $\cos^2\theta$ on $e_1$ and $\sin^2\theta$ on $e_2$. The two ends therefore share the
same $\sigma$ channel at $180^\circ$ and occupy orthogonal ones at $90^\circ$ — the
geometric origin of the Goodenough–Kanamori–Anderson rules in this model.

For a collinear configuration the ligand's net per-channel polarization is
$P_p = \sigma_i \mu_{i,p} + \sigma_j \mu_{j,p}$ with $\sigma = \pm 1$. A Pauli cost
$E_A = w_L \sum_p P_p^2$ penalizes shared channels (AFM) and a ligand Hund term
$E_B = -J_H^L \sum_{p\neq p'} P_p P_{p'}$ stabilizes orthogonal-channel polarization (FM). The configuration-dependent part per bridge is

$$
J_{\text{bridge}} = 2\,(w_L + J_H^L)\,(\mu_i \cdot \mu_j)
\;-\; 2\,J_H^L \Big(\textstyle\sum_p \mu_{i,p}\Big)\Big(\textstyle\sum_p \mu_{j,p}\Big).
$$

**Sign convention:** $J > 0$ is antiferromagnetic, and the model energy of a collinear
configuration is

$$
E = \tfrac{1}{2}\, \sigma^{\mathsf T} J \, \sigma, \qquad \sigma \in \{+1, 0, -1\}.
$$

Spin magnitude lives *inside* $\mu$ (via $\sum_a \bar m_a$), so the solver runs on unit $\pm 1$ spins.

### 3.2 What each term favors

- **Term A, the Pauli cost $2(w_L + J_H^L)(\mu_i \cdot \mu_j)$ — AFM, geometry-sensitive.**
  It is a $\pi/\sigma$ channel overlap. Two metals that polarize the same ligand $p$ channel pay a Pauli penalty, relieved by aligning antiparallel. All of the angular dependence of the model resides in this term through the $\cos^2\theta / \sin^2\theta$ channel split.
- **Term B, the Hund (on-site exchange) term $-2J_H^L(\sum_p \mu_{i,p})(\sum_p \mu_{j,p})$ — FM,
  geometry-independent.** It depends only on the total intensity each end of the bridge delivers, which
  by the sum rules is independent of the bridge frame. It therefore acts as a fixed FM
  offset that Term A must overcome.

The competition reproduces the GKA rules. Taking $w_L = 1$, $J_H^L = 0.1$, and unit
$\kappa f$, the coupling of a single bridge as a function of M–L–M angle is

| occupancies | $180^\circ$ | $150^\circ$ | $120^\circ$ | $90^\circ$ |
|---|---|---|---|---|
| $e_g$–$e_g$ (e.g. $d^8$–$d^8$) | $+2.00$ AFM | $+1.45$ AFM | $+0.35$ AFM | $-0.20$ **FM** |
| $t_{2g}$–$t_{2g}$ | $+3.60$ AFM | $+3.05$ AFM | $+1.95$ AFM | $+1.40$ AFM |
| $d^5$–$d^5$ (half filled) | $+4.80$ AFM | $+4.80$ AFM | $+4.80$ AFM | $+4.80$ AFM |

For $\sigma$-bonding $e_g$ electrons the overlap follows $\mu_i \cdot \mu_j = \cos^2\theta$
exactly, so Term A collapses at $90^\circ$ and the Hund offset makes the bridge
ferromagnetic. The $t_{2g}$ case weakens with angle but keeps finite $\pi$–$\pi$
overlap, so it stays AFM throughout.

For a half-filled shell every $\bar m_a$ is equal, and the sum rules then make the
channel intensity vector isotropic, $\mu \propto (1,1,1)$. Both terms become
angle-independent, so a $d^5$–$d^5$ bridge is predicted AFM with no geometric
dependence at all. Angular physics for such ions has to come from the bond-length
damping $f_{iL}(r)$ or from the two correction terms below. This is a deficiency of the model which I aim to correct in the future, likely by including a direct exchange interaction between metal sites or a simple description of orbital hybridization.

Two further FM terms are added for specific situations (`bridge_J`):

$$
J_{\text{DE}} = -\,\tau_i \tau_j\, f_{iL} f_{jL} \cos^2\theta,
\qquad
J_{\text{eg}} = -\,t^{e_g}_i t^{e_g}_j \big(g_i + g_j - 2 g_i g_j\big).
$$

- **Double exchange** ($J_{\text{DE}}$) applies only on bridges whose element pair is forced
  into mixed valence by charge balance. A real carrier hops between the two sites, which
  requires parallel spins (Anderson–Hasegawa); the $\cos^2\theta$ factor makes the transfer
  maximal at $180^\circ$ and zero at $90^\circ$.
- **Kugel–Khomskii orbital order** ($J_{\text{eg}}$) applies to orbitally degenerate
  $e_g^1/e_g^3$ ions such as Mn$^{3+}$. With $g = (\text{director} \cdot u)^2$ measuring how
  strongly a site's occupied $e_g$ lobe points along the bond, the factor
  $g_i + g_j - 2g_ig_j$ peaks when exactly *one* of the two lobes is on-axis. AFM
  orbital order, half-filled $\sigma$ on one side of a bridge and empty on the other. This is responsible for the
  in-plane FM ordering of LaMnO$_3$.

Both are combined by a product rule, $x_i x_j$, so same-element pairs see $x^2$ and
cross-element pairs get the geometric combination for free.

## 4. Solving for spin configurations

`quick_mag.spin_solver` searches for low-energy collinear configurations of the Ising
form above. For small magnetic sublattices (≤ `--exact-max-sites`, default 16) it
enumerates exactly; otherwise it runs a multi-restart optimizer (`--n-trials`,
`--n-steps`). The canonical G/C/F/A reference orderings (`quick_mag.reference_configs`) are
always scored on the same coupling matrix for ease of comparison.

## 5. Classifying orderings

`quick_mag.classify_spin_structure` labels a configuration against the canonical
perovskite B-site patterns (F / A / C / G / E). See the
[Spin-Structure Classifier](classify_spin_structure.md) page for the descriptor
definitions and the classification rules.
