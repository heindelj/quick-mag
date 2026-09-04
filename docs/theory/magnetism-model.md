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
half an electron in each $e_g$ orbital rather than one electron in a specific lobe. For the
$e_g^1$ ions where that loss actually matters, §3.5 puts the information back: the
occupied $e_g$ orbital is selected from the ligand geometry by a crystal-field integral
and used *coherently*, in place of the shell average, and §3.4 describes a
ferromagnetic exchange channel that only exists once the orbital is resolved. A
Jahn-Teller distorted geometry therefore carries the orbital order that $\bar m$ discards,
which is why relaxing a structure before solving it can change the predicted ordering.

Weighting the microstates non-uniformly everywhere is physically-correct and may be explored in the future. Unfortunately, this requires having a model of the magnitude of the splitting of degenerate orbitals which can quickly become more expensive than the screening model we aim to provide.

## 3. Exchange couplings

`quick_mag.polarization_model` builds the coupling matrix $J$ from an
**exchange-polarization superexchange** picture. Two metal sites are coupled through every
ligand that bridges them, and those bridge couplings are additive, so the matrix element
the solver sees is a sum over the bridges $\mathcal{B}(i,j)$ connecting the pair:

$$
J_{ij} = \sum_{\text{bridge} \, \in \, \mathcal{B}(i,j)} J_{\text{bridge}} .
$$

A pair is often bridged more than once — the shared ligands of an edge- or face-sharing
pair, or symmetry-equivalent corner ligands. Each individual bridge carries three
contributions,

$$
J_{\text{bridge}} = \underbrace{2\,(w_L + J_H^L)\,(\mu_i \cdot \mu_j)
\;-\; 2\,J_H^L \Big(\textstyle\sum_p \mu_{i,p}\Big)\Big(\textstyle\sum_p \mu_{j,p}\Big)}_{J_{\text{SE}}}
\;\underbrace{-\; \tau_i \tau_j\, f_{iL} f_{jL} \cos^2\theta}_{J_{\text{DE}}}
\;\underbrace{-\; j^{\text{fm}}_i j^{\text{fm}}_j \big( \mu^{\text{occ}}_i \cdot \mu^{\text{emp}}_j + \mu^{\text{emp}}_i \cdot \mu^{\text{occ}}_j \big)}_{J_{\text{OE}}},
$$

where $\mu_i$ is the spin polarization metal $i$ induces in the ligand's $p$ channels
(defined just below) and $\theta$ is the M–L–M angle:

- $J_{\text{SE}}$ is the **superexchange** part, present on every bridge. It is the
  competition between a Pauli cost on shared ligand channels (AFM, and the only
  angle-dependent piece) and a ligand Hund term (FM). This is where the
  Goodenough–Kanamori–Anderson rules come from in the model. Further information is in [§3.1](#31-the-superexchange-term) and
  [§3.2](#32-what-each-term-favors).
- $J_{\text{DE}}$ is **double exchange** (FM), active only on bridges whose element pair is
  forced into mixed valence by charge balance ([§3.3](#33-double-exchange)).
- $J_{\text{OE}}$ is the **occupied-to-empty** Kugel–Khomskii channel (FM), active only when
  both ends are $e_g^1$ sites whose occupied orbital has been resolved from the crystal
  field ([§3.4](#34-the-occupied-to-empty-fm-channel), [§3.5](#35-crystal-field-orbital-order)).

Both $J_{\text{DE}}$ and $J_{\text{OE}}$ vanish whenever either end's per-element factor
($\tau$ or $j^{\text{fm}}$) is zero, so an unparameterized element simply drops the channel.
Only $J_{\text{SE}}$ can produce AFM orderings while the other two terms favor FM configurations.

**Sign convention:** $J > 0$ is antiferromagnetic, and the model energy of a collinear
configuration is

$$
E = \tfrac{1}{2}\, \sigma^{\mathsf T} J \, \sigma, \qquad \sigma \in \{+1, 0, -1\}.
$$

Spin magnitude lives *inside* $\mu$ (via $\sum_a \bar m_a$), so the solver runs on unit
$\pm 1$ spins.

The rest of this section builds up the pieces. Each metal $i$ polarizes the $p$ channels
of a bridging ligand $L$ with intensity

$$
\mu_{i\to L,p} = \kappa_i \, f_{iL}(r) \sum_a \bar m_{i,a}\, W_{a,p},
$$

where $W$ are squared Slater–Koster $pd$ channel weights (`quick_mag.sk_table`) in a
bridge-adapted orthonormal $p$ frame, $\bar m_{i,a}$ is the microstate-averaged net
unpaired spin per $d$ orbital of [§2.1](#21-microstates-and-the-occupancy-vector), and $f_{iL}(r) = \exp\!\big(-\alpha_{iL}(r - R_{0,iL})\big)$ is a
metal–ligand damping of the interaction with $\alpha_{iL} = \sqrt{\alpha_i \alpha_L}$ and $R_{0,iL}$ the sum
of Shannon crystal radii.

### 3.1 The superexchange term

$J_{\text{SE}}$ is the term every bridge carries, and it follows from the ligand
polarizations $\mu_i$ and $\mu_j$ alone. The polarization model consumes squared
Slater–Koster weights, because the induced ligand
spin polarization scales as the hopping squared and is therefore blind to orbital sign.
Taking the signed $\sigma$ and $\pi$ amplitudes $A_\sigma(a, u)$ and $t_\pi(a, u)$ of a
metal–ligand bond from the Slater–Koster table of
[§3.6](#36-slaterkoster-pd-integrals), projecting onto the orthonormal ligand $p$ frame
$\{e_1, e_2, e_3\}$ and squaring per channel ($\sigma$/$\pi$):

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
same $\sigma$ channel at $180^\circ$ and occupy orthogonal ones at $90^\circ$. That behavior is how the Goodenough–Kanamori–Anderson rules arise in this model.

For a collinear configuration the ligand's net per-channel polarization is
$P_p = \sigma_i \mu_{i,p} + \sigma_j \mu_{j,p}$ with $\sigma = \pm 1$. A Pauli cost
$E_{\text{Pauli}} = w_L \sum_p P_p^2$ penalizes shared channels (AFM) and a ligand Hund term
$E_{\text{Hund}} = -J_H^L \sum_{p\neq p'} P_p P_{p'}$ stabilizes orthogonal-channel polarization (FM). Without the ligand Hund term, $90^\circ$ bridges would not favor weak FM alignment in violation of the GKA rules. The complete superexchange contribution per bridge is,

$$
J_{\text{SE}} = 2\,(w_L + J_H^L)\,(\mu_i \cdot \mu_j)
\;-\; 2\,J_H^L \Big(\textstyle\sum_p \mu_{i,p}\Big)\Big(\textstyle\sum_p \mu_{j,p}\Big).
$$

### 3.2 What each term favors

$J_{\text{SE}}$ is itself a competition between two pieces:

- **The Pauli cost $2(w_L + J_H^L)(\mu_i \cdot \mu_j)$ — AFM, geometry-sensitive.**
  It is a $\pi/\sigma$ channel overlap. Two metals that polarize the same ligand $p$ channel pay a Pauli penalty, relieved by aligning antiparallel. All of the angular dependence of the model resides in this term through the $\cos^2\theta / \sin^2\theta$ channel split.
- **The ligand Hund (on-site exchange) term $-2J_H^L(\sum_p \mu_{i,p})(\sum_p \mu_{j,p})$ — FM,
  geometry-independent.** It depends only on the total intensity each end of the bridge delivers, which
  by the sum rules is independent of the bridge frame. It therefore acts as a fixed FM
  offset that the Pauli cost must overcome.

The competition reproduces the GKA rules. Taking $w_L = 1$, $J_H^L = 0.1$, and unit
$\kappa f$, the coupling of a single bridge as a function of M–L–M angle is

| occupancies | $180^\circ$ | $150^\circ$ | $120^\circ$ | $90^\circ$ |
|---|---|---|---|---|
| $e_g$–$e_g$ (e.g. $d^8$–$d^8$) | $+2.00$ AFM | $+1.45$ AFM | $+0.35$ AFM | $-0.20$ **FM** |
| $t_{2g}$–$t_{2g}$ | $+3.60$ AFM | $+3.05$ AFM | $+1.95$ AFM | $+1.40$ AFM |
| $d^5$–$d^5$ (half filled) | $+4.80$ AFM | $+4.80$ AFM | $+4.80$ AFM | $+4.80$ AFM |

For $\sigma$-bonding $e_g$ electrons the overlap follows $\mu_i \cdot \mu_j = \cos^2\theta$
exactly, so the Pauli cost collapses at $90^\circ$ and the Hund offset makes the bridge
ferromagnetic. The $t_{2g}$ case weakens with angle but keeps finite $\pi$–$\pi$
overlap, so it stays AFM throughout.

For a half-filled shell every $\bar m_a$ is equal, and the sum rules then make the
channel intensity vector isotropic, $\mu \propto (1,1,1)$. Both terms become
angle-independent, so a $d^5$–$d^5$ bridge is predicted AFM with no geometric
dependence at all. Angular physics for such ions has to come from the bond-length
damping $f_{iL}(r)$ or from the two additional terms below.

### 3.3 Double exchange

The first of the two additional FM terms is **double exchange**,

$$
J_{\text{DE}} = -\,\tau_i \tau_j\, f_{iL} f_{jL} \cos^2\theta,
$$

which applies only on bridges whose element pair is forced into mixed valence by charge
balance. A real carrier hops between the two sites, which requires parallel spins. The $\cos^2\theta$ factor makes the transfer maximal at $180^\circ$
and zero at $90^\circ$. Like $J_{\text{OE}}$ below, it takes its per-element factors by a
product rule, $x_i x_j$, so it switches off entirely when either element is
unparameterized.

### 3.4 The occupied-to-empty FM channel

Once the occupied $e_g$ orbital $\psi_{\text{occ}}$ is fixed by the crystal field of
[§3.5](#35-crystal-field-orbital-order), its orthogonal partner
$\psi_{\text{emp}}$ is the empty $e_g$ acceptor. Writing $\mu^{\text{occ}}$ and
$\mu^{\text{emp}}$ for the polarization intensities of those two orbitals, the model adds

$$
J_{\text{OE}} = -\, j^{\text{fm}}_i j^{\text{fm}}_j
\Big( \mu^{\text{occ}}_i \cdot \mu^{\text{emp}}_j
\;+\; \mu^{\text{emp}}_i \cdot \mu^{\text{occ}}_j \Big).
$$

This is the Kugel–Khomskii FM mechanism. Occupied-occupied hopping costs energy unless the spins are antiparallel, which is the Pauli cost in $J_{\text{SE}}$. Hopping
into a neighbour's empty orbital is allowed and lowers energy when the two
sites are FM-aligned (Hund's rule). Resolving the occupied orbital coherently is what allows the model to produce A-type configurations rather than simple ferromagnetism.

### 3.5 Crystal-field orbital order

A shell average cannot describe an ion with one $e_g$ electron in one specific lobe, and
the base terms provide no mechanism that makes a $180^\circ$ bridge ferromagnetic. To see
why, note that at $\theta \approx 180^\circ$ both ends polarize a *single* shared $p$
channel, so $\mu_i \cdot \mu_j \approx (\sum_p \mu_{i,p})(\sum_p \mu_{j,p})$ and

$$
J_{\text{SE}} \;\approx\; 2(w_L + J_H^L)\,(\mu_i \cdot \mu_j) \;-\; 2 J_H^L\,(\mu_i \cdot \mu_j)
\;=\; 2 w_L\,(\mu_i \cdot \mu_j),
$$

i.e. the ligand Hund term cancels exactly and the bridge is purely AFM regardless of
$J_H^L$. That term only produces FM at multi-channel ($\sim 90^\circ$) bridges, so the
in-plane ferromagnetism of an orbitally-ordered corner-sharing perovskite such as LaMnO₃
has to come from somewhere else. It comes from resolving the orbital.

`crystal_field_eg_orbital` determines which $e_g$ orbital is occupied directly from the
ligand cage, using a point-charge crystal field truncated at the quadrupole term. In the site's local octahedral frame the ligand-field tensor is

$$
F = \sum_L \frac{q_L}{R_L^{3}}\,\frac{3\,u_L u_L^{\mathsf T} - I}{2}
$$

and the $k=2$ crystal-field matrix over the five real $d$ orbitals is

$$
H_{ab} = \int \big(\hat n^{\mathsf T} Q_a \hat n\big)\big(\hat n^{\mathsf T} F \hat n\big)
\big(\hat n^{\mathsf T} Q_b \hat n\big)\, \mathrm{d}\Omega,
$$

reusing the same quadrupole tensors $Q_a$ as the Slater–Koster table and can be evaluated analytically.

For an $e_g^1$ site the single electron then occupies one coherent orbital
$\psi = c_{z^2}\,|d_{z^2}\rangle + c_{x^2-y^2}\,|d_{x^2-y^2}\rangle$, whose $\sigma$/$\pi$
amplitudes are formed coherently before squaring (`coherent_eg_intensity`),

$$
B_\sigma[p] = \big(A_\sigma^\psi \,(e_p \cdot u)\big)^2, \qquad
A_\sigma^\psi = \sum_a c_a A_\sigma(a, u),
$$

so the cross-terms that carry orbital directionality survive. This corrects the incoherent shell average described earlier by breaking the degeneracy of $e_g$ orbitals in $e_g^1$ ions and is essential for describing A-type magnetic configurations in this model.

### 3.6 Slater–Koster $pd$ integrals

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
These are the amplitudes that [§3.1](#31-the-superexchange-term) squares into the channel
weights $W_\sigma$ and $W_\pi$, and the same $Q_a$ that
[§3.5](#35-crystal-field-orbital-order) reuses for the crystal field.
