from __future__ import annotations

from dataclasses import dataclass

SHANNON_RADII_SOURCE_URL = "http://abulafia.mt.ic.ac.uk/shannon/radius.php"


@dataclass(frozen=True)
class ShannonIonicRadius:
    ion: str
    charge: int
    coordination: str
    spin_state: str | None
    crystal_radius: float
    ionic_radius: float
    key: str | None = None
    source: str = "shannon"


@dataclass(frozen=True)
class ElementRadii:
    ion: str
    radii: tuple[ShannonIonicRadius, ...]

    def find(
        self,
        *,
        charge: int | None = None,
        coordination: str | None = None,
        spin_state: str | None = None,
    ) -> tuple[ShannonIonicRadius, ...]:
        matches = []
        for record in self.radii:
            if charge is not None and record.charge != charge:
                continue
            if coordination is not None and record.coordination != coordination:
                continue
            if spin_state is not None and record.spin_state != spin_state:
                continue
            matches.append(record)
        return tuple(matches)


_RAW_SHANNON_IONIC_RADII = {
    'Ac': (
        (3, 'VI', None, 1.26, 1.12, 'R'),
    ),
    'Ag': (
        (1, 'II', None, 0.81, 0.67, None),
        (1, 'IV', None, 1.14, 1.0, 'C'),
        (1, 'IVSQ', None, 1.16, 1.02, None),
        (1, 'V', None, 1.23, 1.09, 'C'),
        (1, 'VI', None, 1.29, 1.15, 'C'),
        (1, 'VII', None, 1.36, 1.22, None),
        (1, 'VIII', None, 1.42, 1.28, None),
        (2, 'IVSQ', None, 0.93, 0.79, None),
        (2, 'VI', None, 1.08, 0.94, None),
        (3, 'IVSQ', None, 0.81, 0.67, None),
        (3, 'VI', None, 0.89, 0.75, 'R'),
    ),
    'Al': (
        (3, 'IV', None, 0.53, 0.39, '*'),
        (3, 'V', None, 0.62, 0.48, None),
        (3, 'VI', None, 0.675, 0.535, 'R*'),
    ),
    'Am': (
        (2, 'VII', None, 1.35, 1.21, None),
        (2, 'VIII', None, 1.4, 1.26, None),
        (2, 'IX', None, 1.45, 1.31, None),
        (3, 'VI', None, 1.115, 0.975, 'R'),
        (3, 'VIII', None, 1.23, 1.09, None),
        (4, 'VI', None, 0.99, 0.85, 'R'),
        (4, 'VIII', None, 1.09, 0.95, None),
    ),
    'As': (
        (3, 'VI', None, 0.72, 0.58, 'A'),
        (5, 'IV', None, 0.475, 0.335, 'R*'),
        (5, 'VI', None, 0.6, 0.46, 'C*'),
    ),
    'At': (
        (7, 'VI', None, 0.76, 0.62, 'A'),
    ),
    'Au': (
        (1, 'VI', None, 1.51, 1.37, 'A'),
        (3, 'IVSQ', None, 0.82, 0.68, None),
        (3, 'VI', None, 0.99, 0.85, 'A'),
        (5, 'VI', None, 0.71, 0.57, None),
    ),
    'B': (
        (3, 'III', None, 0.15, 0.01, '*'),
        (3, 'IV', None, 0.25, 0.11, '*'),
        (3, 'VI', None, 0.41, 0.27, 'C'),
    ),
    'Ba': (
        (2, 'VI', None, 1.49, 1.35, None),
        (2, 'VII', None, 1.52, 1.38, 'C'),
        (2, 'VIII', None, 1.56, 1.42, None),
        (2, 'IX', None, 1.61, 1.47, None),
        (2, 'X', None, 1.66, 1.52, None),
        (2, 'XI', None, 1.71, 1.57, None),
        (2, 'XII', None, 1.75, 1.61, 'C'),
    ),
    'Be': (
        (2, 'III', None, 0.3, 0.16, None),
        (2, 'IV', None, 0.41, 0.27, '*'),
        (2, 'VI', None, 0.59, 0.45, 'C'),
    ),
    'Bi': (
        (3, 'V', None, 1.1, 0.96, 'C'),
        (3, 'VI', None, 1.17, 1.03, 'R*'),
        (3, 'VIII', None, 1.31, 1.17, 'R'),
        (5, 'VI', None, 0.9, 0.76, 'E'),
    ),
    'Bk': (
        (3, 'VI', None, 1.1, 0.96, 'R'),
        (4, 'VI', None, 0.97, 0.83, 'R'),
        (4, 'VIII', None, 1.07, 0.93, 'R'),
    ),
    'Br': (
        (-1, 'VI', None, 1.82, 1.96, 'P'),
        (3, 'IVSQ', None, 0.73, 0.59, None),
        (5, 'IIIPY', None, 0.45, 0.31, None),
        (7, 'IV', None, 0.39, 0.25, None),
        (7, 'VI', None, 0.53, 0.39, 'A'),
    ),
    'C': (
        (4, 'III', None, 0.06, -0.08, None),
        (4, 'IV', None, 0.29, 0.15, 'P'),
        (4, 'VI', None, 0.3, 0.16, 'A'),
    ),
    'Ca': (
        (2, 'VI', None, 1.14, 1.0, None),
        (2, 'VII', None, 1.2, 1.06, '*'),
        (2, 'VIII', None, 1.26, 1.12, '*'),
        (2, 'IX', None, 1.32, 1.18, None),
        (2, 'X', None, 1.37, 1.23, 'C'),
        (2, 'XII', None, 1.48, 1.34, 'C'),
    ),
    'Cd': (
        (2, 'IV', None, 0.92, 0.78, None),
        (2, 'V', None, 1.01, 0.87, None),
        (2, 'VI', None, 1.09, 0.95, None),
        (2, 'VII', None, 1.17, 1.03, 'C'),
        (2, 'VIII', None, 1.24, 1.1, 'C'),
        (2, 'XII', None, 1.45, 1.31, None),
    ),
    'Ce': (
        (3, 'VI', None, 1.15, 1.01, 'R'),
        (3, 'VII', None, 1.21, 1.07, 'E'),
        (3, 'VIII', None, 1.283, 1.143, 'R'),
        (3, 'IX', None, 1.336, 1.196, 'R'),
        (3, 'X', None, 1.39, 1.25, None),
        (3, 'XII', None, 1.48, 1.34, 'C'),
        (4, 'VI', None, 1.01, 0.87, 'R'),
        (4, 'VIII', None, 1.11, 0.97, 'R'),
        (4, 'X', None, 1.21, 1.07, None),
        (4, 'XII', None, 1.28, 1.14, None),
    ),
    'Cf': (
        (3, 'VI', None, 1.09, 0.95, 'R'),
        (4, 'VI', None, 0.961, 0.821, 'R'),
        (4, 'VIII', None, 1.06, 0.92, None),
    ),
    'Cl': (
        (-1, 'VI', None, 1.67, 1.81, 'P'),
        (5, 'IIIPY', None, 0.26, 0.12, None),
        (7, 'IV', None, 0.22, 0.08, '*'),
        (7, 'VI', None, 0.41, 0.27, 'A'),
    ),
    'Cm': (
        (3, 'VI', None, 1.11, 0.97, 'R'),
        (4, 'VI', None, 0.99, 0.85, 'R'),
        (4, 'VIII', None, 1.09, 0.95, 'R'),
    ),
    'Co': (
        (2, 'IV', 'High Spin', 0.72, 0.58, None),
        (2, 'V', None, 0.81, 0.67, 'C'),
        (2, 'VI', 'Low Spin', 0.79, 0.65, 'R'),
        (2, 'VI', 'High Spin', 0.885, 0.745, 'R*'),
        (2, 'VIII', None, 1.04, 0.9, None),
        (3, 'VI', 'Low Spin', 0.685, 0.545, 'R*'),
        (3, 'VI', 'High Spin', 0.75, 0.61, None),
        (4, 'IV', None, 0.54, 0.4, None),
        (4, 'VI', 'High Spin', 0.67, 0.53, 'R'),
    ),
    'Cr': (
        (2, 'VI', 'Low Spin', 0.87, 0.73, 'E'),
        (2, 'VI', 'High Spin', 0.94, 0.8, 'R*'),
        (3, 'VI', None, 0.755, 0.615, 'R*'),
        (4, 'IV', None, 0.55, 0.41, None),
        (4, 'VI', None, 0.69, 0.55, 'R'),
        (5, 'IV', None, 0.485, 0.345, 'R'),
        (5, 'VI', None, 0.63, 0.49, 'ER'),
        (5, 'VIII', None, 0.71, 0.57, None),
        (6, 'IV', None, 0.4, 0.26, None),
        (6, 'VI', None, 0.58, 0.44, 'C'),
    ),
    'Cs': (
        (1, 'VI', None, 1.81, 1.67, None),
        (1, 'VIII', None, 1.88, 1.74, None),
        (1, 'IX', None, 1.92, 1.78, None),
        (1, 'X', None, 1.95, 1.81, None),
        (1, 'XI', None, 1.99, 1.85, None),
        (1, 'XII', None, 2.02, 1.88, None),
    ),
    'Cu': (
        (1, 'II', None, 0.6, 0.46, None),
        (1, 'IV', None, 0.74, 0.6, 'E'),
        (1, 'VI', None, 0.91, 0.77, 'E'),
        (2, 'IV', None, 0.71, 0.57, None),
        (2, 'IVSQ', None, 0.71, 0.57, '*'),
        (2, 'V', None, 0.79, 0.65, '*'),
        (2, 'VI', None, 0.87, 0.73, None),
        (3, 'VI', 'Low Spin', 0.68, 0.54, None),
    ),
    'D': (
        (1, 'II', None, 0.04, -0.1, None),
    ),
    'Dy': (
        (2, 'VI', None, 1.21, 1.07, None),
        (2, 'VII', None, 1.27, 1.13, None),
        (2, 'VIII', None, 1.33, 1.19, None),
        (3, 'VI', None, 1.052, 0.912, 'R'),
        (3, 'VII', None, 1.11, 0.97, None),
        (3, 'VIII', None, 1.167, 1.027, 'R'),
        (3, 'IX', None, 1.223, 1.083, 'R'),
    ),
    'Er': (
        (3, 'VI', None, 1.03, 0.89, 'R'),
        (3, 'VII', None, 1.085, 0.945, None),
        (3, 'VIII', None, 1.144, 1.004, 'R'),
        (3, 'IX', None, 1.202, 1.062, 'R'),
    ),
    'Eu': (
        (2, 'VI', None, 1.31, 1.17, None),
        (2, 'VII', None, 1.34, 1.2, None),
        (2, 'VIII', None, 1.39, 1.25, None),
        (2, 'IX', None, 1.44, 1.3, None),
        (2, 'X', None, 1.49, 1.35, None),
        (3, 'VI', None, 1.087, 0.947, 'R'),
        (3, 'VII', None, 1.15, 1.01, None),
        (3, 'VIII', None, 1.206, 1.066, 'R'),
        (3, 'IX', None, 1.26, 1.12, 'R'),
    ),
    'F': (
        (-1, 'II', None, 1.145, 1.285, None),
        (-1, 'III', None, 1.16, 1.3, None),
        (-1, 'IV', None, 1.17, 1.31, None),
        (-1, 'VI', None, 1.19, 1.33, None),
        (7, 'VI', None, 0.22, 0.08, 'A'),
    ),
    'Fe': (
        (2, 'IV', 'High Spin', 0.77, 0.63, None),
        (2, 'IVSQ', 'High Spin', 0.78, 0.64, None),
        (2, 'VI', 'Low Spin', 0.75, 0.61, 'E'),
        (2, 'VI', 'High Spin', 0.92, 0.78, 'R*'),
        (2, 'VIII', 'High Spin', 1.06, 0.92, 'C'),
        (3, 'IV', 'High Spin', 0.63, 0.49, '*'),
        (3, 'V', None, 0.72, 0.58, None),
        (3, 'VI', 'Low Spin', 0.69, 0.55, 'R'),
        (3, 'VI', 'High Spin', 0.785, 0.645, 'R*'),
        (3, 'VIII', 'High Spin', 0.92, 0.78, None),
        (4, 'VI', None, 0.725, 0.585, 'R'),
        (6, 'IV', None, 0.39, 0.25, 'R'),
    ),
    'Fr': (
        (1, 'VI', None, 1.94, 1.8, 'A'),
    ),
    'Ga': (
        (3, 'IV', None, 0.61, 0.47, '*'),
        (3, 'V', None, 0.69, 0.55, None),
        (3, 'VI', None, 0.76, 0.62, 'R*'),
    ),
    'Gd': (
        (3, 'VI', None, 1.078, 0.938, 'R'),
        (3, 'VII', None, 1.14, 1.0, None),
        (3, 'VIII', None, 1.193, 1.053, 'R'),
        (3, 'IX', None, 1.247, 1.107, 'RC'),
    ),
    'Ge': (
        (2, 'VI', None, 0.87, 0.73, 'A'),
        (4, 'IV', None, 0.53, 0.39, '*'),
        (4, 'VI', None, 0.67, 0.53, 'R*'),
    ),
    'H': (
        (1, 'I', None, -0.24, -0.38, None),
        (1, 'II', None, -0.04, -0.18, None),
    ),
    'Hf': (
        (4, 'IV', None, 0.72, 0.58, 'R'),
        (4, 'VI', None, 0.85, 0.71, 'R'),
        (4, 'VII', None, 0.9, 0.76, None),
        (4, 'VIII', None, 0.97, 0.83, None),
    ),
    'Hg': (
        (1, 'III', None, 1.11, 0.97, None),
        (1, 'VI', None, 1.33, 1.19, None),
        (2, 'II', None, 0.83, 0.69, None),
        (2, 'IV', None, 1.1, 0.96, None),
        (2, 'VI', None, 1.16, 1.02, None),
        (2, 'VIII', None, 1.28, 1.14, 'R'),
    ),
    'Ho': (
        (3, 'VI', None, 1.041, 0.901, 'R'),
        (3, 'VIII', None, 1.155, 1.015, 'R'),
        (3, 'IX', None, 1.212, 1.072, 'R'),
        (3, 'X', None, 1.26, 1.12, None),
    ),
    'I': (
        (-1, 'VI', None, 2.06, 2.2, 'A'),
        (5, 'IIIPY', None, 0.58, 0.44, '*'),
        (5, 'VI', None, 1.09, 0.95, None),
        (7, 'IV', None, 0.56, 0.42, None),
        (7, 'VI', None, 0.67, 0.53, None),
    ),
    'In': (
        (3, 'IV', None, 0.76, 0.62, None),
        (3, 'VI', None, 0.94, 0.8, 'R*'),
        (3, 'VIII', None, 1.06, 0.92, 'RC'),
    ),
    'Ir': (
        (3, 'VI', None, 0.82, 0.68, 'E'),
        (4, 'VI', None, 0.765, 0.625, 'R'),
        (5, 'VI', None, 0.71, 0.57, 'EM'),
    ),
    'K': (
        (1, 'IV', None, 1.51, 1.37, None),
        (1, 'VI', None, 1.52, 1.38, None),
        (1, 'VII', None, 1.6, 1.46, None),
        (1, 'VIII', None, 1.65, 1.51, None),
        (1, 'IX', None, 1.69, 1.55, None),
        (1, 'X', None, 1.73, 1.59, None),
        (1, 'XII', None, 1.78, 1.64, None),
    ),
    'La': (
        (3, 'VI', None, 1.172, 1.032, 'R'),
        (3, 'VII', None, 1.24, 1.1, None),
        (3, 'VIII', None, 1.3, 1.16, 'R'),
        (3, 'IX', None, 1.356, 1.216, 'R'),
        (3, 'X', None, 1.41, 1.27, None),
        (3, 'XII', None, 1.5, 1.36, 'C'),
    ),
    'Li': (
        (1, 'IV', None, 0.73, 0.59, '*'),
        (1, 'VI', None, 0.9, 0.76, '*'),
        (1, 'VIII', None, 1.06, 0.92, 'C'),
    ),
    'Lu': (
        (3, 'VI', None, 1.001, 0.861, 'R'),
        (3, 'VIII', None, 1.117, 0.977, 'R'),
        (3, 'IX', None, 1.172, 1.032, 'R'),
    ),
    'Mg': (
        (2, 'IV', None, 0.71, 0.57, None),
        (2, 'V', None, 0.8, 0.66, None),
        (2, 'VI', None, 0.86, 0.72, '*'),
        (2, 'VIII', None, 1.03, 0.89, 'C'),
    ),
    'Mn': (
        (2, 'IV', 'High Spin', 0.8, 0.66, None),
        (2, 'V', 'High Spin', 0.89, 0.75, 'C'),
        (2, 'VI', 'Low Spin', 0.81, 0.67, 'E'),
        (2, 'VI', 'High Spin', 0.97, 0.83, 'R*'),
        (2, 'VII', 'High Spin', 1.04, 0.9, 'C'),
        (2, 'VIII', None, 1.1, 0.96, 'R'),
        (3, 'V', None, 0.72, 0.58, None),
        (3, 'VI', 'High Spin', 0.785, 0.645, 'R*'),
        (3, 'VI', 'Low Spin', 0.72, 0.58, 'R'),
        (4, 'IV', None, 0.53, 0.39, 'R'),
        (4, 'VI', None, 0.67, 0.53, 'R*'),
        (5, 'IV', None, 0.47, 0.33, 'R'),
        (6, 'IV', None, 0.395, 0.255, None),
        (7, 'IV', None, 0.39, 0.25, None),
        (7, 'VI', None, 0.6, 0.46, 'A'),
    ),
    'Mo': (
        (3, 'VI', None, 0.83, 0.69, 'E'),
        (4, 'VI', None, 0.79, 0.65, 'RM'),
        (5, 'IV', None, 0.6, 0.46, 'R'),
        (5, 'VI', None, 0.75, 0.61, 'R'),
        (6, 'IV', None, 0.55, 0.41, 'R*'),
        (6, 'V', None, 0.64, 0.5, None),
        (6, 'VI', None, 0.73, 0.59, 'R*'),
        (6, 'VII', None, 0.87, 0.73, None),
    ),
    'N': (
        (-3, 'IV', None, 1.32, 1.46, None),
        (3, 'VI', None, 0.3, 0.16, 'A'),
        (5, 'III', None, 0.044, -0.104, None),
        (5, 'VI', None, 0.27, 0.13, 'A'),
    ),
    'Na': (
        (1, 'IV', None, 1.13, 0.99, None),
        (1, 'V', None, 1.14, 1.0, None),
        (1, 'VI', None, 1.16, 1.02, None),
        (1, 'VII', None, 1.26, 1.12, None),
        (1, 'VIII', None, 1.32, 1.18, None),
        (1, 'IX', None, 1.38, 1.24, 'C'),
        (1, 'XII', None, 1.53, 1.39, None),
    ),
    'Nb': (
        (3, 'VI', None, 0.86, 0.72, None),
        (4, 'VI', None, 0.82, 0.68, 'RE'),
        (4, 'VIII', None, 0.93, 0.79, None),
        (5, 'IV', None, 0.62, 0.48, 'C'),
        (5, 'VI', None, 0.78, 0.64, None),
        (5, 'VII', None, 0.83, 0.69, 'C'),
        (5, 'VIII', None, 0.88, 0.74, None),
    ),
    'Nd': (
        (2, 'VIII', None, 1.43, 1.29, None),
        (2, 'IX', None, 1.49, 1.35, None),
        (3, 'VI', None, 1.123, 0.983, 'R'),
        (3, 'VIII', None, 1.249, 1.109, 'R*'),
        (3, 'IX', None, 1.303, 1.163, 'R'),
        (3, 'XII', None, 1.41, 1.27, 'E'),
    ),
    'Ni': (
        (2, 'IV', None, 0.69, 0.55, None),
        (2, 'IVSQ', None, 0.63, 0.49, None),
        (2, 'V', None, 0.77, 0.63, 'E'),
        (2, 'VI', None, 0.83, 0.69, 'R*'),
        (3, 'VI', 'High Spin', 0.74, 0.6, 'E'),
        (3, 'VI', 'Low Spin', 0.7, 0.56, 'R*'),
        (4, 'VI', 'Low Spin', 0.62, 0.48, 'R'),
    ),
    'No': (
        (2, 'VI', None, 1.24, 1.1, 'E'),
    ),
    'Np': (
        (2, 'VI', None, 1.24, 1.1, None),
        (3, 'VI', None, 1.15, 1.01, 'R'),
        (4, 'VI', None, 1.01, 0.87, 'R'),
        (4, 'VIII', None, 1.12, 0.98, 'R'),
        (5, 'VI', None, 0.89, 0.75, None),
        (6, 'VI', None, 0.86, 0.72, 'R'),
        (7, 'VI', None, 0.85, 0.71, 'A'),
    ),
    'O': (
        (-2, 'II', None, 1.21, 1.35, None),
        (-2, 'III', None, 1.22, 1.36, None),
        (-2, 'IV', None, 1.24, 1.38, None),
        (-2, 'VI', None, 1.26, 1.4, None),
        (-2, 'VIII', None, 1.28, 1.42, None),
    ),
    'OH': (
        (-1, 'II', None, 1.18, 1.32, None),
        (-1, 'III', None, 1.2, 1.34, None),
        (-1, 'IV', None, 1.21, 1.35, 'E'),
        (-1, 'VI', None, 1.23, 1.37, 'E'),
    ),
    'Os': (
        (4, 'VI', None, 0.77, 0.63, 'RM'),
        (5, 'VI', None, 0.715, 0.575, 'E'),
        (6, 'V', None, 0.63, 0.49, None),
        (6, 'VI', None, 0.685, 0.545, 'E'),
        (7, 'VI', None, 0.665, 0.525, 'E'),
        (8, 'IV', None, 0.53, 0.39, None),
    ),
    'P': (
        (3, 'VI', None, 0.58, 0.44, 'A'),
        (5, 'IV', None, 0.31, 0.17, '*'),
        (5, 'V', None, 0.43, 0.29, None),
        (5, 'VI', None, 0.52, 0.38, 'C'),
    ),
    'Pa': (
        (3, 'VI', None, 1.18, 1.04, 'E'),
        (4, 'VI', None, 1.04, 0.9, 'R'),
        (4, 'VIII', None, 1.15, 1.01, None),
        (5, 'VI', None, 0.92, 0.78, None),
        (5, 'VIII', None, 1.05, 0.91, None),
        (5, 'IX', None, 1.09, 0.95, None),
    ),
    'Pb': (
        (2, 'IVPY', None, 1.12, 0.98, 'C'),
        (2, 'VI', None, 1.33, 1.19, None),
        (2, 'VII', None, 1.37, 1.23, 'C'),
        (2, 'VIII', None, 1.43, 1.29, 'C'),
        (2, 'IX', None, 1.49, 1.35, 'C'),
        (2, 'X', None, 1.54, 1.4, 'C'),
        (2, 'XI', None, 1.59, 1.45, 'C'),
        (2, 'XII', None, 1.63, 1.49, None),
        (4, 'IV', None, 0.79, 0.65, 'E'),
        (4, 'V', None, 0.87, 0.73, 'E'),
        (4, 'VI', None, 0.915, 0.775, 'R'),
        (4, 'VIII', None, 1.08, 0.94, 'R'),
    ),
    'Pd': (
        (1, 'II', None, 0.73, 0.59, None),
        (2, 'IVSQ', None, 0.78, 0.64, None),
        (2, 'VI', None, 1.0, 0.86, None),
        (3, 'VI', None, 0.9, 0.76, None),
        (4, 'VI', None, 0.755, 0.615, 'R'),
    ),
    'Pm': (
        (3, 'VI', None, 1.11, 0.97, 'R'),
        (3, 'VIII', None, 1.233, 1.093, 'R'),
        (3, 'IX', None, 1.284, 1.144, 'R'),
    ),
    'Po': (
        (4, 'VI', None, 1.08, 0.94, 'R'),
        (4, 'VIII', None, 1.22, 1.08, 'R'),
        (6, 'VI', None, 0.81, 0.67, 'A'),
    ),
    'Pr': (
        (3, 'VI', None, 1.13, 0.99, 'R'),
        (3, 'VIII', None, 1.266, 1.126, 'R'),
        (3, 'IX', None, 1.319, 1.179, 'R'),
        (4, 'VI', None, 0.99, 0.85, 'R'),
        (4, 'VIII', None, 1.1, 0.96, 'R'),
    ),
    'Pt': (
        (2, 'IVSQ', None, 0.74, 0.6, None),
        (2, 'VI', None, 0.94, 0.8, 'A'),
        (4, 'VI', None, 0.765, 0.625, 'R'),
        (5, 'VI', None, 0.71, 0.57, 'ER'),
    ),
    'Pu': (
        (3, 'VI', None, 1.14, 1.0, 'R'),
        (4, 'VI', None, 1.0, 0.86, 'R'),
        (4, 'VIII', None, 1.1, 0.96, None),
        (5, 'VI', None, 0.88, 0.74, 'E'),
        (6, 'VI', None, 0.85, 0.71, 'R'),
    ),
    'Ra': (
        (2, 'VIII', None, 1.62, 1.48, 'R'),
        (2, 'XII', None, 1.84, 1.7, 'R'),
    ),
    'Rb': (
        (1, 'VI', None, 1.66, 1.52, None),
        (1, 'VII', None, 1.7, 1.56, None),
        (1, 'VIII', None, 1.75, 1.61, None),
        (1, 'IX', None, 1.77, 1.63, 'E'),
        (1, 'X', None, 1.8, 1.66, None),
        (1, 'XI', None, 1.83, 1.69, None),
        (1, 'XII', None, 1.86, 1.72, None),
        (1, 'XIV', None, 1.97, 1.83, None),
    ),
    'Re': (
        (4, 'VI', None, 0.77, 0.63, 'RM'),
        (5, 'VI', None, 0.72, 0.58, 'E'),
        (6, 'VI', None, 0.69, 0.55, 'E'),
        (7, 'IV', None, 0.52, 0.38, None),
        (7, 'VI', None, 0.67, 0.53, None),
    ),
    'Rh': (
        (3, 'VI', None, 0.805, 0.665, 'R'),
        (4, 'VI', None, 0.74, 0.6, 'RM'),
        (5, 'VI', None, 0.69, 0.55, None),
    ),
    'Ru': (
        (3, 'VI', None, 0.82, 0.68, None),
        (4, 'VI', None, 0.76, 0.62, 'RM'),
        (5, 'VI', None, 0.705, 0.565, 'ER'),
        (7, 'IV', None, 0.52, 0.38, None),
        (8, 'IV', None, 0.5, 0.36, None),
    ),
    'S': (
        (-2, 'VI', None, 1.7, 1.84, 'P'),
        (4, 'VI', None, 0.51, 0.37, 'A'),
        (6, 'IV', None, 0.26, 0.12, '*'),
        (6, 'VI', None, 0.43, 0.29, 'C'),
    ),
    'Sb': (
        (3, 'IVPY', None, 0.9, 0.76, None),
        (3, 'V', None, 0.94, 0.8, None),
        (3, 'VI', None, 0.9, 0.76, 'A'),
        (5, 'VI', None, 0.74, 0.6, '*'),
    ),
    'Sc': (
        (3, 'VI', None, 0.885, 0.745, 'R*'),
        (3, 'VIII', None, 1.01, 0.87, 'R*'),
    ),
    'Se': (
        (-2, 'VI', None, 1.84, 1.98, 'P'),
        (4, 'VI', None, 0.64, 0.5, 'A'),
        (6, 'IV', None, 0.42, 0.28, '*'),
        (6, 'VI', None, 0.56, 0.42, 'C'),
    ),
    'Si': (
        (4, 'IV', None, 0.4, 0.26, '*'),
        (4, 'VI', None, 0.54, 0.4, 'R*'),
    ),
    'Sm': (
        (2, 'VII', None, 1.36, 1.22, None),
        (2, 'VIII', None, 1.41, 1.27, None),
        (2, 'IX', None, 1.46, 1.32, None),
        (3, 'VI', None, 1.098, 0.958, 'R'),
        (3, 'VII', None, 1.16, 1.02, 'E'),
        (3, 'VIII', None, 1.219, 1.079, 'R'),
        (3, 'IX', None, 1.272, 1.132, 'R'),
        (3, 'XII', None, 1.38, 1.24, 'C'),
    ),
    'Sn': (
        (4, 'IV', None, 0.69, 0.55, 'R'),
        (4, 'V', None, 0.76, 0.62, 'C'),
        (4, 'VI', None, 0.83, 0.69, 'R*'),
        (4, 'VII', None, 0.89, 0.75, None),
        (4, 'VIII', None, 0.95, 0.81, 'C'),
    ),
    'Sr': (
        (2, 'VI', None, 1.32, 1.18, None),
        (2, 'VII', None, 1.35, 1.21, None),
        (2, 'VIII', None, 1.4, 1.26, None),
        (2, 'IX', None, 1.45, 1.31, None),
        (2, 'X', None, 1.5, 1.36, 'C'),
        (2, 'XII', None, 1.58, 1.44, 'C'),
    ),
    'Ta': (
        (3, 'VI', None, 0.86, 0.72, 'E'),
        (4, 'VI', None, 0.82, 0.68, 'E'),
        (5, 'VI', None, 0.78, 0.64, None),
        (5, 'VII', None, 0.83, 0.69, None),
        (5, 'VIII', None, 0.88, 0.74, None),
    ),
    'Tb': (
        (3, 'VI', None, 1.063, 0.923, 'R'),
        (3, 'VII', None, 1.12, 0.98, 'E'),
        (3, 'VIII', None, 1.18, 1.04, 'R'),
        (3, 'IX', None, 1.235, 1.095, 'R'),
        (4, 'VI', None, 0.9, 0.76, 'R'),
        (4, 'VIII', None, 1.02, 0.88, None),
    ),
    'Tc': (
        (4, 'VI', None, 0.785, 0.645, 'RM'),
        (5, 'VI', None, 0.74, 0.6, 'ER'),
        (7, 'IV', None, 0.51, 0.37, None),
        (7, 'VI', None, 0.7, 0.56, 'A'),
    ),
    'Te': (
        (-2, 'VI', None, 2.07, 2.21, 'P'),
        (4, 'III', None, 0.66, 0.52, None),
        (4, 'IV', None, 0.8, 0.66, None),
        (4, 'VI', None, 1.11, 0.97, None),
        (6, 'IV', None, 0.57, 0.43, 'C'),
        (6, 'VI', None, 0.7, 0.56, '*'),
    ),
    'Th': (
        (4, 'VI', None, 1.08, 0.94, 'C'),
        (4, 'VIII', None, 1.19, 1.05, 'RC'),
        (4, 'IX', None, 1.23, 1.09, '*'),
        (4, 'X', None, 1.27, 1.13, 'E'),
        (4, 'XI', None, 1.32, 1.18, 'C'),
        (4, 'XII', None, 1.35, 1.21, 'C'),
    ),
    'Ti': (
        (2, 'VI', None, 1.0, 0.86, 'E'),
        (3, 'VI', None, 0.81, 0.67, 'R*'),
        (4, 'IV', None, 0.56, 0.42, 'C'),
        (4, 'V', None, 0.65, 0.51, 'C'),
        (4, 'VI', None, 0.745, 0.605, 'R*'),
        (4, 'VIII', None, 0.88, 0.74, 'C'),
    ),
    'Tl': (
        (1, 'VI', None, 1.64, 1.5, 'R'),
        (1, 'VIII', None, 1.73, 1.59, 'R'),
        (1, 'XII', None, 1.84, 1.7, 'RE'),
        (3, 'IV', None, 0.89, 0.75, None),
        (3, 'VI', None, 1.025, 0.885, 'R'),
        (3, 'VIII', None, 1.12, 0.98, 'C'),
    ),
    'Tm': (
        (2, 'VI', None, 1.17, 1.03, None),
        (2, 'VII', None, 1.23, 1.09, None),
        (3, 'VI', None, 1.02, 0.88, 'R'),
        (3, 'VIII', None, 1.134, 0.994, 'R'),
        (3, 'IX', None, 1.192, 1.052, 'R'),
    ),
    'U': (
        (3, 'VI', None, 1.165, 1.025, 'R'),
        (4, 'VI', None, 1.03, 0.89, None),
        (4, 'VII', None, 1.09, 0.95, 'E'),
        (4, 'VIII', None, 1.14, 1.0, 'R*'),
        (4, 'IX', None, 1.19, 1.05, None),
        (4, 'XII', None, 1.31, 1.17, 'E'),
        (5, 'VI', None, 0.9, 0.76, None),
        (5, 'VII', None, 0.98, 0.84, 'E'),
        (6, 'II', None, 0.59, 0.45, None),
        (6, 'IV', None, 0.66, 0.52, None),
        (6, 'VI', None, 0.87, 0.73, '*'),
        (6, 'VII', None, 0.95, 0.81, 'E'),
        (6, 'VIII', None, 1.0, 0.86, None),
    ),
    'V': (
        (2, 'VI', None, 0.93, 0.79, None),
        (3, 'VI', None, 0.78, 0.64, 'R*'),
        (4, 'V', None, 0.67, 0.53, None),
        (4, 'VI', None, 0.72, 0.58, 'R*'),
        (4, 'VIII', None, 0.86, 0.72, 'E'),
        (5, 'IV', None, 0.495, 0.355, 'R*'),
        (5, 'V', None, 0.6, 0.46, '*'),
        (5, 'VI', None, 0.68, 0.54, None),
    ),
    'W': (
        (4, 'VI', None, 0.8, 0.66, 'RM'),
        (5, 'VI', None, 0.76, 0.62, 'R'),
        (6, 'IV', None, 0.56, 0.42, '*'),
        (6, 'V', None, 0.65, 0.51, None),
        (6, 'VI', None, 0.74, 0.6, '*'),
    ),
    'Xe': (
        (8, 'IV', None, 0.54, 0.4, None),
        (8, 'VI', None, 0.62, 0.48, None),
    ),
    'Y': (
        (3, 'VI', None, 1.04, 0.9, 'R*'),
        (3, 'VII', None, 1.1, 0.96, None),
        (3, 'VIII', None, 1.159, 1.019, 'R*'),
        (3, 'IX', None, 1.215, 1.075, 'R'),
    ),
    'Yb': (
        (2, 'VI', None, 1.16, 1.02, None),
        (2, 'VII', None, 1.22, 1.08, 'E'),
        (2, 'VIII', None, 1.28, 1.14, None),
        (3, 'VI', None, 1.008, 0.868, 'R*'),
        (3, 'VII', None, 1.065, 0.925, 'E'),
        (3, 'VIII', None, 1.125, 0.985, 'R'),
        (3, 'IX', None, 1.182, 1.042, 'R'),
    ),
    'Zn': (
        (2, 'IV', None, 0.74, 0.6, '*'),
        (2, 'V', None, 0.82, 0.68, '*'),
        (2, 'VI', None, 0.88, 0.74, 'R*'),
        (2, 'VIII', None, 1.04, 0.9, 'C'),
    ),
    'Zr': (
        (4, 'IV', None, 0.73, 0.59, 'R'),
        (4, 'V', None, 0.8, 0.66, 'C'),
        (4, 'VI', None, 0.86, 0.72, 'R*'),
        (4, 'VII', None, 0.92, 0.78, '*'),
        (4, 'VIII', None, 0.98, 0.84, '*'),
        (4, 'IX', None, 1.03, 0.89, None),
    ),
}


def _build_element_radii(ion: str, rows: tuple[tuple[int, str, str | None, float, float, str | None], ...]) -> ElementRadii:
    return ElementRadii(
        ion=ion,
        radii=tuple(
            ShannonIonicRadius(
                ion=ion,
                charge=charge,
                coordination=coordination,
                spin_state=spin_state,
                crystal_radius=crystal_radius,
                ionic_radius=ionic_radius,
                key=key,
            )
            for charge, coordination, spin_state, crystal_radius, ionic_radius, key in rows
        ),
    )


_BASE_SHANNON_IONIC_RADII = {
    ion: _build_element_radii(ion, rows)
    for ion, rows in _RAW_SHANNON_IONIC_RADII.items()
}

SHANNON_IONIC_RADII = {
    ion: ElementRadii(
        ion=ion,
        radii=base_element.radii
    )
    for ion, base_element in _BASE_SHANNON_IONIC_RADII.items()
}

def get_element_radii(ion: str) -> ElementRadii:
    """Return all Shannon radii entries for a given ionic species label."""
    return SHANNON_IONIC_RADII[ion]


def find_ionic_radii(
    ion: str | None = None,
    *,
    charge: int | None = None,
    coordination: str | None = None,
    spin_state: str | None = None,
) -> tuple[ShannonIonicRadius, ...]:
    """Query the hard-coded Shannon radii table."""
    if ion is not None:
        return get_element_radii(ion).find(
            charge=charge,
            coordination=coordination,
            spin_state=spin_state,
        )

    matches = []
    for element in SHANNON_IONIC_RADII.values():
        matches.extend(
            element.find(
                charge=charge,
                coordination=coordination,
                spin_state=spin_state,
            )
        )
    return tuple(matches)


__all__ = [
    "SHANNON_IONIC_RADII",
    "SHANNON_RADII_SOURCE_URL",
    "ShannonIonicRadius",
    "find_ionic_radii",
]
