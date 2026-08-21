# LaFeO3 Defect Examples

Run inside the conda env where you have installed quick-mag:

```bash
python examples/build_with_defects/lafeo3_defects.py
```

The script uses the quick_mag Python API to build four related 2x2x2 LaFeO3
cells:

- pristine LaFeO3
- LaFeO3 with one randomly selected X-site oxygen vacancy
- LaFeO3 with Zn substituted onto one Fe/B site
- the same Zn substitution plus one proton on a neighboring oxygen for charge compensation

For each structure, it writes a CIF into `generated/`, prints the top quick_mag
oxidation-state assignment, and reports CHGNet magnetic-moment diagnostics when
the optional CHGNet dependencies are installed.