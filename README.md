# symmscreen

Symmetry projectors and modulation-loss (Lambda) estimators for screening molecular crystals for directional dark matter detection.

Companion package to **"The Role of Symmetries in Dark Matter Detector Design"** (arXiv:..., Benjamin Lillard, Jack D. Shergold, and Juri Smirnov). The paper develops a symmetry-projection framework that predicts how much of a molecule's directional dark matter scattering signal survives crystallisation, using only the crystal's space group and the molecule's point-group symmetry and geometry. This package implements exactly that framework (paper Secs. 3-6): the crystal and molecule point-group projectors, the combined survival operator, and the three Lambda estimators for the fractional modulation-signal loss.

This package is intentionally lightweight, and does not compute electronic structure, form factors, scattering rates, or true modulation amplitudes. See [SCarFFF](https://github.com/jdshergold/SCarFFF) and [vsdm](https://github.com/blillard/vsdm) for software implementing these calculations.

## Installation

```bash
pip install symmscreen
```

## Quickstart

The fastest way to get the modulation loss estimate, $\Lambda_\mathrm{coord}$, for a molecular crystal with known coordinates is:

```python
import symmscreen as ss

ss.lambda_coord("your_crystal.cif")
```

There's a matching one-line function for the quadrupole class of the crystal, and of the molecule itself, too: `ss.crystal_quadrupole_class(cif_path)`, `ss.molecule_quadrupole_class(cif_path)`.

If you want more than one metric for the same crystal, or want to inspect the underlying projector matrices, build a `CombinedSurvival` object directly instead. This caches the crystal/molecule projectors internally, so nothing is recomputed between calls. Since a CIF gives you the molecule's actual orientation, $\mathcal{T}$, the natural estimators here are the orientation-specific ones, `lambda_L` and `lambda_coord`:

```python
from symmscreen import CombinedSurvival

survival = CombinedSurvival.from_cif("your_crystal.cif")

survival.crystal.quadrupole_class      # k in {0, 1, 2, 3, 5}, the quadrupole class of the crystal, Q_k.
survival.molecule.quadrupole_class     # k in {0, 1, 2, 3, 5}, the quadrupole class of the molecule's own symmetry H.
survival.crystal.matrix(l=2)           # Pi_2^(L), the l=2 projector for the crystal symmetry.
survival.molecule.matrix(l=2)          # Pi_2^(H), the l=2 projector for the internal molecule symmetry.
survival.matrix(l=2)                   # C^(2)(T), the combined survival operator.
survival.lambda_L()                    # Lambda^(L)(T), modulation signal loss due to crystallisation alone, for this crystal's specific orientation, T.
survival.lambda_coord()                # Lambda_coord, coordinate-aware estimator of total modulation signal loss.
```

Or work purely from symmetry labels, with no CIF or coordinates at all. Here the relative orientation, $\mathcal{T}$, is generally *unknown*, so the natural estimators are the analytic $\mathcal{T}$-averaged quantities, `lambda_L_avg` and `lambda_ideal_avg`. These have matching one-line functions that take the same symmetry labels rather than a CIF:

```python
import symmscreen as ss

ss.lambda_L_avg("D6h", crys_pg_symbol="D2h")
ss.lambda_ideal_avg("D6h", crys_pg_symbol="D2h")   # ~ sqrt(2)/5, matching the benzene example in the paper.
```

Or build the combined survival operator and access them from there, as well as the matrices:

```python
from symmscreen import CombinedSurvival

survival = CombinedSurvival.from_symmetry(mol_pg_symbol="D6h", crys_pg_symbol="D2h")
survival.matrix(l=2)          # C^(2), combined survival operator at l=2. Defaults to T=I, with T the relative orientation.
survival.lambda_L_avg()       # <Lambda^(L)>_T, loss due to crystallisation alone, averaged over T.
survival.lambda_ideal_avg()   # <Lambda>_T, ~sqrt(2)/5, matching the benzene example in the paper.
```

`from_symmetry` also takes an optional relative rotation `T` (the paper's
$\mathcal{T}$), for when you want a specific embedding rather than the $\mathcal{T}$-averaged estimators above:

```python
from symmscreen import CombinedSurvival
from scipy.spatial.transform import Rotation

T = Rotation.from_euler("z", 90, degrees=True).as_matrix()   # e.g. a 90 degree rotation about z.

survival = CombinedSurvival.from_symmetry(mol_pg_symbol="D6h", crys_pg_symbol="D2h", T=T)
survival.matrix(l=2)  # C^(2)(T), combined survival operator at l=2 for a fixed T.
survival.lambda_L()   # Lambda^(L)(T), orientation-specific crystallistion loss, rather than T-averaged.
```

$\mathcal{T}$ defaults to the identity if omitted.

## Getting a CIF

`symmscreen` does not contain any example structures, as crystal structure databases are generally not freely redistributable. However, experimentally determined molecular crystal structures are available from:

- [Cambridge Structural Database (CCDC)](https://www.ccdc.cam.ac.uk/) — the largest database of organic and metal-organic crystal structures. Structures can be visualised, and CIFs can be downloaded via [CCDC Access Structures](https://www.ccdc.cam.ac.uk/structures/). Full access requires an institutional licence.
- [Crystallography Open Database (COD)](https://www.crystallography.net/cod/) — fully open-access, no licence required.

## Citation

If you use this package, please use the following citation: 

Bibtex goes here.
