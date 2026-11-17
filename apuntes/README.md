# CC5513 lecture notes

[`CC5513_apuntes.pdf`](CC5513_apuntes.pdf) compiles the fourteen topics of the course into one
document (130 pages, in English). It explains each topic in detail, with derivations, worked
examples, algorithms, exercises and 78 figures computed for the notes.

| Chapter | Topic | Slides | Notebook |
|---|---|---|---|
| 1 | Introduction and pipeline | Tema 1 | |
| 2 | 3D representations | Tema 2 | `Tema02_Representaciones_3D` |
| 3 | Mesh data structures | Tema 3 | |
| 4 | Differential geometry of surfaces | Tema 4 | `Tema04_LaplaceBeltrami_Suavizado` |
| 5 | Laplace-Beltrami operator and smoothing | Tema 4 | `Tema04_LaplaceBeltrami_Suavizado` |
| 6 | 3D acquisition | Tema 5 | |
| 7 | Local registration (ICP) | Tema 5 | |
| 8 | Global registration (Harris 3D, spin images, 4PCS, RANSAC) | Tema 6 | `Tema06_4PCS_RANSAC` |
| 9 | Normals and surface reconstruction | Tema 7 | `Tema07_Normales_y_Reconstruccion` |
| 10 | Simplification and subdivision | Tema 8 | |
| 11 | Parameterization | Tema 9 | |
| 12 | Remeshing | Tema 10 | |
| 13 | Deformation | Tema 11 | |
| 14 | Shape analysis | Tema 12 | |
| 15 | Spectral shape analysis | Tema 13 | `Tema13_Analisis_Espectral` |
| 16 | Shape matching and functional maps | Tema 14 | `Tema14_Shape_Matching_Mapas_Funcionales` |
| A, B | Linear algebra toolkit; notebooks and code | | |

The notebooks are in [`../notebooks`](../notebooks).

## Layout

- `main.tex`, `preamble.tex`, `chapters/`, `refs.bib`: LaTeX sources.
- `figs/`: every figure of the notes (PNG and PDF).
- `scripts/`: the Python scripts that generate the figures. `gp.py` is a small shared library
  (mesh loading, cotangent and mass matrices, eigensolvers, curvature, geodesics, a software
  renderer); `fig_chXX_YY.py` produce the figures of the corresponding chapters and reuse the course
  demos in `../codigo` and the meshes in `../modelos`.

## Building

```bash
cd apuntes
make pdf          # latexmk -pdf -jobname=CC5513_apuntes main.tex
```

Requires a TeX Live installation with `latexmk`, `natbib`, TikZ, `algorithm2e`, `tcolorbox`,
`cleveref` and the Latin Modern fonts.

## Regenerating the figures

```bash
pip install -r requirements.txt     # on Linux, PyMeshLab also needs: apt install libopengl0
make figures                        # all scripts, a few minutes in total
cd scripts && python3 fig_ch14_16.py fmap_figures   # or only selected figure functions
```

Conventions: the analyst's Laplacian `Delta = M^-1 C` (C negative semi-definite) in chapters 4, 5
and 13, and the positive Laplacian `L = -C` with `L phi = lambda M phi` in chapters 15 and 16;
meshes are normalised to unit bounding-box diagonal or, in the spectral chapters, to unit area.
