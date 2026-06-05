# Vibration analysis post-processing

Post-processing scripts and LaTeX report for the sounding-rocket vibration
study developed for the Noise and Vibrations course.

## Running the scripts

The data and results paths are relative, so the scripts must be run
from the project root.

```bash
uv run python scripts/convergence.py
uv run python scripts/harmonic.py
```

## Building the report

The report is built with LuaLaTeX (not XeLaTeX/pdfLaTeX) and requires
`--shell-escape` for SVG inclusion via Inkscape:

```bash
cd latex
lualatex -shell-escape main.tex && bibtex main && lualatex -shell-escape main.tex && lualatex -shell-escape main.tex
```

## Author

Alejandro Gil Getino 
Universidad de León  
agilge00@estudiantes.unileon.es
