# Build the lecture notes and regenerate the figures.
NAME = CC5513_apuntes
PY = python3

.PHONY: pdf figures clean

pdf:
	latexmk -pdf -interaction=nonstopmode -jobname=$(NAME) main.tex

figures:
	cd scripts && for s in fig_ch01_03 fig_ch04_05 fig_ch06_08 fig_ch09 fig_ch10_13 fig_ch14_16; do \
		PYTHONDONTWRITEBYTECODE=1 $(PY) $$s.py || exit 1; done

clean:
	latexmk -c -jobname=$(NAME) main.tex
