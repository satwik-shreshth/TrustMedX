TEX = trustmedx_paper

.PHONY: paper serve clean

paper:
	pdflatex -interaction=nonstopmode $(TEX).tex
	bibtex $(TEX)
	pdflatex -interaction=nonstopmode $(TEX).tex
	pdflatex -interaction=nonstopmode $(TEX).tex

serve:
	python -m http.server 8000

clean:
	rm -f $(TEX).aux $(TEX).log $(TEX).bbl $(TEX).blg $(TEX).out
