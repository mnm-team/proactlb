.DEFAULT_GOAL=all
PDFFINAL:=../pdf/$(shell pwd | rev | cut -f 3 -d / | rev).pdf

#SUBDIRS = pictures
BIBINPUTS:=./bib/*.bib
GLSINPUTS:=./bib/*.tex
INPUT = main.tex content/*.tex ./Abstract.txt  ./Zusammenfassung.txt

.PHONY = all
all: main

main: $(INPUT)
	latexmk -pdf $<
	cp main.pdf $(PDFFINAL)

cont:
	latexmk -pdflatex='pdflatex -interaction=nonstopmode -halt-on-error -file-line-error -synctex=1' -pvc -view=none -pdf $<

clean:
	-rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb *.auxlock main.pdf *.synctex.gz *.run.xml *.bbl *.bcf *.blg *.lof *.fdb_latexmk *.fls *.cb *.cb2
	-rm -f *.bak *.~ *.dvi
	-rm -f main.acr main.glg main.gls main.idx main.ind main.lot main.slo
	-rm -f main.acn main.alg main.glo main.glsdefs main.ilg main.ist main.slg main.sls
