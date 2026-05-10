.PHONY: pack

pack:
	cp report/report.pdf report.pdf
	zip -9 xnedel11_xmachu05.zip -r data report results src README.txt requirements.txt
