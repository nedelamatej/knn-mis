.PHONY: pack

pack:
	cp report/report.pdf report.pdf
	zip -9 xnedel11_xmachu05.zip -r data report results/Qwen2.5* src README.txt report.pdf requirements.txt
