all :
	python setup.py build_ext --inplace

test : all
	python -m pytest

html:
	make -C doc html &> /dev/null
