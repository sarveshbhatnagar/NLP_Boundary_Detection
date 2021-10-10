install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	black *.py
	
lint:
	pylint --disable=R,C SBD.py collocations.py
	
test:
	python -m pytest -vv --cov=SBD test_SBD.py
	
all: install lint test format

run:
	python SBD.py SBD.train SBD.test
	python SBD.py SBD.train SBD.test -o
	python SBD.py SBD.train SBD.test -c