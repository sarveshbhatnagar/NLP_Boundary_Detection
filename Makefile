install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

format:
	black *.py
	
lint:
	pylint --disable=R,C SBD.py
	
test:
	python -m pytest -vv --cov=SBD test_SBD.py
	
all: install lint test format