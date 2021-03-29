.PHONY: clean
clean:
	find . -name '*.pyo' -delete
	find . -name '*.pyc' -delete
	find . -name __pycache__ -delete
	find . -name '*~' -delete

.PHONY: lint
lint:
	flake8 woodwork && isort --check-only woodwork

.PHONY: lint-fix
lint-fix:
	autopep8 --in-place --recursive --max-line-length=100 --exclude="*/migrations/*" --select="E225,E303,E302,E203,E128,E231,E251,E271,E127,E126,E301,W291,W293,E226,E306,E221,E261,E111,E114" woodwork
	isort woodwork

.PHONY: test
test: lint
	pytest woodwork/

.PHONY: testcoverage
testcoverage: lint
	pytest woodwork/ --cov=woodwork

.PHONY: installdeps
installdeps:
	pip install --upgrade pip
	pip install -e .
	pip install -r dev-requirements.txt

.PHONY: installdeps-test
installdeps-test:
	pip install -r test-requirements.txt

.PHONY: installdeps-dev
installdeps-dev:
	pip install -r dev-requirements.txt

.PHONY: checkdeps
checkdeps:
	$(eval allow_list='numpy|pandas|scikit|click|pyarrow|distributed|dask|pyspark|koalas')
	pip freeze | grep -v "woodwork.git" | grep -E $(allow_list) > $(OUTPUT_PATH)

.PHONY: generate-min-deps
generate-min-deps:
	python tools/minimum_dependency/minimum_dependency_generator.py $(OUTPUT_PATH) --requirements_paths $(INPUT_PATHS)

.PHONY: test-min-deps-gen
test-min-deps-gen:
	python -m pytest tools/minimum_dependency/test_minimum_dependency_generator.py

.PHONY: package_woodwork
package_woodwork:
	python setup.py sdist
	$(eval DT_VERSION=$(shell python setup.py --version))
	tar -zxvf "dist/woodwork-${DT_VERSION}.tar.gz"
	mv "woodwork-${DT_VERSION}" unpacked_sdist
