.PHONY: clean
clean:
	find . -name '*.pyo' -delete
	find . -name '*.pyc' -delete
	find . -name __pycache__ -delete
	find . -name '*~' -delete
	find . -name '.coverage.*' -delete

.PHONY: lint
lint:
	isort --check-only woodwork
	python docs/notebook_version_standardizer.py check-execution
	black woodwork -t py39 --check
	flake8 woodwork

.PHONY: lint-fix
lint-fix:
	black -t py39 woodwork
	isort woodwork
	python docs/notebook_version_standardizer.py standardize

.PHONY: test
test:
	pytest woodwork/

.PHONY: testcoverage
testcoverage:
	pytest woodwork/ --cov=woodwork

.PHONY: installdeps
installdeps:
	pip install --upgrade pip
	pip install -e .
	pip install -e .[dev]

.PHONY: installdeps-test
installdeps-test:
	pip install -e .[test]

.PHONY: installdeps-dev
installdeps-dev:
	pip install -e .[dev]

.PHONY: checkdeps
checkdeps:
	$(eval allow_list='numpy|pandas|scikit|click|pyarrow|distributed|dask|pyspark')
	pip freeze | grep -v "woodwork.git" | grep -E $(allow_list) > $(OUTPUT_FILEPATH)

.PHONY: package_woodwork
package_woodwork:
	python setup.py sdist
	$(eval DT_VERSION=$(shell python setup.py --version))
	tar -zxvf "dist/woodwork-${DT_VERSION}.tar.gz"
	mv "woodwork-${DT_VERSION}" unpacked_sdist
