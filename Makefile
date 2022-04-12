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
	pytest woodwork/ --durations 10

.PHONY: test-logical-types
test-logical-types:
	pytest woodwork/tests/logical_types/ --durations 0

.PHONY: testcoverage
testcoverage:
	pytest woodwork/ --cov=woodwork

.PHONY: installdeps
installdeps: upgradepip
	pip install -e ".[dev]"

.PHONY: checkdeps
checkdeps:
	$(eval allow_list='numpy|pandas|scikit|click|pyarrow|distributed|dask|pyspark')
	pip freeze | grep -v "woodwork.git" | grep -E $(allow_list) > $(OUTPUT_FILEPATH)

.PHONY: upgradepip
upgradepip:
	python -m pip install --upgrade pip

.PHONY: upgradebuild
upgradebuild:
	python -m pip install --upgrade build

.PHONY: package_woodwork
package_woodwork: upgradepip upgradebuild
	python -m build
	$(eval DT_VERSION := $(shell grep '__version__\s=' woodwork/version.py | grep -o '[^ ]*$$'))
	tar -zxvf "dist/woodwork-${DT_VERSION}.tar.gz"
	mv "woodwork-${DT_VERSION}" unpacked_sdist
