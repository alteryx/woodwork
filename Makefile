.PHONY: clean
clean:
	find . -name '*.pyo' -delete
	find . -name '*.pyc' -delete
	find . -name __pycache__ -delete
	find . -name '*~' -delete
	find . -name '.coverage.*' -delete

.PHONY: lint
lint:
	python docs/notebook_version_standardizer.py check-execution
	black . --config=./pyproject.toml --check
	ruff . --config=./pyproject.toml

.PHONY: lint-fix
lint-fix:
	python docs/notebook_version_standardizer.py standardize
	black . --config=./pyproject.toml
	ruff . --config=./pyproject.toml --fix

.PHONY: test
test:
	pytest woodwork/

.PHONY: testcoverage
testcoverage:
	pytest woodwork/ --cov=woodwork

.PHONY: installdeps
installdeps: upgradepip
	pip install -e .

.PHONY: installdeps-dev
installdeps-dev: upgradepip
	pip install -e ".[dev]"
	pre-commit install

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

.PHONY: upgradesetuptools
upgradesetuptools:
	python -m pip install --upgrade setuptools

.PHONY: package
package: upgradepip upgradebuild upgradesetuptools
	python -m build
	$(eval PACKAGE=$(shell python -c 'import setuptools; setuptools.setup()' --version))
	tar -zxvf "dist/woodwork-${PACKAGE}.tar.gz"
	mv "woodwork-${PACKAGE}" unpacked_sdist
