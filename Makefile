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
	black woodwork docs/source -t py310 --check
	flake8 woodwork

.PHONY: lint-fix
lint-fix:
	black woodwork docs/source -t py310
	isort woodwork
	python docs/notebook_version_standardizer.py standardize

.PHONY: test
test:
	pytest woodwork/ -n auto

.PHONY: testcoverage
testcoverage:
	pytest woodwork/ --cov=woodwork -n auto

.PHONY: installdeps
installdeps: upgradepip
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
	$(eval PACKAGE=$(shell python -c "from pep517.meta import load; metadata = load('.'); print(metadata.version)"))
	tar -zxvf "dist/woodwork-${PACKAGE}.tar.gz"
	mv "woodwork-${PACKAGE}" unpacked_sdist
