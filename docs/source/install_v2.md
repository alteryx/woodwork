# Install V2

Woodwork is available for Python 3.8 and 3.9. It can be installed from PyPi, conda-forge, or from source.

To install Woodwork, run the following command:

````{tab} PyPI
```console
$ python -m pip install woodwork
```
````

````{tab} Conda
```console
$ conda install -c conda-forge woodwork
```
````

## Add-ons


Woodwork allows users to install add-ons.

### All Add-ons
````{tab} PyPI
:new-set:
```console
$ python -m pip install "woodwork[complete]"
```
````

````{tab} Conda
```console
$ conda install -c conda-forge dask koalas pyspark alteryx-open-src-update-checker
```
````

### Dask
You can use Woodwork with Dask DataFrames by running:

````{tab} PyPI
:new-set:
```console
$ python -m pip install "woodwork[dask]"
```
````

````{tab} Conda
```console
$ conda install -c conda-forge dask
```
````

### Koalas
You can use Woodwork with Koalas DataFrames by running:

````{tab} PyPI
:new-set:
```{hint}
Be sure to install [Scala and Spark](#scala-and-spark) if you want to use Koalas
```

```console
$ python -m pip install "woodwork[koalas]"
```
````

````{tab} Conda
```{hint}
Be sure to install [Scala and Spark](#scala-and-spark) if you want to use Koalas
```
```console
$ conda install -c conda-forge koalas pyspark
```
````

#### Update Checker
You can receive automatic notifications of new Woodwork releases

````{tab} PyPI
:new-set:
```console
$ python -m pip install "woodwork[update_checker]"
```
````

````{tab} Conda
```console
$ conda install -c conda-forge alteryx-open-src-update-checker
```
````

## Source

To install Woodwork from source, clone the repository from [Github](https://github.com/alteryx/woodwork), and install the dependencies.

```{hint}
Be sure to install [Scala and Spark](#scala-and-spark) if you want to run all unit tests
```

```bash
git clone https://github.com/alteryx/woodwork.git
cd woodwork
python -m pip install .
```

## Scala and Spark

````{tab} macOS
:new-set:
```console
$ brew tap AdoptOpenJDK/openjdk
$ brew install --cask adoptopenjdk11
$ brew install scala apache-spark
$ echo 'export JAVA_HOME=$(/usr/libexec/java_home)' >> ~/.zshrc
$ echo 'export PATH="/usr/local/opt/openjdk@11/bin:$PATH"' >> ~/.zshrc
```
````

````{tab} Ubuntu
```console
$ sudo apt install openjdk-11-jre openjdk-11-jdk scala -y
```
````

````{tab} Amazon Linux
```console
$ sudo amazon-linux-extras install java-openjdk11 scala -y
```
````

## Optional Python Dependencies
Woodwork has several other Python dependencies that are used only for specific methods. Attempting to use one of these methods without having the necessary library installed will result in an ``ImportError`` with instructions on how to install the necessary dependency.

| Dependency        | Min Version | Notes                                  |
|-------------------|-------------|----------------------------------------|
| boto3             | 1.10.45     | Required to read/write to URLs and S3  |
| smart_open        | 5.0.0       | Required to read/write to URLs and S3  |
| pyarrow           | 4.0.1       | Required to serialize to parquet       |
| dask[distributed] | 2021.10.0   | Required to use with Dask DataFrames   |
| koalas            | 1.8.0       | Required to use with Koalas DataFrames |
| pyspark           | 3.0.0       | Required to use with Koalas DataFrames |


# Development

To make contributions to the codebase, please follow the guidelines [here](https://github.com/alteryx/woodwork/blob/main/contributing.md).