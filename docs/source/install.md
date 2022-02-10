# Install

Woodwork is available for Python 3.7, 3.8 and 3.9. It can be installed from PyPi, conda-forge, or from source.

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

Woodwork allows users to install add-ons. Woodwork allows users to install add-ons individually or all at once:

```{hint}
Be sure to install [Scala and Spark](#scala-and-spark) if you want to use Koalas
```

````{tab} PyPI
```{tab} All Add-ons
```console
$ python -m pip install "woodwork[complete]"
```
```{tab} Dask
```console
$ python -m pip install "woodwork[dask]"
```
```{tab} Koalas
```console
$ python -m pip install "woodwork[koalas]"
```
```{tab} Update Checker
```console
$ python -m pip install "woodwork[update_checker]"
```
````
````{tab} Conda
```{tab} All Add-ons
```console
$ conda install -c conda-forge dask koalas pyspark alteryx-open-src-update-checker
```
```{tab} Dask
```console
$ conda install -c conda-forge dask
```
```{tab} Koalas
```console
$ conda install -c conda-forge koalas pyspark
```
```{tab} Update Checker
```console
$ conda install -c conda-forge alteryx-open-src-update-checker
```
````
- **Dask**: Use Woodwork with Dask DataFrames
- **Koalas**: Use Woodwork with Koalas DataFrames
- **Update Checker**: Receive automatic notifications of new Woodwork releases

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
$ echo "export SPARK_HOME=/opt/spark" >> ~/.profile
$ echo "export PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin" >> ~/.profile
$ echo "export PYSPARK_PYTHON=/usr/bin/python3" >> ~/.profile
```
````

````{tab} Amazon Linux
```console
$ sudo amazon-linux-extras install java-openjdk11 scala -y
$ amazon-linux-extras enable java-openjdk11
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
