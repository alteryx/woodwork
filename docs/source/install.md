# Install

Woodwork is available for Python 3.9 - 3.12. It can be installed from PyPI, conda-forge, or from source.

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

````{tab} PyPI
```{tab} All Add-ons
```console
$ python -m pip install "woodwork[complete]"
```
```{tab} Update Checker
```console
$ python -m pip install "woodwork[updater]"
```
````
````{tab} Conda
```{tab} All Add-ons
```console
$ conda install -c conda-forge alteryx-open-src-update-checker
```
```{tab} Update Checker
```console
$ conda install -c conda-forge alteryx-open-src-update-checker
```
````
- **Update Checker**: Receive automatic notifications of new Woodwork releases

## Source

To install Woodwork from source, clone the repository from [Github](https://github.com/alteryx/woodwork), and install the dependencies.

```bash
git clone https://github.com/alteryx/woodwork.git
cd woodwork
python -m pip install .
```

## Docker

It is also possible to run Woodwork inside a Docker container.
You can do so by installing it as a package inside a container (following the normal install guide) or
creating a new image with Woodwork pre-installed, using the following commands in your `Dockerfile`:

```dockerfile
FROM --platform=linux/x86_64 python:3.9-slim-buster
RUN apt update && apt -y update
RUN apt install -y build-essential
RUN pip3 install --upgrade --quiet pip
RUN pip3 install woodwork
```

## Optional Python Dependencies
Woodwork has several other Python dependencies that are used only for specific methods. Attempting to use one of these methods without having the necessary library installed will result in an ``ImportError`` with instructions on how to install the necessary dependency.

| Dependency        | Min Version | Notes                                  |
|-------------------|-------------|----------------------------------------|
| boto3             | 1.34.32     | Required to read/write to URLs and S3  |
| smart_open        | 5.0.0       | Required to read/write to URLs and S3  |
| pyarrow           | 15.0.0      | Required to serialize to parquet       |


# Development

To make contributions to the codebase, please follow the guidelines [here](https://github.com/alteryx/woodwork/blob/main/contributing.md).
