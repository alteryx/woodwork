import click
import pkg_resources
import woodwork as ww
import pandas as pd


@click.group()
def cli():
    pass


@click.command(name='list-logical-types')
def list_ltypes():
    print(ww.utils.list_logical_types())


cli.add_command(list_ltypes)


if __name__ == "__main__":
    cli()
