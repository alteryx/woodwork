import click

import woodwork as ww


@click.group()
def cli():
    pass


@click.command(name='list-logical-types')
def list_ltypes():
    print(ww.list_logical_types())


@click.command(name='list-semantic-tags')
def list_stags():
    print(ww.list_semantic_tags())


cli.add_command(list_ltypes)
cli.add_command(list_stags)


if __name__ == "__main__":
    cli()
