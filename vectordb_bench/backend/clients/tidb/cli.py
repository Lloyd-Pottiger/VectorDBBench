from typing import Annotated, Unpack

import click
from pydantic import SecretStr

from vectordb_bench.backend.clients import DB

from ....cli.cli import CommonTypedDict, cli, click_parameter_decorators_from_typed_dict, run


class TiDBTypedDict(CommonTypedDict):
    delete: Annotated[
        bool,
        click.option(
            "--delete/--skip-delete",
            type=bool,
            default=False,
            help="Run delete benchmark after load/optimize",
            show_default=True,
        ),
    ]
    user_name: Annotated[
        str,
        click.option(
            "--username",
            type=str,
            help="Username",
            default="root",
            show_default=True,
        ),
    ]
    password: Annotated[
        str,
        click.option(
            "--password",
            type=str,
            default="",
            show_default=True,
            help="Password",
        ),
    ]
    host: Annotated[
        str,
        click.option(
            "--host",
            type=str,
            default="127.0.0.1",
            show_default=True,
            help="Db host",
        ),
    ]
    port: Annotated[
        int,
        click.option(
            "--port",
            type=int,
            default=4000,
            show_default=True,
            help="Db Port",
        ),
    ]
    db_name: Annotated[
        str,
        click.option(
            "--db-name",
            type=str,
            default="test",
            show_default=True,
            help="Db name",
        ),
    ]
    ssl: Annotated[
        bool,
        click.option(
            "--ssl/--no-ssl",
            default=False,
            show_default=True,
            is_flag=True,
            help="Enable or disable SSL, for TiDB Serverless SSL must be enabled",
        ),
    ]
    build_spfresh_index: Annotated[
        bool,
        click.option(
            "--build-spfresh-index/--skip-build-spfresh-index",
            default=True,
            show_default=True,
            is_flag=True,
            help="Create SPFRESH vector index during TiDB table creation",
        ),
    ]
    delete_commit_interval: Annotated[
        int,
        click.option(
            "--delete-commit-interval",
            type=int,
            default=1000,
            show_default=True,
            help="Commit every N delete statements during TiDB delete benchmark",
        ),
    ]
    delete_timeout: Annotated[
        float | None,
        click.option(
            "--delete-timeout",
            type=float,
            default=None,
            help="Override delete benchmark timeout in seconds; defaults to the case optimize timeout when unset",
        ),
    ]
    delete_search_wait_timeout: Annotated[
        float,
        click.option(
            "--delete-search-wait-timeout",
            type=float,
            default=30.0,
            show_default=True,
            help="Max seconds to wait for post-delete ANN results to disappear",
        ),
    ]
    delete_search_poll_interval: Annotated[
        float,
        click.option(
            "--delete-search-poll-interval",
            type=float,
            default=1.0,
            show_default=True,
            help="Polling interval in seconds for post-delete ANN verification",
        ),
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(TiDBTypedDict)
def TiDB(
    **parameters: Unpack[TiDBTypedDict],
):
    from .config import TiDBConfig, TiDBIndexConfig

    run(
        db=DB.TiDB,
        db_config=TiDBConfig(
            db_label=parameters["db_label"],
            user_name=parameters["username"],
            password=SecretStr(parameters["password"]),
            host=parameters["host"],
            port=parameters["port"],
            db_name=parameters["db_name"],
            ssl=parameters["ssl"],
        ),
        db_case_config=TiDBIndexConfig(
            build_spfresh_index=parameters["build_spfresh_index"],
            delete_commit_interval=parameters["delete_commit_interval"],
            delete_timeout=parameters["delete_timeout"],
            delete_search_wait_timeout=parameters["delete_search_wait_timeout"],
            delete_search_poll_interval=parameters["delete_search_poll_interval"],
        ),
        **parameters,
    )
