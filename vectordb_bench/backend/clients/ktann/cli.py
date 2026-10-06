"""Registration uses the VectorDBBench CLI's normal run path."""

from typing import Annotated, Unpack

import click

from ....cli.cli import CommonTypedDict, cli, click_parameter_decorators_from_typed_dict, run
from .. import DB


class KTANNTypedDict(CommonTypedDict):
    socket_path: Annotated[str, click.option("--socket-path", required=True)]
    dataset_identity: Annotated[str, click.option("--dataset-identity", required=True)]
    companion_dir: Annotated[str | None, click.option("--companion-dir", default=None)]
    leaf_beam: Annotated[int | None, click.option("--leaf-beam", type=int, default=None)]


@cli.command()
@click_parameter_decorators_from_typed_dict(KTANNTypedDict)
def KTANN(**parameters: Unpack[KTANNTypedDict]):
    if parameters["case_type"] not in {"Performance768D1M", "Performance768D1M1P", "Performance768D1M99P", "PerformanceCustomDataset"}:
        raise click.UsageError("KTANN bridge supports Cohere 1M (including numeric filters) and custom ANN performance cases only")
    from .config import KTANNCaseConfig, KTANNConfig

    run(
        db=DB.KTANN,
        db_config=KTANNConfig(
            db_label=parameters["db_label"],
            socket_path=parameters["socket_path"],
            dataset=parameters["dataset_identity"],
            companion_dir=parameters["companion_dir"],
        ),
        db_case_config=KTANNCaseConfig(leaf_beam=parameters["leaf_beam"]),
        **parameters,
    )
