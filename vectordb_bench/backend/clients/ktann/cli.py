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

    refinement_rounds: Annotated[int, click.option("--refinement-rounds", type=click.IntRange(0, 5), default=2)]
    refinement_memory_bytes: Annotated[
        int, click.option("--refinement-memory-bytes", type=click.IntRange(min=1), default=32 << 30)
    ]


@cli.command()
@click_parameter_decorators_from_typed_dict(KTANNTypedDict)
def KTANN(**parameters: Unpack[KTANNTypedDict]):
    if parameters["case_type"] not in {
        "Performance768D1M", "Performance768D1M1P", "Performance768D1M99P",
        "PerformanceCustomDataset", "LabelFilterPerformanceCase",
    }:
        raise click.UsageError(
            "KTANN bridge supports Cohere 1M (including numeric and label filters) and custom ANN performance cases"
        )
    if parameters["case_type"] == "LabelFilterPerformanceCase":
        from ...dataset import DatasetWithSizeType

        dataset = DatasetWithSizeType.CohereMedium
        if parameters["dataset_with_size_type"] not in {None, dataset.value}:
            raise click.UsageError("KTANN label workloads support Medium Cohere (768dim, 1M)")
        if parameters["label_percentage"] not in dataset.get_manager().data.scalar_label_percentages:
            raise click.UsageError("label percentage has no official Cohere ground truth")
    from .config import KTANNCaseConfig, KTANNConfig

    run(
        db=DB.KTANN,
        db_config=KTANNConfig(
            db_label=parameters["db_label"],
            socket_path=parameters["socket_path"],
            dataset=parameters["dataset_identity"],
            companion_dir=parameters["companion_dir"],
        ),
        db_case_config=KTANNCaseConfig(
            leaf_beam=parameters["leaf_beam"],
            refinement_rounds=parameters["refinement_rounds"],
            refinement_memory_bytes=parameters["refinement_memory_bytes"],
        ),
        **parameters,
    )
