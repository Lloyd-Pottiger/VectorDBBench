"""Configuration for the private, explicitly launched KTANN benchmark bridge."""

from pydantic import BaseModel, Field

from ..api import DBCaseConfig, DBConfig, MetricType


class KTANNConfig(DBConfig):
    socket_path: str
    dataset: str
    companion_dir: str | None = None

    def to_dict(self) -> dict:
        return self.model_dump(include={"socket_path", "dataset", "companion_dir"})


class KTANNCaseConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType | None = None
    leaf_beam: int | None = Field(default=None, ge=1, le=16384)

    refinement_rounds: int = Field(default=2, ge=0, le=5)
    refinement_memory_bytes: int = Field(default=32 << 30, gt=0)

    def index_param(self) -> dict:
        return {
            "refinement_rounds": self.refinement_rounds,
            "refinement_memory_bytes": self.refinement_memory_bytes,
        }

    def search_param(self) -> dict:
        return {"leaf_beam": self.leaf_beam}
