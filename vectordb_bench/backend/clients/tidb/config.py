from enum import Enum
from typing import Any, TypedDict

from pydantic import BaseModel, SecretStr, validator

from ..api import DBCaseConfig, DBConfig, MetricType


class TiDBConfigDict(TypedDict):
    host: str
    port: int
    user: str
    password: str
    database: str
    ssl_verify_cert: bool
    ssl_verify_identity: bool


class TiDBConfig(DBConfig):
    user_name: str = "root"
    password: SecretStr
    host: str = "127.0.0.1"
    port: int = 4000
    db_name: str = "test"
    ssl: bool = False

    def to_dict(self) -> TiDBConfigDict:
        pwd_str = self.password.get_secret_value()
        return {
            "host": self.host,
            "port": self.port,
            "user": self.user_name,
            "password": pwd_str,
            "database": self.db_name,
            "ssl_verify_cert": self.ssl,
            "ssl_verify_identity": self.ssl,
        }

    @validator("*")
    def not_empty_field(cls, v: any, field: any):
        if v is None:
            return v
        if field.name in ["password", "db_label"]:
            return v
        if isinstance(v, str | SecretStr) and len(v) == 0:
            raise ValueError("Empty string!")
        return v


class SPFreshBuildMode(str, Enum):
    INLINE = "inline"
    NON_INLINE = "non-inline"
    SPLIT = "split"


class TiDBIndexConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType | None = None
    build_spfresh_index: bool = True
    spfresh_build_mode: SPFreshBuildMode = SPFreshBuildMode.INLINE
    spfresh_split_ratio: float = 0.8
    delete_commit_interval: int = 1000
    delete_timeout: float | None = None
    delete_search_wait_timeout: float = 30.0
    delete_search_poll_interval: float = 1.0

    @validator("delete_commit_interval")
    def validate_delete_commit_interval(cls, v: int):
        if v < 1:
            raise ValueError("delete_commit_interval must be >= 1")
        return v

    @validator("spfresh_split_ratio")
    def validate_spfresh_split_ratio(cls, v: float):
        if v <= 0 or v >= 1:
            raise ValueError("spfresh_split_ratio must be > 0 and < 1")
        return v

    @validator("delete_timeout", "delete_search_wait_timeout", "delete_search_poll_interval")
    def validate_positive_delete_search_window(cls, v: float, field: Any):
        if v is None and field.name == "delete_timeout":
            return v
        if v <= 0:
            msg = f"{field.name} must be > 0"
            raise ValueError(msg)
        return v

    def get_metric_fn(self) -> str:
        if self.metric_type == MetricType.L2:
            return "vec_l2_distance"
        if self.metric_type == MetricType.COSINE:
            return "vec_cosine_distance"
        msg = f"Unsupported metric type: {self.metric_type}"
        raise ValueError(msg)

    def index_param(self) -> dict:
        return {
            "build_spfresh_index": self.build_spfresh_index,
            "spfresh_build_mode": self.spfresh_build_mode.value,
            "spfresh_split_ratio": self.spfresh_split_ratio,
            "metric_fn": self.get_metric_fn(),
        }

    def search_param(self) -> dict:
        return {
            "metric_fn": self.get_metric_fn(),
        }
