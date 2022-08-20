from typing import List
from dataclasses import (dataclass, field)


@dataclass
class GCPConfig:
    project_id: str = "auto-trader-359210"
    bucket_name: str = "preprocessed-thashimoto"
    secret_id: str = "neptune_api_key"


@dataclass
class NeptuneConfig:
    project: str = "thashimoto/auto-trader"
    model_id: str = "AUT-LGBM"
    model_version_id: str = ""
