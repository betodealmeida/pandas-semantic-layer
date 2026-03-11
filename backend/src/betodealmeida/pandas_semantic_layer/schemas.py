from __future__ import annotations

from pydantic import BaseModel, Field


class PandasConfiguration(BaseModel):
    """
    Configuration for the Pandas semantic layer.

    Since the data is in-memory, no connection parameters are needed.
    """

    dataset: str = Field(
        default="sales",
        description="The dataset to use.",
        json_schema_extra={"enum": ["sales"]},
    )
