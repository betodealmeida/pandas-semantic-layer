from __future__ import annotations

from typing import Any

from superset_core.semantic_layers.semantic_layer import SemanticLayer

from pandas_semantic_layer.schemas import PandasConfiguration
from pandas_semantic_layer.semantic_view import PandasSemanticView


class PandasSemanticLayer(SemanticLayer[PandasConfiguration, PandasSemanticView]):
    id = "pandas"
    name = "Pandas Semantic Layer"
    description = "In-memory semantic layer backed by a Pandas DataFrame."

    @classmethod
    def from_configuration(
        cls,
        configuration: dict[str, Any],
    ) -> PandasSemanticLayer:
        config = PandasConfiguration.model_validate(configuration)
        return cls(config)

    @classmethod
    def get_configuration_schema(
        cls,
        configuration: PandasConfiguration | None = None,
    ) -> dict[str, Any]:
        return PandasConfiguration.model_json_schema()

    @classmethod
    def get_runtime_schema(
        cls,
        configuration: PandasConfiguration,
        runtime_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # No runtime parameters needed for in-memory data
        return {"type": "object", "properties": {}}

    def __init__(self, configuration: PandasConfiguration):
        self.configuration = configuration

    def get_semantic_views(
        self,
        runtime_configuration: dict[str, Any],
    ) -> set[PandasSemanticView]:
        return {PandasSemanticView("sales")}

    def get_semantic_view(
        self,
        name: str,
        additional_configuration: dict[str, Any],
    ) -> PandasSemanticView:
        if name != "sales":
            raise ValueError(f'Semantic view "{name}" does not exist.')
        return PandasSemanticView("sales")
