from __future__ import annotations

from typing import Any

from superset_core.semantic_layers.decorators import semantic_layer
from superset_core.semantic_layers.layer import SemanticLayer

from .schemas import PandasConfiguration
from .view import PandasSemanticView

VIEW_NAMES = frozenset({"sales", "marketing"})


@semantic_layer(
    id="pandas",
    name="Pandas Semantic Layer",
    description="In-memory semantic layer backed by a Pandas DataFrame.",
)
class PandasSemanticLayer(SemanticLayer[PandasConfiguration, PandasSemanticView]):
    configuration_class = PandasConfiguration

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
        return {PandasSemanticView(name) for name in VIEW_NAMES}

    def get_semantic_view(
        self,
        name: str,
        additional_configuration: dict[str, Any],
    ) -> PandasSemanticView:
        if name not in VIEW_NAMES:
            raise ValueError(f'Semantic view "{name}" does not exist.')
        return PandasSemanticView(name)
