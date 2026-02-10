from superset.semantic_layers.registry import registry

from .semantic_layer import PandasSemanticLayer

registry["pandas"] = PandasSemanticLayer
print("Pandas Semantic Layer extension registered")
