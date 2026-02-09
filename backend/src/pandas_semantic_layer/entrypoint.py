from pandas_semantic_layer.semantic_layer import PandasSemanticLayer
from superset.semantic_layers.registry import registry

registry["pandas"] = PandasSemanticLayer
print("Pandas Semantic Layer extension registered")
