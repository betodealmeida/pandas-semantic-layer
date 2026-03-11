# The @semantic_layer decorator on PandasSemanticLayer handles registration
# automatically. This import triggers the decorator at extension load time.
from .layer import PandasSemanticLayer  # noqa: F401
