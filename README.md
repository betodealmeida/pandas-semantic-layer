# pandas-semantic-layer
A dummy semantic layer extension for Superset.

This extension now exposes two semantic views:

- `sales`
- `marketing`

Each view includes overlapping but intentionally constrained metric/dimension compatibility,
so developers can test `get_compatible_metrics` and `get_compatible_dimensions` behavior.
