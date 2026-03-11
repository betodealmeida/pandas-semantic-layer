from __future__ import annotations

import pyarrow as pa
import pyarrow.compute as pc
from superset_core.semantic_layers.view import (
    SemanticView,
    SemanticViewFeature,
)
from superset_core.semantic_layers.types import (
    Dimension,
    Filter,
    GroupLimit,
    SemanticQuery,
    Metric,
    Operator,
    OrderDirection,
    OrderTuple,
    PredicateType,
    SemanticRequest,
    SemanticResult,
)

from .data import (
    DIMENSION_COLUMNS,
    METRIC_DEFINITIONS,
    SALES_DATA,
)

REQUEST_TYPE = "pandas"


class PandasSemanticView(SemanticView):
    features = frozenset(
        {
            SemanticViewFeature.GROUP_LIMIT,
        }
    )

    def __init__(self, name: str):
        self.name = name
        self._data = SALES_DATA
        self.dimensions = self.get_dimensions()
        self.metrics = self.get_metrics()

    def uid(self) -> str:
        return f"pandas.{self.name}"

    def get_dimensions(self) -> set[Dimension]:
        type_map = {
            "product_category": pa.utf8(),
            "region": pa.utf8(),
            "sale_date": pa.date32(),
        }
        return {
            Dimension(
                dim_id,
                col_name,
                type_map[col_name],
                f"The {col_name.replace('_', ' ')} dimension.",
                col_name,
            )
            for dim_id, col_name in DIMENSION_COLUMNS.items()
        }

    def get_metrics(self) -> set[Metric]:
        type_map = {
            "total_revenue": pa.decimal128(38, 10),
            "total_units_sold": pa.int64(),
            "average_price": pa.decimal128(38, 10),
        }
        agg_labels = {
            "total_revenue": "SUM(revenue)",
            "total_units_sold": "SUM(units_sold)",
            "average_price": "AVG(price)",
        }
        return {
            Metric(
                metric_id,
                metric_name,
                type_map[metric_name],
                agg_labels[metric_name],
                f"The {metric_name.replace('_', ' ')} metric.",
            )
            for metric_id, (_, __) in METRIC_DEFINITIONS.items()
            for metric_name in [metric_id.split(".")[-1]]
        }

    def get_compatible_metrics(
        self,
        selected_metrics: set[Metric],
        selected_dimensions: set[Dimension],
    ) -> set[Metric]:
        return self.metrics

    def get_compatible_dimensions(
        self,
        selected_metrics: set[Metric],
        selected_dimensions: set[Dimension],
    ) -> set[Dimension]:
        return self.dimensions

    def _apply_filters(
        self,
        table: pa.Table,
        filters: set[Filter] | None,
    ) -> pa.Table:
        if not filters:
            return table

        for filter_ in filters:
            if filter_.operator == Operator.ADHOC:
                continue
            if filter_.type != PredicateType.WHERE:
                continue

            col_name = DIMENSION_COLUMNS.get(filter_.column.id, filter_.column.name)
            column = table.column(col_name)
            op = filter_.operator.value
            value = filter_.value

            if op == "IS NULL":
                mask = pc.is_null(column)
            elif op == "IS NOT NULL":
                mask = pc.is_valid(column)
            elif op == "IN":
                values = list(value) if isinstance(value, (set, frozenset)) else [value]
                mask = pc.is_in(column, pa.array(values))
            elif op == "NOT IN":
                values = list(value) if isinstance(value, (set, frozenset)) else [value]
                mask = pc.invert(pc.is_in(column, pa.array(values)))
            elif op == "=":
                mask = pc.equal(column, value)
            elif op == "!=":
                mask = pc.not_equal(column, value)
            elif op == ">":
                mask = pc.greater(column, value)
            elif op == "<":
                mask = pc.less(column, value)
            elif op == ">=":
                mask = pc.greater_equal(column, value)
            elif op == "<=":
                mask = pc.less_equal(column, value)
            else:
                continue

            table = table.filter(mask)

        return table

    def get_values(
        self,
        dimension: Dimension,
        filters: set[Filter] | None = None,
    ) -> SemanticResult:
        table = self._apply_filters(self._data, filters)
        col_name = DIMENSION_COLUMNS[dimension.id]
        unique_values = pc.unique(table.column(col_name))
        result = pa.table({dimension.name: unique_values})

        return SemanticResult(
            requests=[SemanticRequest(REQUEST_TYPE, f"SELECT DISTINCT {col_name}")],
            results=result,
        )

    def get_table(self, query: SemanticQuery) -> SemanticResult:
        metrics = query.metrics
        dimensions = query.dimensions
        filters = query.filters
        order = query.order
        limit = query.limit
        offset = query.offset
        group_limit = query.group_limit

        if not metrics and not dimensions:
            return SemanticResult(requests=[], results=pa.table({}))

        table = self._apply_filters(self._data, filters)

        if group_limit and dimensions:
            table = self._apply_group_limit(table, group_limit)

        if dimensions and metrics:
            dim_cols = [DIMENSION_COLUMNS[d.id] for d in dimensions]
            aggregations = []
            for metric in metrics:
                source_col, agg_func = METRIC_DEFINITIONS[metric.id]
                aggregations.append((source_col, agg_func))
            result = table.group_by(dim_cols).aggregate(aggregations)
            # Rename columns: PyArrow group_by produces "<col>_<func>" names
            rename_map = {}
            for metric in metrics:
                source_col, agg_func = METRIC_DEFINITIONS[metric.id]
                rename_map[f"{source_col}_{agg_func}"] = metric.name
            for dim in dimensions:
                col = DIMENSION_COLUMNS[dim.id]
                if col != dim.name:
                    rename_map[col] = dim.name
            result = _rename_columns(result, rename_map)
        elif dimensions:
            dim_cols = [DIMENSION_COLUMNS[d.id] for d in dimensions]
            result = table.select(dim_cols)
            result = _unique_rows(result)
            rename_map = {
                DIMENSION_COLUMNS[d.id]: d.name
                for d in dimensions
                if DIMENSION_COLUMNS[d.id] != d.name
            }
            if rename_map:
                result = _rename_columns(result, rename_map)
        else:
            # Only metrics, aggregate entire table
            columns = {}
            for metric in metrics:
                source_col, agg_func = METRIC_DEFINITIONS[metric.id]
                column = table.column(source_col)
                if agg_func == "sum":
                    columns[metric.name] = [pc.sum(column).as_py()]
                elif agg_func == "mean":
                    columns[metric.name] = [pc.mean(column).as_py()]
                elif agg_func == "min":
                    columns[metric.name] = [pc.min(column).as_py()]
                elif agg_func == "max":
                    columns[metric.name] = [pc.max(column).as_py()]
                elif agg_func == "count":
                    columns[metric.name] = [pc.count(column).as_py()]
            result = pa.table(columns)

        # Apply ordering
        if order:
            sort_keys = []
            for element, direction in order:
                if isinstance(element, (Dimension, Metric)):
                    sort_order = "ascending" if direction == OrderDirection.ASC else "descending"
                    sort_keys.append((element.name, sort_order))
            if sort_keys:
                result = result.sort_by(sort_keys)

        # Apply offset and limit
        if offset or limit is not None:
            start = offset or 0
            length = limit if limit is not None else result.num_rows - start
            result = result.slice(start, length)

        description = self._describe_query(
            metrics,
            dimensions,
            filters,
            order,
            limit,
            offset,
        )
        return SemanticResult(
            requests=[SemanticRequest(REQUEST_TYPE, description)],
            results=result,
        )

    def get_row_count(self, query: SemanticQuery) -> SemanticResult:
        if not query.metrics and not query.dimensions:
            return SemanticResult(
                requests=[],
                results=pa.table({"COUNT": [0]}),
            )

        result = self.get_table(query)
        count = result.results.num_rows

        return SemanticResult(
            requests=result.requests,
            results=pa.table({"COUNT": [count]}),
        )

    def _apply_group_limit(
        self,
        table: pa.Table,
        group_limit: GroupLimit,
    ) -> pa.Table:
        """Filter the table to only include the top N groups."""
        limited_dim_cols = [DIMENSION_COLUMNS[d.id] for d in group_limit.dimensions]

        if group_limit.filters is not None:
            filter_table = self._apply_filters(self._data, set(group_limit.filters))
        else:
            filter_table = table

        ascending = group_limit.direction == OrderDirection.ASC
        sort_order = "ascending" if ascending else "descending"

        if group_limit.metric is not None:
            source_col, agg_func = METRIC_DEFINITIONS[group_limit.metric.id]
            group_agg = filter_table.group_by(limited_dim_cols).aggregate(
                [(source_col, agg_func)]
            )
            agg_col_name = f"{source_col}_{agg_func}"
            group_agg = group_agg.sort_by([(agg_col_name, sort_order)])
        else:
            group_agg = _unique_rows(filter_table.select(limited_dim_cols))
            group_agg = group_agg.sort_by([(limited_dim_cols[0], sort_order)])

        top_groups = group_agg.slice(0, group_limit.top).select(limited_dim_cols)

        # Semi-join: keep only rows in table whose dimension values are in top_groups
        # Use a right-join on the dimension columns then select original columns
        keys = limited_dim_cols[0] if len(limited_dim_cols) == 1 else limited_dim_cols
        return table.join(top_groups, keys=keys, join_type="inner")

    def _describe_query(
        self,
        metrics: list[Metric],
        dimensions: list[Dimension],
        filters: set[Filter] | None,
        order: list[OrderTuple] | None,
        limit: int | None,
        offset: int | None,
    ) -> str:
        parts = ["SELECT"]
        if dimensions:
            parts.append("DIMENSIONS " + ", ".join(d.name for d in dimensions))
        if metrics:
            parts.append("METRICS " + ", ".join(m.name for m in metrics))
        if filters:
            parts.append(f"FILTERS ({len(list(filters))} applied)")
        if order:
            order_parts = []
            for element, direction in order:
                if isinstance(element, (Dimension, Metric)):
                    order_parts.append(f"{element.name} {direction.value}")
            if order_parts:
                parts.append("ORDER BY " + ", ".join(order_parts))
        if limit is not None:
            parts.append(f"LIMIT {limit}")
        if offset is not None:
            parts.append(f"OFFSET {offset}")
        return " ".join(parts)

    __repr__ = uid


def _rename_columns(table: pa.Table, rename_map: dict[str, str]) -> pa.Table:
    """Rename columns in a PyArrow table."""
    new_names = [rename_map.get(name, name) for name in table.column_names]
    return table.rename_columns(new_names)


def _unique_rows(table: pa.Table) -> pa.Table:
    """Return a table with duplicate rows removed."""
    # Group by all columns with no aggregation to get distinct rows
    return table.group_by(table.column_names).aggregate([])
