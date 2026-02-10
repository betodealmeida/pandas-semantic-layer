from __future__ import annotations

from pandas import DataFrame
from superset_core.semantic_layers.semantic_view import (
    SemanticView,
    SemanticViewFeature,
)
from superset_core.semantic_layers.types import (
    DATE,
    DECIMAL,
    Dimension,
    Filter,
    GroupLimit,
    INTEGER,
    Metric,
    Operator,
    OrderDirection,
    OrderTuple,
    PredicateType,
    SemanticRequest,
    SemanticResult,
    STRING,
)

from pandas_semantic_layer.data import (
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
        self._data = SALES_DATA.copy()
        self.dimensions = self.get_dimensions()
        self.metrics = self.get_metrics()

    def uid(self) -> str:
        return f"pandas.{self.name}"

    def get_dimensions(self) -> set[Dimension]:
        type_map = {
            "product_category": STRING,
            "region": STRING,
            "sale_date": DATE,
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
            "total_revenue": DECIMAL,
            "total_units_sold": INTEGER,
            "average_price": DECIMAL,
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

    def _apply_filters(
        self,
        df: DataFrame,
        filters: set[Filter] | None,
    ) -> DataFrame:
        if not filters:
            return df

        for filter_ in filters:
            if filter_.operator == Operator.ADHOC:
                continue
            if filter_.type != PredicateType.WHERE:
                continue

            col_name = DIMENSION_COLUMNS.get(filter_.column.id, filter_.column.name)
            op = filter_.operator.value
            value = filter_.value

            if op == "IS NULL":
                df = df[df[col_name].isna()]
            elif op == "IS NOT NULL":
                df = df[df[col_name].notna()]
            elif op == "IN":
                values = list(value) if isinstance(value, (set, frozenset)) else [value]
                df = df[df[col_name].isin(values)]
            elif op == "NOT IN":
                values = list(value) if isinstance(value, (set, frozenset)) else [value]
                df = df[~df[col_name].isin(values)]
            elif op == "=":
                df = df[df[col_name] == value]
            elif op == "!=":
                df = df[df[col_name] != value]
            elif op == ">":
                df = df[df[col_name] > value]
            elif op == "<":
                df = df[df[col_name] < value]
            elif op == ">=":
                df = df[df[col_name] >= value]
            elif op == "<=":
                df = df[df[col_name] <= value]

        return df

    def get_values(
        self,
        dimension: Dimension,
        filters: set[Filter] | None = None,
    ) -> SemanticResult:
        df = self._apply_filters(self._data.copy(), filters)
        col_name = DIMENSION_COLUMNS[dimension.id]
        result = DataFrame(df[col_name].unique(), columns=[dimension.name])

        return SemanticResult(
            requests=[SemanticRequest(REQUEST_TYPE, f"SELECT DISTINCT {col_name}")],
            results=result,
        )

    def get_dataframe(
        self,
        metrics: list[Metric],
        dimensions: list[Dimension],
        filters: set[Filter] | None = None,
        order: list[OrderTuple] | None = None,
        limit: int | None = None,
        offset: int | None = None,
        *,
        group_limit: GroupLimit | None = None,
    ) -> SemanticResult:
        if not metrics and not dimensions:
            return SemanticResult(requests=[], results=DataFrame())

        df = self._apply_filters(self._data.copy(), filters)

        if group_limit and dimensions:
            df = self._apply_group_limit(df, group_limit)

        if dimensions and metrics:
            dim_cols = [DIMENSION_COLUMNS[d.id] for d in dimensions]
            agg_dict = {}
            for metric in metrics:
                source_col, agg_func = METRIC_DEFINITIONS[metric.id]
                agg_dict[source_col] = agg_func
            result = df.groupby(dim_cols, as_index=False).agg(agg_dict)
            # Rename columns: source columns -> metric/dimension names
            rename_map = {}
            for metric in metrics:
                source_col, _ = METRIC_DEFINITIONS[metric.id]
                rename_map[source_col] = metric.name
            for dim in dimensions:
                col = DIMENSION_COLUMNS[dim.id]
                if col != dim.name:
                    rename_map[col] = dim.name
            result.rename(columns=rename_map, inplace=True)
        elif dimensions:
            dim_cols = [DIMENSION_COLUMNS[d.id] for d in dimensions]
            result = df[dim_cols].drop_duplicates()
            rename_map = {
                DIMENSION_COLUMNS[d.id]: d.name
                for d in dimensions
                if DIMENSION_COLUMNS[d.id] != d.name
            }
            if rename_map:
                result.rename(columns=rename_map, inplace=True)
        else:
            # Only metrics, aggregate entire DataFrame
            agg_results = {}
            for metric in metrics:
                source_col, agg_func = METRIC_DEFINITIONS[metric.id]
                agg_results[metric.name] = [getattr(df[source_col], agg_func)()]
            result = DataFrame(agg_results)

        # Apply ordering
        if order:
            sort_cols = []
            ascending = []
            for element, direction in order:
                if isinstance(element, (Dimension, Metric)):
                    sort_cols.append(element.name)
                    ascending.append(direction == OrderDirection.ASC)
            if sort_cols:
                result = result.sort_values(sort_cols, ascending=ascending)

        # Apply offset and limit
        if offset:
            result = result.iloc[offset:]
        if limit is not None:
            result = result.iloc[:limit]

        result = result.reset_index(drop=True)

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

    def get_row_count(
        self,
        metrics: list[Metric],
        dimensions: list[Dimension],
        filters: set[Filter] | None = None,
        order: list[OrderTuple] | None = None,
        limit: int | None = None,
        offset: int | None = None,
        *,
        group_limit: GroupLimit | None = None,
    ) -> SemanticResult:
        if not metrics and not dimensions:
            return SemanticResult(
                requests=[],
                results=DataFrame([[0]], columns=["COUNT"]),
            )

        result = self.get_dataframe(
            metrics,
            dimensions,
            filters,
            order,
            limit,
            offset,
            group_limit=group_limit,
        )
        count = len(result.results)

        return SemanticResult(
            requests=result.requests,
            results=count,
        )

    def _apply_group_limit(
        self,
        df: DataFrame,
        group_limit: GroupLimit,
    ) -> DataFrame:
        """Filter the DataFrame to only include the top N groups."""
        limited_dim_cols = [DIMENSION_COLUMNS[d.id] for d in group_limit.dimensions]

        if group_limit.filters is not None:
            filter_df = self._apply_filters(self._data.copy(), set(group_limit.filters))
        else:
            filter_df = df

        ascending = group_limit.direction == OrderDirection.ASC
        if group_limit.metric is not None:
            source_col, agg_func = METRIC_DEFINITIONS[group_limit.metric.id]
            group_agg = filter_df.groupby(limited_dim_cols, as_index=False).agg(
                {source_col: agg_func}
            )
            group_agg = group_agg.sort_values(source_col, ascending=ascending)
        else:
            group_agg = filter_df[limited_dim_cols].drop_duplicates()
            group_agg = group_agg.sort_values(limited_dim_cols[0], ascending=ascending)

        top_groups = group_agg.head(group_limit.top)[limited_dim_cols]
        return df.merge(top_groups, on=limited_dim_cols, how="inner")

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
