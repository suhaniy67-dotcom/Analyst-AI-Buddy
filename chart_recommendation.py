import pandas as pd


def generate_chart_recommendations(df):

    recommendations = []

    # =====================================================
    # DETECT COLUMN TYPES
    # =====================================================

    numerical_columns = df.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns.tolist()

    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()


    # =====================================================
    # SMART DATE DETECTION
    # =====================================================

    date_columns = []

    # First priority: columns whose names clearly indicate dates
    date_keywords = [
        "date",
        "time",
        "timestamp",
        "datetime"
    ]

    for column in df.columns:

        column_name = str(column).lower().strip()

        # IMPORTANT:
        # Invoice is NOT a date
        if column_name in [
            "invoice",
            "invoice no",
            "invoice number",
            "invoiceno",
            "customerid",
            "customer id"
        ]:
            continue

        if any(
            keyword in column_name
            for keyword in date_keywords
        ):

            try:

                converted = pd.to_datetime(
                    df[column],
                    errors="coerce"
                )

                valid_ratio = converted.notna().mean()

                if valid_ratio >= 0.50:

                    date_columns.append(column)

            except Exception:

                pass


    # =====================================================
    # FALLBACK DATE DETECTION
    # =====================================================

    if not date_columns:

        for column in df.columns:

            column_name = str(column).lower().strip()

            # Never treat IDs/invoices as dates
            if (
                "invoice" in column_name
                or "customer" in column_name
                or "id" in column_name
            ):
                continue

            try:

                converted = pd.to_datetime(
                    df[column],
                    errors="coerce"
                )

                valid_ratio = converted.notna().mean()

                if valid_ratio >= 0.80:

                    # Avoid numeric columns being interpreted
                    # as timestamps such as 1970-01-01
                    if not pd.api.types.is_numeric_dtype(
                        df[column]
                    ):

                        date_columns.append(column)

            except Exception:

                pass


    # =====================================================
    # REMOVE ID-LIKE COLUMNS
    # =====================================================

    id_columns = []

    for column in numerical_columns + categorical_columns:

        column_name = str(column).lower().strip()

        if (
            "id" in column_name
            or "invoice" in column_name
            or "customer_id" in column_name
            or "customer id" in column_name
        ):

            id_columns.append(column)


    numerical_columns = [
        column
        for column in numerical_columns
        if column not in id_columns
    ]

    categorical_columns = [
        column
        for column in categorical_columns
        if column not in id_columns
    ]


    # =====================================================
    # 1. DATE + NUMERICAL → LINE CHART
    # =====================================================

    if date_columns and numerical_columns:

        date_column = date_columns[0]
        numerical_column = numerical_columns[0]

        try:

            temp_df = df[
                [date_column, numerical_column]
            ].copy()

            temp_df[date_column] = pd.to_datetime(
                temp_df[date_column],
                errors="coerce"
            )

            temp_df[numerical_column] = pd.to_numeric(
                temp_df[numerical_column],
                errors="coerce"
            )

            temp_df = temp_df.dropna(
                subset=[
                    date_column,
                    numerical_column
                ]
            )

            # Group by month for large datasets
            grouped = (
                temp_df
                .set_index(date_column)
                .resample("ME")[numerical_column]
                .sum()
                .dropna()
            )

            labels = [
                date.strftime("%Y-%m")
                for date in grouped.index
            ]

            values = [
                float(value)
                for value in grouped.values
            ]

            recommendations.append({

                "chart_type": "line",

                "x_column": date_column,

                "y_column": numerical_column,

                "title":
                    f"{numerical_column} Over Time",

                "labels": labels,

                "values": values,

                "reason":
                    "A line chart is suitable for showing "
                    "how a numerical value changes over time."

            })

        except Exception as e:

            print(
                "Date chart recommendation error:",
                e
            )


    # =====================================================
    # 2. CATEGORICAL + NUMERICAL → BAR CHART
    # =====================================================

    if categorical_columns and numerical_columns:

        categorical_column = categorical_columns[0]
        numerical_column = numerical_columns[0]

        try:

            temp_df = df[
                [categorical_column, numerical_column]
            ].copy()

            temp_df[numerical_column] = pd.to_numeric(
                temp_df[numerical_column],
                errors="coerce"
            )

            temp_df = temp_df.dropna(
                subset=[
                    categorical_column,
                    numerical_column
                ]
            )

            grouped = (
                temp_df
                .groupby(
                    categorical_column,
                    observed=True
                )[numerical_column]
                .sum()
                .sort_values(
                    ascending=False
                )
                .head(10)
            )

            labels = [
                str(value)
                for value in grouped.index
            ]

            values = [
                float(value)
                for value in grouped.values
            ]

            recommendations.append({

                "chart_type": "bar",

                "x_column": categorical_column,

                "y_column": numerical_column,

                "title":
                    f"Top {categorical_column} by {numerical_column}",

                "labels": labels,

                "values": values,

                "reason":
                    "A bar chart is suitable for comparing "
                    "a numerical value across categories."

            })

        except Exception as e:

            print(
                "Categorical chart recommendation error:",
                e
            )


    # =====================================================
    # 3. TWO NUMERICAL COLUMNS → SCATTER PLOT
    # =====================================================

    if len(numerical_columns) >= 2:

        x_column = numerical_columns[0]
        y_column = numerical_columns[1]

        try:

            temp_df = df[
                [x_column, y_column]
            ].copy()

            temp_df[x_column] = pd.to_numeric(
                temp_df[x_column],
                errors="coerce"
            )

            temp_df[y_column] = pd.to_numeric(
                temp_df[y_column],
                errors="coerce"
            )

            temp_df = temp_df.dropna()

            # Limit points for dashboard performance
            temp_df = temp_df.head(1000)

            labels = [
                float(value)
                for value in temp_df[x_column]
            ]

            values = [
                float(value)
                for value in temp_df[y_column]
            ]

            recommendations.append({

                "chart_type": "scatter",

                "x_column": x_column,

                "y_column": y_column,

                "title":
                    f"{y_column} vs {x_column}",

                "labels": labels,

                "values": values,

                "reason":
                    "A scatter plot is suitable for "
                    "examining the relationship between "
                    "two numerical variables."

            })

        except Exception as e:

            print(
                "Scatter chart recommendation error:",
                e
            )


    # =====================================================
    # RETURN
    # =====================================================

    return recommendations