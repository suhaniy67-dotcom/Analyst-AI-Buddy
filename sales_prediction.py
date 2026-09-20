import pandas as pd
import numpy as np


def find_date_column(df):
    """
    Automatically find the most likely date column.
    """

    # First check column names
    date_keywords = [
        "date",
        "orderdate",
        "invoicedate",
        "transactiondate",
        "purchasedate",
        "timestamp"
    ]

    for column in df.columns:

        column_name = (
            str(column)
            .lower()
            .replace("_", "")
            .replace(" ", "")
            .replace("-", "")
        )

        for keyword in date_keywords:

            if keyword in column_name:

                converted = pd.to_datetime(
                    df[column],
                    errors="coerce",
                    format="mixed"
                )

                if converted.notna().sum() > 0:

                    return column

    # If no date name was found, try object/date columns
    for column in df.columns:

        try:

            converted = pd.to_datetime(
                df[column],
                errors="coerce",
                format="mixed"
            )

            valid_ratio = (
                converted.notna().sum()
                / max(len(df), 1)
            )

            if valid_ratio >= 0.7:

                return column

        except Exception:
            pass

    return None


def find_revenue_column(df):
    """
    Find an existing revenue/sales column.
    """

    revenue_keywords = [
        "revenue",
        "sales",
        "amount",
        "totalamount",
        "totalprice",
        "netamount",
        "turnover"
    ]

    for column in df.columns:

        column_name = (
            str(column)
            .lower()
            .replace("_", "")
            .replace(" ", "")
            .replace("-", "")
        )

        for keyword in revenue_keywords:

            if keyword in column_name:

                if pd.api.types.is_numeric_dtype(
                    df[column]
                ):

                    return column

    return None


def find_quantity_column(df):

    keywords = [
        "quantity",
        "qty",
        "units",
        "unitssold"
    ]

    for column in df.columns:

        name = (
            str(column)
            .lower()
            .replace("_", "")
            .replace(" ", "")
        )

        for keyword in keywords:

            if keyword in name:

                if pd.api.types.is_numeric_dtype(
                    df[column]
                ):

                    return column

    return None


def find_price_column(df):

    keywords = [
        "unitprice",
        "price",
        "sellingprice",
        "saleprice"
    ]

    for column in df.columns:

        name = (
            str(column)
            .lower()
            .replace("_", "")
            .replace(" ", "")
        )

        for keyword in keywords:

            if keyword in name:

                if pd.api.types.is_numeric_dtype(
                    df[column]
                ):

                    return column

    return None


def prepare_sales_data(df):

    dataframe = df.copy()

    date_column = find_date_column(
        dataframe
    )

    if date_column is None:

        return {
            "success": False,
            "message":
                "No suitable date column found."
        }

    dataframe[date_column] = pd.to_datetime(
        dataframe[date_column],
        errors="coerce",
        format="mixed"
    )

    dataframe = dataframe.dropna(
        subset=[date_column]
    )

    # --------------------------------------------------
    # FIND REVENUE
    # --------------------------------------------------

    revenue_column = find_revenue_column(
        dataframe
    )

    revenue_source = "existing"

    # If revenue does not exist,
    # calculate Quantity × UnitPrice
    if revenue_column is None:

        quantity_column = find_quantity_column(
            dataframe
        )

        price_column = find_price_column(
            dataframe
        )

        if (
            quantity_column is None
            or price_column is None
        ):

            return {
                "success": False,
                "message":
                    "Could not find Revenue/Sales or Quantity and Price columns."
            }

        dataframe["Calculated_Revenue"] = (
            pd.to_numeric(
                dataframe[quantity_column],
                errors="coerce"
            )
            *
            pd.to_numeric(
                dataframe[price_column],
                errors="coerce"
            )
        )

        revenue_column = "Calculated_Revenue"

        revenue_source = (
            f"{quantity_column} × {price_column}"
        )

    # --------------------------------------------------
    # CLEAN REVENUE
    # --------------------------------------------------

    dataframe[revenue_column] = pd.to_numeric(
        dataframe[revenue_column],
        errors="coerce"
    )

    dataframe = dataframe.dropna(
        subset=[revenue_column]
    )

    # --------------------------------------------------
    # MONTHLY REVENUE
    # --------------------------------------------------

    dataframe["Month"] = (
        dataframe[date_column]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    monthly = (
        dataframe
        .groupby("Month")[revenue_column]
        .sum()
        .reset_index()
        .sort_values("Month")
    )

    monthly.columns = [
        "Month",
        "Revenue"
    ]

    return {
        "success": True,
        "data": monthly,
        "date_column": date_column,
        "revenue_column": revenue_column,
        "revenue_source": revenue_source
    }


def forecast_revenue(monthly_data, periods=6):

    data = monthly_data.copy()

    if len(data) < 3:

        return {
            "success": False,
            "message":
                "At least 3 months of historical data are required for prediction."
        }

    data = data.sort_values(
        "Month"
    ).reset_index(drop=True)

    # --------------------------------------------------
    # SIMPLE LINEAR REGRESSION
    # --------------------------------------------------

    x = np.arange(
        len(data)
    )

    y = data["Revenue"].values

    # Fit linear trend
    coefficients = np.polyfit(
        x,
        y,
        1
    )

    slope = coefficients[0]

    intercept = coefficients[1]

    # Future positions
    future_x = np.arange(
        len(data),
        len(data) + periods
    )

    predicted = (
        slope * future_x
        + intercept
    )

    # Do not return negative revenue
    predicted = np.maximum(
        predicted,
        0
    )

    last_month = data["Month"].max()

    future_months = pd.date_range(
        start=last_month
        + pd.DateOffset(months=1),
        periods=periods,
        freq="MS"
    )

    forecast = pd.DataFrame({

        "Month":
            future_months,

        "Predicted_Revenue":
            predicted

    })

    # --------------------------------------------------
    # GROWTH
    # --------------------------------------------------

    last_actual = float(
        data["Revenue"].iloc[-1]
    )

    first_prediction = float(
        forecast[
            "Predicted_Revenue"
        ].iloc[0]
    )

    if last_actual != 0:

        growth = (
            (
                first_prediction
                - last_actual
            )
            / abs(last_actual)
        ) * 100

    else:

        growth = 0

    return {

        "success": True,

        "historical":
            data,

        "forecast":
            forecast,

        "growth":
            float(growth),

        "last_actual":
            last_actual,

        "next_prediction":
            first_prediction

    }