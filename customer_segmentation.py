import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# ============================================================
# FIND CUSTOMER COLUMN
# ============================================================

def find_customer_column(df):

    keywords = [
        "customerid",
        "customer_id",
        "customer",
        "clientid",
        "client_id",
        "client"
    ]

    for column in df.columns:

        name = (
            str(column)
            .lower()
            .replace("_", "")
            .replace(" ", "")
            .replace("-", "")
        )

        for keyword in keywords:

            clean_keyword = keyword.replace(
                "_", ""
            )

            if clean_keyword in name:

                return column

    return None


# ============================================================
# FIND DATE COLUMN
# ============================================================

def find_date_column(df):

    keywords = [
        "date",
        "invoicedate",
        "orderdate",
        "transactiondate",
        "purchasedate"
    ]

    for column in df.columns:

        name = (
            str(column)
            .lower()
            .replace("_", "")
            .replace(" ", "")
            .replace("-", "")
        )

        for keyword in keywords:

            if keyword in name:

                converted = pd.to_datetime(
                    df[column],
                    errors="coerce",
                    format="mixed"
                )

                if converted.notna().sum() > 0:

                    return column

    # Fallback
    for column in df.columns:

        try:

            converted = pd.to_datetime(
                df[column],
                errors="coerce",
                format="mixed"
            )

            ratio = (
                converted.notna().sum()
                / max(len(df), 1)
            )

            if ratio >= 0.7:

                return column

        except Exception:
            pass

    return None


# ============================================================
# FIND QUANTITY COLUMN
# ============================================================

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


# ============================================================
# FIND PRICE COLUMN
# ============================================================

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


# ============================================================
# CREATE RFM DATA
# ============================================================

def create_rfm(df):

    dataframe = df.copy()

    customer_column = find_customer_column(
        dataframe
    )

    date_column = find_date_column(
        dataframe
    )

    quantity_column = find_quantity_column(
        dataframe
    )

    price_column = find_price_column(
        dataframe
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if customer_column is None:

        return {
            "success": False,
            "message":
                "Customer ID column could not be found."
        }


    if date_column is None:

        return {
            "success": False,
            "message":
                "Date column could not be found."
        }


    if (
        quantity_column is None
        or price_column is None
    ):

        return {
            "success": False,
            "message":
                "Quantity and Price columns are required."
        }


    # --------------------------------------------------------
    # CLEAN REQUIRED COLUMNS
    # --------------------------------------------------------

    dataframe[date_column] = pd.to_datetime(
        dataframe[date_column],
        errors="coerce",
        format="mixed"
    )

    dataframe[quantity_column] = pd.to_numeric(
        dataframe[quantity_column],
        errors="coerce"
    )

    dataframe[price_column] = pd.to_numeric(
        dataframe[price_column],
        errors="coerce"
    )


    dataframe = dataframe.dropna(
        subset=[
            customer_column,
            date_column,
            quantity_column,
            price_column
        ]
    )


    # --------------------------------------------------------
    # REMOVE INVALID VALUES
    # --------------------------------------------------------

    dataframe = dataframe[
        dataframe[quantity_column] > 0
    ]

    dataframe = dataframe[
        dataframe[price_column] >= 0
    ]


    if len(dataframe) == 0:

        return {
            "success": False,
            "message":
                "No valid transaction records found."
        }


    # --------------------------------------------------------
    # CALCULATE REVENUE
    # --------------------------------------------------------

    dataframe["Revenue"] = (
        dataframe[quantity_column]
        *
        dataframe[price_column]
    )


    # --------------------------------------------------------
    # REFERENCE DATE
    # --------------------------------------------------------

    reference_date = (
        dataframe[date_column].max()
        + pd.Timedelta(days=1)
    )


    # --------------------------------------------------------
    # RFM CALCULATION
    # --------------------------------------------------------

    rfm = (
        dataframe
        .groupby(customer_column)
        .agg(

            Recency=(
                date_column,
                lambda x:
                (
                    reference_date
                    - x.max()
                ).days
            ),

            Frequency=(
                date_column,
                "nunique"
            ),

            Monetary=(
                "Revenue",
                "sum"
            )

        )
        .reset_index()
    )


    rfm = rfm.rename(
        columns={
            customer_column:
                "CustomerID"
        }
    )


    # --------------------------------------------------------
    # REMOVE INVALID RFM VALUES
    # --------------------------------------------------------

    rfm = rfm.replace(
        [np.inf, -np.inf],
        np.nan
    )

    rfm = rfm.dropna(
        subset=[
            "Recency",
            "Frequency",
            "Monetary"
        ]
    )


    return {

        "success": True,

        "rfm": rfm,

        "customer_column":
            customer_column,

        "date_column":
            date_column,

        "quantity_column":
            quantity_column,

        "price_column":
            price_column

    }


# ============================================================
# K-MEANS SEGMENTATION
# ============================================================

def perform_segmentation(
    rfm,
    number_of_clusters=4
):

    data = rfm.copy()


    # --------------------------------------------------------
    # CHECK CUSTOMER COUNT
    # --------------------------------------------------------

    if len(data) < number_of_clusters:

        return {

            "success": False,

            "message":
                f"At least {number_of_clusters} customers are required."

        }


    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    features = [
        "Recency",
        "Frequency",
        "Monetary"
    ]


    X = data[features].copy()


    # --------------------------------------------------------
    # LOG TRANSFORMATION
    #
    # Helps reduce the effect of very large spending values.
    # --------------------------------------------------------

    X_log = np.log1p(X)


    # --------------------------------------------------------
    # STANDARDIZATION
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X_log
    )


    # --------------------------------------------------------
    # K-MEANS
    # --------------------------------------------------------

    kmeans = KMeans(
        n_clusters=number_of_clusters,
        random_state=42,
        n_init=10
    )


    clusters = kmeans.fit_predict(
        X_scaled
    )


    data["Cluster"] = clusters


    # --------------------------------------------------------
    # CALCULATE CLUSTER SUMMARY
    # --------------------------------------------------------

    summary = (
        data
        .groupby("Cluster")
        .agg(

            Customers=(
                "CustomerID",
                "count"
            ),

            Avg_Recency=(
                "Recency",
                "mean"
            ),

            Avg_Frequency=(
                "Frequency",
                "mean"
            ),

            Avg_Monetary=(
                "Monetary",
                "mean"
            ),

            Total_Revenue=(
                "Monetary",
                "sum"
            )

        )
        .reset_index()
    )


    # --------------------------------------------------------
    # ASSIGN BUSINESS SEGMENT NAMES
    # --------------------------------------------------------

    # Rank clusters by overall customer value.
    #
    # Higher frequency + monetary
    # and lower recency = better.

    summary["ValueScore"] = (
        summary["Avg_Frequency"]
        + (
            summary["Avg_Monetary"]
            /
            max(
                summary["Avg_Monetary"].mean(),
                1
            )
        )
        - (
            summary["Avg_Recency"]
            /
            max(
                summary["Avg_Recency"].mean(),
                1
            )
        )
    )


    summary = summary.sort_values(
        "ValueScore",
        ascending=False
    ).reset_index(
        drop=True
    )


    names = [
        "High Value",
        "Loyal Customers",
        "Potential Customers",
        "At Risk"
    ]


    for index, row in summary.iterrows():

        cluster = row["Cluster"]

        if index < len(names):

            name = names[index]

        else:

            name = (
                f"Customer Segment {index + 1}"
            )

        data.loc[
            data["Cluster"] == cluster,
            "Segment"
        ] = name


    # --------------------------------------------------------
    # REBUILD SUMMARY WITH SEGMENT NAME
    # --------------------------------------------------------

    summary["Segment"] = summary[
        "Cluster"
    ].map(

        data
        .drop_duplicates("Cluster")
        .set_index("Cluster")["Segment"]

    )


    # --------------------------------------------------------
    # ROUND VALUES
    # --------------------------------------------------------

    summary["Avg_Recency"] = (
        summary["Avg_Recency"]
        .round(2)
    )

    summary["Avg_Frequency"] = (
        summary["Avg_Frequency"]
        .round(2)
    )

    summary["Avg_Monetary"] = (
        summary["Avg_Monetary"]
        .round(2)
    )

    summary["Total_Revenue"] = (
        summary["Total_Revenue"]
        .round(2)
    )


    return {

        "success": True,

        "customers":
            data,

        "summary":
            summary,

        "clusters":
            number_of_clusters

    }