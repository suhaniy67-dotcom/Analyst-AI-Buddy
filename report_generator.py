import os
import pandas as pd
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak
)

from sales_prediction import (
    prepare_sales_data,
    forecast_revenue
)

from customer_segmentation import (
    create_rfm,
    perform_segmentation
)


# ============================================================
# PREPARE SALES FORECAST
# ============================================================

def get_sales_forecast(df):

    try:

        prepared = prepare_sales_data(df)

        if not prepared["success"]:
            return None

        forecast_result = forecast_revenue(
            prepared["data"]
        )

        if not forecast_result["success"]:
            return None

        return forecast_result

    except Exception:

        return None


# ============================================================
# PREPARE CUSTOMER SEGMENTATION
# ============================================================

def get_customer_segmentation(df):

    try:

        rfm_result = create_rfm(df)

        if not rfm_result["success"]:
            return None

        segmentation_result = perform_segmentation(
            rfm_result["rfm"]
        )

        if not segmentation_result["success"]:
            return None

        return segmentation_result

    except Exception:

        return None


# ============================================================
# EXCEL REPORT
# ============================================================

def generate_excel_report(
    df,
    output_path,
    ai_insights=None
):

    sales_result = get_sales_forecast(df)

    segmentation_result = get_customer_segmentation(df)

    with pd.ExcelWriter(
        output_path,
        engine="openpyxl"
    ) as writer:

        # ----------------------------------------------------
        # DATASET
        # ----------------------------------------------------

        df.to_excel(
            writer,
            sheet_name="Dataset",
            index=False
        )

        # ----------------------------------------------------
        # SUMMARY
        # ----------------------------------------------------

        summary_data = {
            "Metric": [
                "Total Rows",
                "Total Columns",
                "Missing Values",
                "Duplicate Rows"
            ],

            "Value": [
                len(df),
                len(df.columns),
                int(df.isnull().sum().sum()),
                int(df.duplicated().sum())
            ]
        }

        summary_df = pd.DataFrame(
            summary_data
        )

        summary_df.to_excel(
            writer,
            sheet_name="Summary",
            index=False
        )

        # ----------------------------------------------------
        # COLUMN INFORMATION
        # ----------------------------------------------------

        column_info = pd.DataFrame({

            "Column": df.columns,

            "Data Type": [
                str(dtype)
                for dtype in df.dtypes
            ],

            "Missing Values": [
                int(
                    df[column].isnull().sum()
                )
                for column in df.columns
            ]

        })

        column_info.to_excel(
            writer,
            sheet_name="Column Info",
            index=False
        )

        # ----------------------------------------------------
        # STATISTICS
        # ----------------------------------------------------

        numeric_df = df.select_dtypes(
            include="number"
        )

        if not numeric_df.empty:

            numeric_df.describe().T.to_excel(
                writer,
                sheet_name="Statistics"
            )

        # ----------------------------------------------------
        # SALES FORECAST
        # ----------------------------------------------------

        if sales_result is not None:

            historical = (
                sales_result["historical"]
                .copy()
            )

            historical["Type"] = "Historical"

            historical = historical.rename(
                columns={
                    "Revenue": "Revenue"
                }
            )

            forecast = (
                sales_result["forecast"]
                .copy()
            )

            forecast["Type"] = "Predicted"

            forecast = forecast.rename(
                columns={
                    "Predicted_Revenue":
                        "Revenue"
                }
            )

            sales_report = pd.concat(
                [
                    historical[
                        ["Month", "Revenue", "Type"]
                    ],

                    forecast[
                        ["Month", "Revenue", "Type"]
                    ]
                ],
                ignore_index=True
            )

            sales_report.to_excel(
                writer,
                sheet_name="Sales Forecast",
                index=False
            )

        # ----------------------------------------------------
        # CUSTOMER SEGMENTATION
        # ----------------------------------------------------

        if segmentation_result is not None:

            customers = (
                segmentation_result["customers"]
                .copy()
            )

            customers.to_excel(
                writer,
                sheet_name="Customer Segments",
                index=False
            )

            segment_summary = (
                segmentation_result["summary"]
                .copy()
            )

            segment_summary.to_excel(
                writer,
                sheet_name="Segment Summary",
                index=False
            )

        # ----------------------------------------------------
        # AI INSIGHTS
        # ----------------------------------------------------

        if ai_insights:

            insights_df = pd.DataFrame({
                "AI Insights": [
                    ai_insights
                ]
            })

            insights_df.to_excel(
                writer,
                sheet_name="AI Insights",
                index=False
            )

    return output_path


# ============================================================
# PDF REPORT
# ============================================================

def generate_pdf_report(
    df,
    output_path,
    ai_insights=None
):

    sales_result = get_sales_forecast(df)

    segmentation_result = (
        get_customer_segmentation(df)
    )

    document = SimpleDocTemplate(

        output_path,

        pagesize=A4,

        rightMargin=40,

        leftMargin=40,

        topMargin=40,

        bottomMargin=40

    )

    styles = getSampleStyleSheet()

    story = []

    # ========================================================
    # TITLE
    # ========================================================

    story.append(
        Paragraph(
            "AnalystBuddyAI",
            styles["Title"]
        )
    )

    story.append(
        Paragraph(
            "AI-Powered Data Analysis Report",
            styles["Heading2"]
        )
    )

    story.append(
        Spacer(1, 10)
    )

    story.append(
        Paragraph(
            "Generated on: "
            + datetime.now().strftime(
                "%d-%m-%Y %H:%M"
            ),
            styles["Normal"]
        )
    )

    story.append(
        Spacer(1, 20)
    )

    # ========================================================
    # 1. DATASET OVERVIEW
    # ========================================================

    story.append(
        Paragraph(
            "1. Dataset Overview",
            styles["Heading2"]
        )
    )

    story.append(
        Spacer(1, 8)
    )

    overview_data = [

        ["Metric", "Value"],

        [
            "Total Rows",
            str(len(df))
        ],

        [
            "Total Columns",
            str(len(df.columns))
        ],

        [
            "Missing Values",
            str(
                int(
                    df.isnull()
                    .sum()
                    .sum()
                )
            )
        ],

        [
            "Duplicate Rows",
            str(
                int(
                    df.duplicated()
                    .sum()
                )
            )
        ]

    ]

    overview_table = Table(
        overview_data,
        colWidths=[
            3 * inch,
            2 * inch
        ]
    )

    overview_table.setStyle(
        TableStyle([

            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.lightgrey
            ),

            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold"
            ),

            (
                "GRID",
                (0, 0),
                (-1, -1),
                1,
                colors.grey
            ),

            (
                "ALIGN",
                (1, 1),
                (-1, -1),
                "CENTER"
            )

        ])
    )

    story.append(
        overview_table
    )

    story.append(
        Spacer(1, 20)
    )

    # ========================================================
    # 2. AI INSIGHTS
    # ========================================================

    story.append(
        Paragraph(
            "2. AI-Generated Insights",
            styles["Heading2"]
        )
    )

    story.append(
        Spacer(1, 8)
    )

    if ai_insights:

        # Convert line breaks into paragraphs

        insight_lines = str(
            ai_insights
        ).split("\n")

        for line in insight_lines:

            line = line.strip()

            if line:

                story.append(
                    Paragraph(
                        line,
                        styles["BodyText"]
                    )
                )

                story.append(
                    Spacer(1, 5)
                )

    else:

        story.append(
            Paragraph(
                "AI insights were not available "
                "for this report.",
                styles["Normal"]
            )
        )

    story.append(
        Spacer(1, 15)
    )

    # ========================================================
    # 3. SALES FORECAST
    # ========================================================

    story.append(
        Paragraph(
            "3. Sales / Revenue Forecast",
            styles["Heading2"]
        )
    )

    story.append(
        Spacer(1, 8)
    )

    if sales_result is not None:

        growth = sales_result["growth"]

        last_actual = (
            sales_result["last_actual"]
        )

        next_prediction = (
            sales_result["next_prediction"]
        )

        story.append(
            Paragraph(
                f"Latest actual revenue: "
                f"{last_actual:,.2f}",
                styles["Normal"]
            )
        )

        story.append(
            Spacer(1, 5)
        )

        story.append(
            Paragraph(
                f"Next predicted revenue: "
                f"{next_prediction:,.2f}",
                styles["Normal"]
            )
        )

        story.append(
            Spacer(1, 5)
        )

        story.append(
            Paragraph(
                f"Predicted change: "
                f"{growth:.2f}%",
                styles["Normal"]
            )
        )

        story.append(
            Spacer(1, 10)
        )

        forecast_data = [
            [
                "Month",
                "Predicted Revenue"
            ]
        ]

        for _, row in (
            sales_result["forecast"]
            .iterrows()
        ):

            forecast_data.append([
                row["Month"].strftime(
                    "%b %Y"
                ),

                f"{row['Predicted_Revenue']:,.2f}"
            ])

        forecast_table = Table(
            forecast_data,
            repeatRows=1
        )

        forecast_table.setStyle(
            TableStyle([

                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.lightgrey
                ),

                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold"
                ),

                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    1,
                    colors.grey
                ),

                (
                    "ALIGN",
                    (1, 1),
                    (-1, -1),
                    "RIGHT"
                )

            ])
        )

        story.append(
            forecast_table
        )

    else:

        story.append(
            Paragraph(
                "Sales prediction could not be generated "
                "for this dataset.",
                styles["Normal"]
            )
        )

    story.append(
        Spacer(1, 20)
    )

    # ========================================================
    # 4. CUSTOMER SEGMENTATION
    # ========================================================

    story.append(
        Paragraph(
            "4. Customer Segmentation",
            styles["Heading2"]
        )
    )

    story.append(
        Spacer(1, 8)
    )

    if segmentation_result is not None:

        summary = (
            segmentation_result["summary"]
            .copy()
        )

        story.append(
            Paragraph(
                f"Customers analyzed: "
                f"{len(segmentation_result['customers'])}",
                styles["Normal"]
            )
        )

        story.append(
            Spacer(1, 10)
        )

        segment_data = [[
            "Segment",
            "Customers",
            "Avg Recency",
            "Avg Frequency",
            "Avg Monetary"
        ]]

        for _, row in summary.iterrows():

            segment_data.append([

                str(row["Segment"]),

                str(int(row["Customers"])),

                f"{row['Avg_Recency']:.2f}",

                f"{row['Avg_Frequency']:.2f}",

                f"{row['Avg_Monetary']:,.2f}"

            ])

        segment_table = Table(
            segment_data,
            repeatRows=1
        )

        segment_table.setStyle(
            TableStyle([

                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.lightgrey
                ),

                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold"
                ),

                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    1,
                    colors.grey
                ),

                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    7
                )

            ])
        )

        story.append(
            segment_table
        )

    else:

        story.append(
            Paragraph(
                "Customer segmentation could not be "
                "generated for this dataset.",
                styles["Normal"]
            )
        )

    story.append(
        Spacer(1, 20)
    )

    # ========================================================
    # 5. NUMERICAL SUMMARY
    # ========================================================

    story.append(
        Paragraph(
            "5. Numerical Summary",
            styles["Heading2"]
        )
    )

    story.append(
        Spacer(1, 8)
    )

    numeric_df = df.select_dtypes(
        include="number"
    )

    if not numeric_df.empty:

        statistics = (
            numeric_df
            .describe()
            .round(2)
        )

        stats_data = [
            ["Metric"]
            + [
                str(column)
                for column in statistics.columns
            ]
        ]

        for index in statistics.index:

            stats_data.append(
                [str(index)]
                + [
                    str(value)
                    for value
                    in statistics.loc[index]
                ]
            )

        stats_table = Table(
            stats_data,
            repeatRows=1
        )

        stats_table.setStyle(
            TableStyle([

                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.lightgrey
                ),

                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold"
                ),

                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey
                ),

                (
                    "FONTSIZE",
                    (0, 0),
                    (-1, -1),
                    6
                )

            ])
        )

        story.append(
            stats_table
        )

    else:

        story.append(
            Paragraph(
                "No numerical columns were found.",
                styles["Normal"]
            )
        )

    story.append(
        Spacer(1, 20)
    )

    # ========================================================
    # 6. CONCLUSION
    # ========================================================

    story.append(
        Paragraph(
            "6. Conclusion",
            styles["Heading2"]
        )
    )

    story.append(
        Spacer(1, 8)
    )

    conclusion = (
        "AnalystBuddyAI processed the uploaded dataset "
        "and generated an integrated analytical report "
        "covering data quality, AI insights, revenue "
        "forecasting and customer segmentation. "
        "These outputs can support data-driven business "
        "decision making."
    )

    story.append(
        Paragraph(
            conclusion,
            styles["Normal"]
        )
    )

    # ========================================================
    # BUILD PDF
    # ========================================================

    document.build(
        story
    )

    return output_path