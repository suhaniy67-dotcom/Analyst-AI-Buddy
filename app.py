from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    jsonify,
    flash
)

import os
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
import os
import google.generativeai as genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY is not loaded. Check your .env file."
    )

genai.configure(api_key=api_key)

model = genai.GenerativeModel(
    "gemini-3.6-flash"
)

# Your chart recommendation module
from chart_recommendation import generate_chart_recommendations
from sales_prediction import (
    prepare_sales_data,
    forecast_revenue
)
from customer_segmentation import (
    create_rfm,
    perform_segmentation
)
from report_generator import (
    generate_pdf_report,
    generate_excel_report
)
# ============================================================
# FLASK APP CONFIGURATION
# ============================================================

app = Flask(__name__)

app.secret_key = os.environ.get(
    "SECRET_KEY",
    "analystbuddyai-secret-key"
)

UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {
    "csv",
    "xlsx",
    "xls"
}


# ============================================================
# GLOBAL DATAFRAME
# ============================================================

df = None


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def allowed_file(filename):

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


def load_dataset(filepath):

    extension = filepath.rsplit(".", 1)[1].lower()

    if extension == "csv":

        return pd.read_csv(filepath)

    elif extension in ["xlsx", "xls"]:

        return pd.read_excel(filepath)

    else:

        raise ValueError(
            "Unsupported file format."
        )


def clean_dataset(dataframe):

    dataframe = dataframe.copy()

    # Remove completely empty rows
    dataframe = dataframe.dropna(
        how="all"
    )

    # Remove duplicate rows
    dataframe = dataframe.drop_duplicates()

    # Strip spaces from column names
    dataframe.columns = [
        str(column).strip()
        for column in dataframe.columns
    ]

    # Try to convert object columns to dates
    for column in dataframe.columns:

        if dataframe[column].dtype == "object":

            converted = pd.to_datetime(
                dataframe[column],
                errors="coerce",
                format="mixed"
            )

            # Convert only when most non-null values
            # successfully look like dates
            if (
                converted.notna().sum() > 0
                and converted.notna().sum()
                >= 0.8 * dataframe[column].notna().sum()
            ):

                dataframe[column] = converted

    return dataframe


def get_dataset_statistics(dataframe):

    numeric_columns = dataframe.select_dtypes(
        include=[
            "int64",
            "int32",
            "float64",
            "float32"
        ]
    ).columns.tolist()

    categorical_columns = dataframe.select_dtypes(
        include=[
            "object",
            "category"
        ]
    ).columns.tolist()

    date_columns = dataframe.select_dtypes(
        include=[
            "datetime64[ns]",
            "datetime64[ns, UTC]"
        ]
    ).columns.tolist()

    return {
        "rows": len(dataframe),

        "columns": len(
            dataframe.columns
        ),

        "numeric_columns":
            numeric_columns,

        "categorical_columns":
            categorical_columns,

        "date_columns":
            date_columns,

        "missing_values":
            int(
                dataframe.isna()
                .sum()
                .sum()
            ),

        "duplicate_rows":
            int(
                dataframe.duplicated()
                .sum()
            )
    }


def create_basic_charts(dataframe):

    charts = {}

    numeric_columns = dataframe.select_dtypes(
        include=[
            "int64",
            "int32",
            "float64",
            "float32"
        ]
    ).columns.tolist()

    categorical_columns = dataframe.select_dtypes(
        include=[
            "object",
            "category"
        ]
    ).columns.tolist()


    # Remove ID-like columns
    useful_numeric = []

    for column in numeric_columns:

        name = column.lower()

        if (
            "id" not in name
            and "customer" not in name
            and "invoice" not in name
        ):

            useful_numeric.append(column)


    useful_categorical = []

    for column in categorical_columns:

        name = column.lower()

        if (
            "id" not in name
            and "customer id" not in name
            and "invoice" not in name
        ):

            useful_categorical.append(column)


    # --------------------------------------------------------
    # CATEGORY + NUMERIC
    # --------------------------------------------------------

    if useful_categorical and useful_numeric:

        category_column = useful_categorical[0]
        numeric_column = useful_numeric[0]

        temp = dataframe[
            [
                category_column,
                numeric_column
            ]
        ].dropna()

        grouped = (
            temp
            .groupby(category_column)[numeric_column]
            .sum()
            .sort_values(
                ascending=False
            )
            .head(10)
        )

        charts[
            f"{numeric_column} by {category_column}"
        ] = {

            "labels": [
                str(value)
                for value in grouped.index
            ],

            "values": [
                float(value)
                for value in grouped.values
            ]
        }


    # --------------------------------------------------------
    # NUMERIC DISTRIBUTION
    # --------------------------------------------------------

    for column in useful_numeric[:2]:

        series = dataframe[column].dropna()

        if len(series) > 0:

            # Create simple value buckets
            try:

                counts, bins = pd.cut(
                    series,
                    bins=10,
                    retbins=True
                ).value_counts(
                    sort=False
                )

                labels = [
                    str(interval)
                    for interval in counts.index
                ]

                values = [
                    int(value)
                    for value in counts.values
                ]

                charts[
                    f"{column} Distribution"
                ] = {

                    "labels": labels,

                    "values": values
                }

            except Exception:
                pass


    return charts


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():
    return render_template(
        "index.html"
    )

    if "logged_in" not in session:

        return redirect(
            url_for("login")
        )

    if df is None:
        return redirect(
        url_for("upload")
    )
    return redirect(
        url_for("dashboard")
    )

# ============================================================
# LOGIN
# ============================================================

@app.route(
    "/login",
    methods=["GET", "POST"]
)
def login():

    if request.method == "POST":

        username = request.form.get(
            "username",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        ).strip()

        # Simple project login
        # Replace with database authentication later

        if username and password:

            session["logged_in"] = True

            session["username"] = username

            return redirect(
                url_for("dashboard")
            )

        flash(
            "Please enter username and password."
        )

    return render_template(
        "login.html"
    )


# ============================================================
# LOGOUT
# ============================================================

@app.route("/logout")
def logout():

    session.clear()

    return redirect(
        url_for("login")
    )


# ============================================================
# UPLOAD PAGE
# ============================================================

@app.route(
    "/upload",
    methods=["GET", "POST"]
)
def upload():

    global df

    if "logged_in" not in session:

        return redirect(
            url_for("login")
        )


    if request.method == "POST":

        file = request.files.get(
            "file"
        )

        if file is None:

            flash(
                "Please select a file."
            )

            return redirect(
                url_for("upload")
            )


        if file.filename == "":

            flash(
                "Please select a file."
            )

            return redirect(
                url_for("upload")
            )


        if not allowed_file(
            file.filename
        ):

            flash(
                "Only CSV and Excel files are supported."
            )

            return redirect(
                url_for("upload")
            )


        filepath = os.path.join(
            app.config["UPLOAD_FOLDER"],
            file.filename
        )

        file.save(filepath)


        try:

            # Load dataset
            uploaded_df = load_dataset(
                filepath
            )

            # Clean dataset
            df = clean_dataset(
                uploaded_df
            )


            # Store filepath in session
            session["dataset_path"] = filepath

            session["dataset_name"] = (
                file.filename
            )


            flash(
                "Dataset uploaded and cleaned successfully."
            )

            return redirect(
                url_for("dashboard")
            )


        except Exception as error:

            flash(
                f"Unable to process dataset: {error}"
            )

            return redirect(
                url_for("upload")
            )


    return render_template(
        "upload.html"
    )


# ============================================================
# DASHBOARD PAGE
# ============================================================

@app.route("/dashboard")
def dashboard():

    if "logged_in" not in session:

        return redirect(
            url_for("login")
        )


    if df is None:

        return redirect(
            url_for("upload")
        )


    return render_template(
        "dashboard.html"
    )


# ============================================================
# DASHBOARD DATA API
# ============================================================

@app.route("/api/dashboard-data")
def dashboard_data():

    global df

    # Check login
    if "logged_in" not in session:

        return jsonify({
            "success": False,
            "message": "Please login first."
        }), 401


    # Check dataset
    if df is None:

        return jsonify({
            "success": False,
            "message": "No dataset uploaded."
        })


    # ----------------------------------------------------
    # BASIC DATASET STATISTICS
    # ----------------------------------------------------

    statistics = get_dataset_statistics(
        df
    )


    # ----------------------------------------------------
    # BASIC CHARTS
    # ----------------------------------------------------

    charts = create_basic_charts(
        df
    )


    # ----------------------------------------------------
    # CHART RECOMMENDATIONS
    # ----------------------------------------------------

    recommendations = generate_chart_recommendations(
        df
    )


    # ----------------------------------------------------
    # RETURN DASHBOARD DATA
    # ----------------------------------------------------

    return jsonify({

        "success": True,

        "rows":
            statistics["rows"],

        "columns":
            statistics["columns"],

        "missing_values":
            statistics["missing_values"],

        "duplicate_rows":
            statistics["duplicate_rows"],

        "numeric_columns":
            statistics["numeric_columns"],

        "categorical_columns":
            statistics["categorical_columns"],

        "date_columns":
            statistics["date_columns"],

        "charts":
            charts,

        "recommendations":
            recommendations

    })
@app.route("/api/ask", methods=["POST"])
def ask_ai():

    global df

    try:

        # Check whether dataset exists

        if df is None:

            return jsonify({
                "success": False,
                "message": "Please upload a dataset first."
            })


        # Get question from frontend

        data = request.get_json()

        question = data.get(
            "question",
            ""
        ).strip()


        if not question:

            return jsonify({
                "success": False,
                "message": "Please enter a question."
            })


        # Create a small dataset summary
        # instead of sending the entire dataset

        summary = df.head(20).to_string(
            index=False
        )


        columns = ", ".join(
            df.columns.astype(str)
        )


        prompt = f"""
You are AnalystBuddyAI, an AI data analysis assistant.

Answer the user's question using the dataset information provided below.

Dataset columns:
{columns}

Sample data:
{summary}

Dataset rows:
{len(df)}

User question:
{question}

Instructions:
- Give a clear and simple answer.
- Use the dataset information when possible.
- If the requested information cannot be determined from the provided data, say so.
- Do not invent values.
"""


        # Gemini call

        response = model.generate_content(
            prompt
        )


        answer = response.text


        return jsonify({

            "success": True,

            "answer": answer

        })


    except Exception as error:

        print(
            "AI Q&A Error:",
            error
        )


        return jsonify({

            "success": False,

            "message":
                "Unable to generate AI answer: "
                + str(error)

        })

    except Exception as error:

        return jsonify({

            "success": False,

            "message":
                f"Unable to generate dashboard data: {error}"

        }), 500

# ============================================================
# SALES / REVENUE PREDICTION API
# ============================================================

@app.route(
    "/api/sales-prediction",
    methods=["GET"]
)
def sales_prediction():

    global df

    # --------------------------------------------------------
    # LOGIN CHECK
    # --------------------------------------------------------

    if "logged_in" not in session:

        return jsonify({

            "success": False,

            "message":
                "Please login first."

        }), 401

    # --------------------------------------------------------
    # DATASET CHECK
    # --------------------------------------------------------

    if df is None:

        return jsonify({

            "success": False,

            "message":
                "Please upload a dataset first."

        })

    try:

        # ----------------------------------------------------
        # PREPARE SALES DATA
        # ----------------------------------------------------

        result = prepare_sales_data(df)

        if not result["success"]:

            return jsonify({

                "success": False,

                "message":
                    result["message"]

            })

        monthly = result["data"]

        # ----------------------------------------------------
        # FORECAST
        # ----------------------------------------------------

        forecast_result = forecast_revenue(
            monthly,
            periods=6
        )

        if not forecast_result["success"]:

            return jsonify({

                "success": False,

                "message":
                    forecast_result["message"]

            })

        historical = (
            forecast_result["historical"]
        )

        forecast = (
            forecast_result["forecast"]
        )

        # ----------------------------------------------------
        # HISTORICAL DATA
        # ----------------------------------------------------

        historical_data = []

        for _, row in historical.iterrows():

            historical_data.append({

                "month":
                    row["Month"].strftime(
                        "%Y-%m"
                    ),

                "revenue":
                    round(
                        float(
                            row["Revenue"]
                        ),
                        2
                    )

            })

        # ----------------------------------------------------
        # FORECAST DATA
        # ----------------------------------------------------

        forecast_data = []

        for _, row in forecast.iterrows():

            forecast_data.append({

                "month":
                    row["Month"].strftime(
                        "%Y-%m"
                    ),

                "predicted_revenue":
                    round(
                        float(
                            row["Predicted_Revenue"]
                        ),
                        2
                    )

            })

        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        return jsonify({

            "success": True,

            "date_column":
                result["date_column"],

            "revenue_column":
                result["revenue_column"],

            "revenue_source":
                result["revenue_source"],

            "historical":
                historical_data,

            "forecast":
                forecast_data,

            "last_actual":
                round(
                    forecast_result[
                        "last_actual"
                    ],
                    2
                ),

            "next_prediction":
                round(
                    forecast_result[
                        "next_prediction"
                    ],
                    2
                ),

            "growth":
                round(
                    forecast_result[
                        "growth"
                    ],
                    2
                )

        })

    except Exception as error:

        print(
            "Sales Prediction Error:",
            error
        )

        return jsonify({

            "success": False,

            "message":
                f"Unable to generate sales prediction: {error}"

        }), 500
# ============================================================
# SALES PREDICTION PAGE
# ============================================================

@app.route("/prediction")
def prediction():

    if "logged_in" not in session:

        return redirect(
            url_for("login")
        )

    if df is None:

        return redirect(
            url_for("upload")
        )

    return render_template(
        "prediction.html"
    )
# ============================================================
# CUSTOMER SEGMENTATION API
# ============================================================

@app.route(
    "/api/customer-segmentation",
    methods=["GET"]
)
def customer_segmentation():

    global df

    # --------------------------------------------------------
    # LOGIN CHECK
    # --------------------------------------------------------

    if "logged_in" not in session:

        return jsonify({

            "success": False,

            "message":
                "Please login first."

        }), 401


    # --------------------------------------------------------
    # DATASET CHECK
    # --------------------------------------------------------

    if df is None:

        return jsonify({

            "success": False,

            "message":
                "Please upload a dataset first."

        })


    try:

        # ----------------------------------------------------
        # CREATE RFM
        # ----------------------------------------------------

        rfm_result = create_rfm(
            df
        )


        if not rfm_result["success"]:

            return jsonify({

                "success": False,

                "message":
                    rfm_result["message"]

            })


        rfm = rfm_result["rfm"]


        # ----------------------------------------------------
        # K-MEANS
        # ----------------------------------------------------

        result = perform_segmentation(
            rfm,
            number_of_clusters=4
        )


        if not result["success"]:

            return jsonify({

                "success": False,

                "message":
                    result["message"]

            })


        customers = result[
            "customers"
        ]

        summary = result[
            "summary"
        ]


        # ----------------------------------------------------
        # CUSTOMER DATA
        # ----------------------------------------------------

        customer_data = []

        for _, row in customers.iterrows():

            customer_data.append({

                "customer_id":
                    str(
                        row["CustomerID"]
                    ),

                "recency":
                    round(
                        float(
                            row["Recency"]
                        ),
                        2
                    ),

                "frequency":
                    round(
                        float(
                            row["Frequency"]
                        ),
                        2
                    ),

                "monetary":
                    round(
                        float(
                            row["Monetary"]
                        ),
                        2
                    ),

                "cluster":
                    int(
                        row["Cluster"]
                    ),

                "segment":
                    str(
                        row["Segment"]
                    )

            })


        # ----------------------------------------------------
        # SUMMARY DATA
        # ----------------------------------------------------

        summary_data = []

        for _, row in summary.iterrows():

            summary_data.append({

                "cluster":
                    int(
                        row["Cluster"]
                    ),

                "segment":
                    str(
                        row["Segment"]
                    ),

                "customers":
                    int(
                        row["Customers"]
                    ),

                "avg_recency":
                    float(
                        row["Avg_Recency"]
                    ),

                "avg_frequency":
                    float(
                        row["Avg_Frequency"]
                    ),

                "avg_monetary":
                    float(
                        row["Avg_Monetary"]
                    ),

                "total_revenue":
                    float(
                        row["Total_Revenue"]
                    )

            })


        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        return jsonify({

            "success": True,

            "total_customers":
                len(customers),

            "clusters":
                result["clusters"],

            "customer_column":
                rfm_result[
                    "customer_column"
                ],

            "date_column":
                rfm_result[
                    "date_column"
                ],

            "quantity_column":
                rfm_result[
                    "quantity_column"
                ],

            "price_column":
                rfm_result[
                    "price_column"
                ],

            "customers":
                customer_data,

            "summary":
                summary_data

        })


    except Exception as error:

        print(
            "Customer Segmentation Error:",
            error
        )

        return jsonify({

            "success": False,

            "message":
                "Unable to perform customer segmentation: "
                + str(error)

        }), 500
# ============================================================
# CUSTOMER SEGMENTATION PAGE
# ============================================================

@app.route("/segmentation")
def segmentation():

    if "logged_in" not in session:

        return redirect(
            url_for("login")
        )


    if df is None:

        return redirect(
            url_for("upload")
        )


    return render_template(
        "segmentation.html"
    )  
# ============================================================
# PDF REPORT
# ============================================================

@app.route("/generate-pdf")
def generate_pdf():

    if "logged_in" not in session:
        return redirect(url_for("login"))

    if df is None:
        return redirect(url_for("upload"))

    report_folder = "reports"

    os.makedirs(
        report_folder,
        exist_ok=True
    )

    output_path = os.path.join(
        report_folder,
        "AnalystBuddyAI_Report.pdf"
    )

    # Get AI insights
    ai_insights = None

    try:

        prompt = f"""
You are a professional data analyst.

Analyze this dataset summary and provide
5 concise business insights.

Dataset shape:
Rows: {len(df)}
Columns: {len(df.columns)}

Columns:
{list(df.columns)}

Numerical summary:
{df.describe(include="all").to_string()}
"""

        response = model.generate_content(
            prompt
        )

        ai_insights = response.text

    except Exception as e:

        print(
            "AI report insight error:",
            e
        )

    generate_pdf_report(
        df,
        output_path,
        ai_insights
    )

    return redirect(
        url_for(
            "download_report",
            filename="AnalystBuddyAI_Report.pdf"
        )
    )
# ============================================================
# EXCEL REPORT
# ============================================================

@app.route("/generate-excel")
def generate_excel():

    if "logged_in" not in session:
        return redirect(url_for("login"))

    if df is None:
        return redirect(url_for("upload"))

    report_folder = "reports"

    os.makedirs(
        report_folder,
        exist_ok=True
    )

    output_path = os.path.join(
        report_folder,
        "AnalystBuddyAI_Report.xlsx"
    )

    # Get AI insights
    ai_insights = None

    try:

        prompt = f"""
You are a professional data analyst.

Analyze this dataset and provide
5 concise business insights.

Dataset shape:
Rows: {len(df)}
Columns: {len(df.columns)}

Columns:
{list(df.columns)}

Numerical summary:
{df.describe(include="all").to_string()}
"""

        response = model.generate_content(
            prompt
        )

        ai_insights = response.text

    except Exception as e:

        print(
            "AI Excel insight error:",
            e
        )

    generate_excel_report(
        df,
        output_path,
        ai_insights
    )

    return redirect(
        url_for(
            "download_report",
            filename="AnalystBuddyAI_Report.xlsx"
        )
    )

# ============================================================
# DOWNLOAD REPORT
# ============================================================

@app.route("/download-report/<filename>")
def download_report(filename):

    if "logged_in" not in session:
        return redirect(url_for("login"))

    from flask import send_from_directory

    return send_from_directory(
        "reports",
        filename,
        as_attachment=True
    )  
# ============================================================
# REPORTS PAGE
# ============================================================

@app.route("/reports")
def reports():

    if "logged_in" not in session:
        return redirect(url_for("login"))

    if df is None:
        return redirect(url_for("upload"))

    return render_template("reports.html")
# ============================================================
# AI ANALYSIS PAGE
# ============================================================

# ============================================================
# AI ANALYSIS PAGE
# ============================================================

@app.route("/analysis")
def analysis():

    global df

    # ----------------------------------------------------
    # CHECK LOGIN
    # ----------------------------------------------------

    if "logged_in" not in session:

        return redirect(
            url_for("login")
        )


    # ----------------------------------------------------
    # CHECK DATASET
    # ----------------------------------------------------

    if df is None:

        return redirect(
            url_for("upload")
        )


    try:

        # ------------------------------------------------
        # DATASET STATISTICS
        # ------------------------------------------------

        statistics = get_dataset_statistics(df)

        rows = statistics["rows"]

        columns = statistics["columns"]

        missing_values = statistics["missing_values"]

        duplicate_rows = statistics["duplicate_rows"]

        numeric_columns = statistics[
            "numeric_columns"
        ]

        categorical_columns = statistics[
            "categorical_columns"
        ]

        date_columns = statistics[
            "date_columns"
        ]


        # ------------------------------------------------
        # CLEANING RECOMMENDATIONS
        # ------------------------------------------------

        cleaning_messages = []


        if missing_values > 0:

            cleaning_messages.append(
                f"• {missing_values} missing values "
                "were detected."
            )

        else:

            cleaning_messages.append(
                "• No missing values were detected."
            )


        if duplicate_rows > 0:

            cleaning_messages.append(
                f"• {duplicate_rows} duplicate rows "
                "were detected."
            )

        else:

            cleaning_messages.append(
                "• No duplicate rows were detected."
            )


        if numeric_columns:

            cleaning_messages.append(
                "• Numeric columns: "
                + ", ".join(numeric_columns)
            )


        if categorical_columns:

            cleaning_messages.append(
                "• Categorical columns: "
                + ", ".join(categorical_columns)
            )


        if date_columns:

            cleaning_messages.append(
                "• Date columns: "
                + ", ".join(date_columns)
            )


        cleaning_recommendations = "\n".join(
            cleaning_messages
        )


        # ------------------------------------------------
        # CHART RECOMMENDATIONS
        # ------------------------------------------------

        try:

            chart_recommendations = (
                generate_chart_recommendations(df)
            )

        except Exception as error:

            print(
                "Chart recommendation error:",
                error
            )

            chart_recommendations = []


        # ------------------------------------------------
        # AI INSIGHTS
        # ------------------------------------------------

        ai_insights = (
            "Loading AI-generated insights..."
        )


        # ------------------------------------------------
        # FILE NAME
        # ------------------------------------------------

        filename = session.get(
            "filename",
            "Uploaded Dataset"
        )


        # ------------------------------------------------
        # RENDER PAGE
        # ------------------------------------------------

        return render_template(

            "analysis.html",

            filename=filename,

            rows=rows,

            columns=columns,

            missing_values=missing_values,

            duplicate_rows=duplicate_rows,

            cleaning_recommendations=
                cleaning_recommendations,

            ai_insights=ai_insights,

            chart_recommendations=
                chart_recommendations,

            charts=[],

            numeric_columns=
                numeric_columns,

            categorical_columns=
                categorical_columns,

            date_columns=
                date_columns
        )


    except Exception as error:

        print(
            "ANALYSIS PAGE ERROR:",
            error
        )

        return render_template(

            "analysis.html",

            filename="Uploaded Dataset",

            rows=0,

            columns=0,

            missing_values=0,

            duplicate_rows=0,

            cleaning_recommendations=
                "Unable to analyze the dataset.",

            ai_insights=
                "Unable to generate AI insights.",

            chart_recommendations=[],

            charts=[],

            numeric_columns=[],

            categorical_columns=[],

            date_columns=[]
        )

# ============================================================
# GEMINI AI INSIGHTS API
# ============================================================

@app.route(
    "/api/ai-insights",
    methods=["GET"]
)
def ai_insights():

    global df


    if "logged_in" not in session:

        return jsonify({

            "success": False,

            "message":
                "Please login first."

        }), 401


    if df is None:

        return jsonify({

            "success": False,

            "message":
                "Please upload a dataset first."

        })


    try:

        import google.generativeai as genai


        api_key = os.environ.get(
            "GEMINI_API_KEY"
        )


        if not api_key:

            return jsonify({

                "success": False,

                "message":
                    "GEMINI_API_KEY is not configured."

            })


        statistics = get_dataset_statistics(
            df
        )


        # Create a compact dataset description
        sample = df.head(10).to_string(
            index=False
        )


        prompt = f"""
You are AnalystBuddyAI, an intelligent
data analytics assistant.

Analyze the uploaded dataset.

Dataset statistics:
Rows: {statistics["rows"]}
Columns: {statistics["columns"]}
Numeric columns: {statistics["numeric_columns"]}
Categorical columns: {statistics["categorical_columns"]}
Date columns: {statistics["date_columns"]}
Missing values: {statistics["missing_values"]}
Duplicate rows: {statistics["duplicate_rows"]}

Sample data:

{sample}

Provide:

1. Important observations
2. Important trends
3. Potential business insights
4. Data quality observations
5. Useful recommendations

Keep the answer clear and concise.
Do not invent information that is not supported
by the dataset.
"""


        response = model.generate_content(
            prompt
        )


        return jsonify({

            "success": True,

            "insights":
                response.text

        })


    except Exception as error:

        return jsonify({

            "success": False,

            "message":
                f"Unable to generate AI insights: {error}"

        }), 500


# ============================================================
# AI CHART RECOMMENDATION API
# ============================================================

@app.route(
    "/api/chart-recommendations"
)
def chart_recommendations():

    global df


    if "logged_in" not in session:

        return jsonify({

            "success": False,

            "message":
                "Please login first."

        }), 401


    if df is None:

        return jsonify({

            "success": False,

            "message":
                "Please upload a dataset first."

        })


    try:

        recommendations = (
            generate_chart_recommendations(df)
        )


        return jsonify({

            "success": True,

            "recommendations":
                recommendations

        })


    except Exception as error:

        return jsonify({

            "success": False,

            "message":
                f"Unable to generate chart recommendations: {error}"

        }), 500


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    print(
        "\n=================================================="
    )

    print(
        "        AnalystBuddyAI is running"
    )

    print(
        "        http://127.0.0.1:5000/"
    )

    print(
        "==================================================\n"
    )


    # IMPORTANT:
    # debug=False prevents Flask from showing
    # multiple development URLs/reloader messages.

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )