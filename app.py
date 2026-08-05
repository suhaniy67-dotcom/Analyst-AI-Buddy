from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/login")
def login():
    return render_template("login.html")

@app.route("/register")
def register():
    return render_template("register.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/upload")
def upload():
    return render_template("upload.html")

@app.route("/analysis")
def analysis():
    return render_template("analysis.html")

@app.route("/reports")
def reports():
    return render_template("reports.html")

if __name__ == "__main__":
    app.run(debug=True)