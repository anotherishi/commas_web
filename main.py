from flask import (
    Flask,
    request,
    render_template,
    make_response,
    redirect,
    flash,
    get_flashed_messages,
    session,
    url_for,
)
from datetime import timedelta

# timedelta(w)

from dbman import *


server = Flask(__name__)
server.secret_key = b'_5#y2L"\tnkldfkjsldnF4Q8z\n\xec]/'


@server.route("/")
def index():
    loggedin = session_id_exists(request.cookies.get("session"))
    response = make_response(render_template("index.html", loggedin=loggedin))
    return response


@server.route("/login", methods=["GET"])
def login():
    if session_id_exists(request.cookies.get("session")):
        return redirect(url_for("dashboard"))
    return render_template("login.html")


@server.route("/login", methods=["POST"])
def handle_login():
    data = request.form
    if not user_exists(data["email"]):
        return render_template("login.html", error="notexist")
    else:
        print(data, "from login url")
        if check_login_request(data):
            uid_new = init_login(data)
            response = redirect(url_for("dashboard"))
            response.set_cookie("session", uid_new, max_age=timedelta(weeks=12))
            print("redirected from login to dash", uid_new)
            return response
        else:
            return render_template("login.html", error="declined")


@server.route("/signup", methods=["GET"])
def signup():
    if session_id_exists(request.cookies.get("session")):
        return redirect(url_for("dashboard"))
    return render_template("signup.html")


@server.route("/signup", methods=["POST"])
def handle_signup():
    data = request.form
    if user_exists(data["email"]):
        return render_template("signup.html", error="exists")
    else:
        session_id = create_user(data)
        response = redirect(url_for("dashboard"))
        response.set_cookie("session", session_id, max_age=timedelta(weeks=12))
        return response


@server.route("/dashboard")
def dashboard():
    cookie = request.cookies.get("session")
    print("cookie from dash url", cookie)
    if cookie:
        if session_id_exists(cookie):
            return render_template("dashboard.html")
        else:
            flash("Session expired, please login again")
            return redirect(url_for("login"))
    else:
        flash("Please login first before opening the dashboard", "info")
        return redirect(url_for("login"))


@server.get("/get")
def get_data():
    cookie = request.cookies.get("session")
    if session_id_exists(cookie):
        data = get_all_data(cookie)
        if data:
            return json.dumps({"status": "yes"})
        else:
            return json.dumps({"status": "no"})
    else:
        return json.dumps({"status": "error"})


if __name__ == "__main__":
    server.run(debug=True, host="0.0.0.0")
