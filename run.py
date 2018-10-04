#!/usr/bin/env python
from flask import render_template, make_response
from flaskexample import app
from functools import wraps, update_wrapper


app.run(host="0.0.0.0", debug=True)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0


@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("loans.html")
