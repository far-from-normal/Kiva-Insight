from flask import render_template, request, session, make_response
from flaskexample import app

from predict_flask_week3 import predict_flask

app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0


@app.after_request
def add_header(response):
    response.headers["X-UA-Compatible"] = "IE=Edge,chrome=1"
    if "Cache-Control" not in response.headers:
        response.headers["Cache-Control"] = "public, max-age=600"
    return response


@app.route("/")
@app.route("/input")
def loans_input():
    return render_template("input.html")


@app.route("/output")
def loans_output():
    # pull 'loan_url' from input field and store it
    loan_url_ = request.args.get("loan_url")
    print(loan_url_)
    # the_result = str(93)
    (
        the_result,
        img_name_1,
        img_name_2,
        img_name_3,
        desc_text_words,
        desc_text_sentences,
        desc_text_paragraphs,
        loanuse_text_words,
        tags_text_words,
        tags_text_hashtags,
    ) = predict_flask(loan_url_)
    print(the_result)

    return render_template(
        "output.html",
        loan_url_=loan_url_,
        the_result=the_result,
        img_name_1=img_name_1,
        img_name_2=img_name_2,
        img_name_3=img_name_3,
        desc_text_words=desc_text_words,
        desc_text_sentences=desc_text_sentences,
        desc_text_paragraphs=desc_text_paragraphs,
        loanuse_text_words=loanuse_text_words,
        tags_text_words=tags_text_words,
        tags_text_hashtags=tags_text_hashtags,
    )
