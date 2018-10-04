from predict_v2 import scrape_loan_id, predict_prob, plot_factors


def predict_flask(loan_url_):

    from urllib.parse import urlparse

    loan_id = urlparse(loan_url_).path.split("/")[-1]

    request_sucess, status = scrape_loan_id(loan_id)
    if status == "fundraising":
        prob = predict_prob()[0][1]
        success_percent = "{0:.0f}".format(100 * prob) + " percent"
        print(success_percent)
        (
            img_name_1,
            img_name_2,
            img_name_3,
            desc_text_words,
            desc_text_sentences,
            desc_text_paragraphs,
            loanuse_text_words,
            tags_text_words,
            tags_text_hashtags,
        ) = plot_factors(loan_id)
    else:
        success_percent = status
        img_name_1, img_name_2, img_name_3 = (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    return (
        success_percent,
        img_name_1,
        img_name_2,
        img_name_3,
        desc_text_words,
        desc_text_sentences,
        desc_text_paragraphs,
        loanuse_text_words,
        tags_text_words,
        tags_text_hashtags,
    )
