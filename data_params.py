# -*- coding: utf-8 -*-
import json
import os
from pathlib import Path
# import re
import time
from collections import Counter
import requests
from datetime import datetime
import pandas as pd

# import spacy

class Data:
    """
    blueprint data variables
    """

    def __init__(self):

        self.APP_ID = "org.insight.Kiva-Loan-Lender-Analysis"
        self.chunksize = 1000

        self.pos_label = [
            "ADJ",
            "ADP",
            "ADV",
            "CCONJ",
            "DET",
            "INTJ",
            "NOUN",
            "NUM",
            "PART",
            "PROPN",
            "PUN",
            "PRON",
            "PUNCT",
            "SPACE",
            "SYM",
            "VERB",
            "X",
        ]

        self.unprocessed_vars = [
            "LOAN_ID",
            "LOAN_NAME",
            "ORIGINAL_LANGUAGE",
            "DESCRIPTION",
            "DESCRIPTION_TRANSLATED",
            "FUNDED_AMOUNT",
            "LOAN_AMOUNT",
            "STATUS",
            "IMAGE_ID",
            "VIDEO_ID",
            "ACTIVITY_NAME",
            "SECTOR_NAME",
            "LOAN_USE",
            "COUNTRY_CODE",
            "COUNTRY_NAME",
            "TOWN_NAME",
            "CURRENCY_POLICY",
            "CURRENCY_EXCHANGE_COVERAGE_RATE",
            "CURRENCY",
            "PARTNER_ID",
            "POSTED_TIME",
            "PLANNED_EXPIRATION_TIME",
            "DISBURSE_TIME",
            "RAISED_TIME",
            "LENDER_TERM",
            "NUM_LENDERS_TOTAL",
            "NUM_JOURNAL_ENTRIES",
            "NUM_BULK_ENTRIES",
            "TAGS",
            "BORROWER_NAMES",
            "BORROWER_GENDERS",
            "BORROWER_PICTURED",
            "REPAYMENT_INTERVAL",
            "DISTRIBUTION_MODEL",
        ]

        self.cols_clean = {"in": self.unprocessed_vars, "out": self.unprocessed_vars}

        self.cols_process = [
            "STATUS",
            "DESCRIPTION",
            "DESCRIPTION_TRANSLATED",
            "LOAN_USE",
            "TAGS",
            "LOAN_AMOUNT",
            "POSTED_TIME",
            "PLANNED_EXPIRATION_TIME",
            "SECTOR_NAME",
            "COUNTRY_CODE",
            "ORIGINAL_LANGUAGE",
        ]

        self.cols_output = [
            "STATUS",
            "DESCRIPTION_TRANSLATED",
            "LOAN_USE",
            "TAGS",
            "LOAN_AMOUNT",
            "FUNDING_DAYS",
            "SECTOR_NAME",
            "COUNTRY_CODE",
            "ORIGINAL_LANGUAGE",
            "WAS_TRANSLATED",
        ]

        self.valid_status = ["expired", "funded"]

        self.dir_to_saved_data = "saved_data"
        self.dir_to_query_data = "queries"
        self.path_to_training_data = Path("loans.csv")

        self.stat_name_select = ["mean", "std", "25%", "50%", "75%", "min", "max"]
        self.stat_name_select = self.stat_name_select[0]

        self.prediction_model = ["lr", "enet", "rf"]
        self.prediction_model = self.prediction_model[1]

def format_scraped_data(json_loan_id, status, unprocessed_vars):
    from iso639 import languages

    # ORIGINAL_LANGUAGE
    english_key = "en"
    language_list = json_loan_id["description"]["languages"]
    original_language = ""
    if english_key in language_list:
        if len(language_list) == 1:
            original_language_key = english_key
        else:
            # remove 'en' from list
            original_language_key = [
                lan for lan in language_list if lan != english_key
            ][0]
        # convert to full language name to match database
        print(original_language_key)
        original_language = languages.get(alpha2=original_language_key).name

    # POSTED_TIME
    posted_time = (
        format_scraped_time(json_loan_id["posted_date"])
        if "posted_date" in json_loan_id
        else ""
    )
    # PLANNED_EXPIRATION_TIME
    planned_expiration_time = (
        format_scraped_time(json_loan_id["planned_expiration_date"])
        if "planned_expiration_date" in json_loan_id
        else ""
    )
    # DISBURSE_TIME
    disburse_time = (
        format_scraped_time(json_loan_id["disburse_date"])
        if "disburse_date" in json_loan_id
        else ""
    )
    # RAISED_TIME
    now_datetime = datetime.now().replace(microsecond=0)
    raised_time = datetime.strftime(now_datetime, "%Y-%m-%d %H:%M:%S.000 +0000")

    # TAGS
    tags = ""
    if "tags" in json_loan_id:
        for tag_dic in json_loan_id["tags"]:
            if "name" in tag_dic:
                tags += " " + tag_dic["name"]

    # 'BORROWER_NAMES', 'BORROWER_GENDERS', 'BORROWER_PICTURED'
    borrower_names = ""
    borrower_genders = ""
    borrower_pictured = ""
    if "borrowers" in json_loan_id:
        for borrowers_dic in json_loan_id["borrowers"]:
            if "first_name" in borrowers_dic:
                borrower_names += " " + borrowers_dic["first_name"]
            if "last_name" in borrowers_dic:
                borrower_names += " " + borrowers_dic["last_name"]
            if "gender" in borrowers_dic:
                if borrowers_dic["gender"] == "F":
                    borrower_genders += " female,"
                else:
                    borrower_genders += " male,"
            if "pictured" in borrowers_dic:
                borrower_pictured += " " + str(borrowers_dic["pictured"]).lower() + ","

    # add variables to dictionary
    data_formatted = {
        "LOAN_ID": json_loan_id["id"] if "id" in json_loan_id else "",
        "LOAN_NAME": json_loan_id["name"] if "name" in json_loan_id else "",
        "ORIGINAL_LANGUAGE": original_language,
        "DESCRIPTION": json_loan_id["description"]["texts"][original_language_key]
        if original_language_key in json_loan_id["description"]["texts"]
        else "",
        "DESCRIPTION_TRANSLATED": json_loan_id["description"]["texts"][english_key]
        if english_key in json_loan_id["description"]["texts"]
        else "",
        "FUNDED_AMOUNT": json_loan_id["funded_amount"]
        if "funded_amount" in json_loan_id
        else "",
        "LOAN_AMOUNT": json_loan_id["loan_amount"]
        if "loan_amount" in json_loan_id
        else "",
        "STATUS": status,
        "IMAGE_ID": json_loan_id["image"]["id"]
        if "id" in json_loan_id["image"]
        else "",
        "VIDEO_ID": "",  # video_id is not returned by website API
        "ACTIVITY_NAME": json_loan_id["activity"] if "activity" in json_loan_id else "",
        "SECTOR_NAME": json_loan_id["sector"] if "sector" in json_loan_id else "",
        "LOAN_USE": json_loan_id["use"] if "use" in json_loan_id else "",
        "COUNTRY_CODE": json_loan_id["location"]["country_code"]
        if "country_code" in json_loan_id["location"]
        else "",
        "COUNTRY_NAME": json_loan_id["location"]["country"]
        if "country" in json_loan_id["location"]
        else "",
        "TOWN_NAME": json_loan_id["location"]["town"]
        if "town" in json_loan_id["location"]
        else "",
        "CURRENCY_POLICY": json_loan_id["terms"]["loss_liability"]["currency_exchange"]
        if "currency_exchange" in json_loan_id["terms"]["loss_liability"]
        else "",
        "CURRENCY_EXCHANGE_COVERAGE_RATE": json_loan_id["terms"]["loss_liability"][
            "currency_exchange_coverage_rate"
        ]
        if "currency_exchange_coverage_rate" in json_loan_id["terms"]["loss_liability"]
        else "",
        "CURRENCY": json_loan_id["terms"]["disbursal_currency"]
        if "disbursal_currency" in json_loan_id["terms"]
        else "",
        "PARTNER_ID": json_loan_id["partner_id"]
        if "partner_id" in json_loan_id
        else "",
        "POSTED_TIME": posted_time,
        "PLANNED_EXPIRATION_TIME": planned_expiration_time,
        "DISBURSE_TIME": disburse_time,
        "RAISED_TIME": raised_time,
        "LENDER_TERM": json_loan_id["terms"]["repayment_term"]
        if "repayment_term" in json_loan_id["terms"]
        else "",
        "NUM_LENDERS_TOTAL": json_loan_id["lender_count"]
        if "lender_count" in json_loan_id
        else "",
        "NUM_JOURNAL_ENTRIES": json_loan_id["journal_totals"]["entries"]
        if "entries" in json_loan_id["journal_totals"]
        else "",
        "NUM_BULK_ENTRIES": json_loan_id["journal_totals"]["bulkEntries"]
        if "bulkEntries" in json_loan_id["journal_totals"]
        else "",
        "TAGS": tags,
        "BORROWER_NAMES": borrower_names,
        "BORROWER_GENDERS": borrower_genders,
        "BORROWER_PICTURED": borrower_pictured,
        "REPAYMENT_INTERVAL": json_loan_id["terms"]["repayment_interval"].lower()
        if "repayment_interval" in json_loan_id["terms"]
        else "",
        "DISTRIBUTION_MODEL": "",  # distribution_model is not returned by website API
    }

    df = pd.DataFrame(data_formatted, index=[0])
    df = df[unprocessed_vars]

    return data_formatted, df


def scrape_loan_data(loan_id, app_id):

    # The basic Kiva API query to get the list of lenders
    # for a particular loan, as selected by the loan ID.
    #    url = 'https://api.kivaws.org/v1/loans/%s' % (loan_id)
    url_loan_id = "https://api.kivaws.org/v1/loans/%s.json&app_id=%s" % (
        loan_id,
        app_id,
    )
    # print(loan_id)
    # print(url_loan_id)

    r = requests.get(url_loan_id, verify=True)
    if r.status_code == 200:
        # print(str(r.content, "utf-8"))
        json_data = json.loads(str(r.content, "utf-8"))
        json_loan_id = json_data["loans"][0]
        status = json_loan_id["status"]
        request_sucess = True
    else:
        json_loan_id = {}
        status = None
        request_sucess = True
        print("bad request")

    return json_loan_id, status, request_sucess


def format_scraped_time(datetime_str):
    datetime_datetime_formatted = datetime.strptime(datetime_str, "%Y-%m-%dT%H:%M:%SZ")
    datetime_str_formatted = datetime.strftime(
        datetime_datetime_formatted, "%Y-%m-%d %H:%M:%S.000 +0000"
    )
    return datetime_str_formatted
