# Project ideas from week 1

## Initial ideas

- Beer recommender
- MLB pitch predictor from Pitchf/x

## Context

A crowdsourced micro-loan platform/website helping impoverished individuals in developing countries to start their own businesses and dreams.

## Need

Fundraising individuals: do not have expert knowledge on getting funded.

## Vision

### Input

Analyze my fundraising at Kiva (input URL).

### Output

Your loan has a 67% chance of being funded.

These features will help you improve your campaign webpage.

## Data

Training data is downloaded from data snapshots at build.kiva.org. Daily updated and archived public data from the Kiva API into easily downloadable snapshots in .csv or .json

Or use HTML scraping on fundraising URLs to get fields that are not accessible wia the API.

Raw data table:
![data_table](raw_data.png)

Class imbalance for the funding status:
![status_class_imbalance](class_imbalance_1.png)

## Algorithms

How to predict success of a loan?

- Binary classification that can predict with a value between 0 and 1?
- Can deal with unbalanced data?
- Can return factor weight importance?

- Random forest
- Logistic regression

## For week 2

- Figure out Kiva API
- Try scraping kiva.org for fields that cannot be gotten with API
- start cleaning / scrubbing data
- Try manual feature engineering
- how to deal with missing data
- figure out SQL / csv database for cleaned data
- Get an initial ML logistic regression / random forest pipeline. Evaluate
- Get better features (NLP)
