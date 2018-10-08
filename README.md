# Kiva-Insight

Users on the the crowdfunding microloan funding website Kiva.org, have a 4.6% failure rate in terms of acquiring loan funding. My aim is to help all users improve their chances of being funded, including foreign-speaking users since the loans they are seeking will significantly improve their quality of life. Most loan applicants/users are common people from developing countries or their translators. My solution is a machine learning product that predicts the probability of a crowdfunding loan being successful with logistic regression. It also gives recommendations on how to improve the structure of some the text boxes on the campaign website by ranking their logistic regression coefficients. The product is a website where users of Kiva can enter their crowdfunding URL into a text search box. After pressing on the GO! button, the users are brought to a second page where they are shown the probability of their campaign's success and are also show the top 6 suggestions on how to improve their page layout.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

#### Python and Packages

This project requires an installation of Anaconda 5.2.0, Python 3.6, with 1) the latest conda installs: numpy, pandas, sklearn, matplotlib, seaborn, and flask, and 2) the pip installs: iso-639, pycountry.

#### Data

This dataset provided by the URL (http://s3.kiva.org/snapshots/kiva_ds_csv.zip) includes a zip file. Inside, the dataset of interest is called 'loans.csv' which includes data from expired and funded Kiva campaigns. It has 34 variables, some of which are scraped text, some are numerical, some are datetime, and some are categorical. The variable names are: "LOAN_ID", "LOAN_NAME", "ORIGINAL_LANGUAGE", "DESCRIPTION", "DESCRIPTION_TRANSLATED", "FUNDED_AMOUNT", "LOAN_AMOUNT", "STATUS", "IMAGE_ID", "VIDEO_ID", "ACTIVITY_NAME", "SECTOR_NAME", "LOAN_USE", "COUNTRY_CODE", "COUNTRY_NAME", "TOWN_NAME", "CURRENCY_POLICY", "CURRENCY_EXCHANGE_COVERAGE_RATE", "CURRENCY", "PARTNER_ID", "POSTED_TIME", "PLANNED_EXPIRATION_TIME", "DISBURSE_TIME", "RAISED_TIME", "LENDER_TERM", "NUM_LENDERS_TOTAL", "NUM_JOURNAL_ENTRIES", "NUM_BULK_ENTRIES", "TAGS", "BORROWER_NAMES", "BORROWER_GENDERS", "BORROWER_PICTURED", "REPAYMENT_INTERVAL", "DISTRIBUTION_MODEL" Where "STATUS" is the label variable.

Extract 'loans.csv' and place it in the project's root directory.

[//]: # (
## Comments on the data

I chose to model the data using two different algorithms since we do not know whether the decision boundary is linear or non-linear: logistic regression and random forest. Since the dataset is imbalanced (5.15% monthly churn), I chose to train by optimizing the F1-score instead of accuracy.

To better deal with class imablance of the churn rate, for both algorithms, class weights were 'balanced'. Other hyperparamters selected with GridSearchCV. The set of hyperparamterers that maximized the F1-score was chosen as the best classifier. Training was performed with 10-fold cross validation.

Data was initially scaled using the StandardScaler for logistic regression only but for both classifiers, categorical features were one-hot encoded.

## Expected results

In the plots/ directory, the following images of confusion matrices should appear:

Logisitic regression classifier:
![img1](plots/logistic_regression_confusion_matrix.png)

Random forest classifier:
![img2](plots/random_forest_confusion_matrix.png)
)

## Authors

* **Jason Boulet** - *Initial work* - [far-from-normal](https://github.com/far-from-normal)


## License

This project is licensed under the MIT License.
