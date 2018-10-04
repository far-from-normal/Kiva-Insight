from train_classifier import train

verbose = 0
n = 100
n_jobs = 1
num_cv = 5
num_iter = 50
resampling = True

clf_type = ["lr", "sgd", "rf", "lightGBM"]
clf_type = clf_type[0]

input_data_filename = "loans_Transformed.csv"
saved_data_dir = "saved_data"

train(
    verbose,
    n,
    n_jobs,
    num_cv,
    num_iter,
    resampling,
    clf_type,
    input_data_filename,
    saved_data_dir,
)
