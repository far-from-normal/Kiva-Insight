import os
import data_params

# %%

data_par = data_params.Data()
unprocessed_vars = data_par.unprocessed_vars
app_id = data_par.APP_ID

scraped_file_name = "loans_Scraped.csv"
write_mode = "a+"

# delete old clean file
try:
    os.remove(scraped_file_name)
except OSError:
    pass

num_requests = 50000
start = 1606000
finish = start - num_requests

for loan_id in range(start, finish, -1):
    print("########################3")
    loan_id_str = str(loan_id)
    #    success, status, data_formatted, df_scraped = scrape_loan_data(loan_id_str, APP_ID, loan_info)
    data_scraped, status, request_sucess = data_params.scrape_loan_data(
        loan_id_str, app_id
    )

    if request_sucess and status == "fundraising":
        data_formatted, df_formatted = data_params.format_scraped_data(
            data_scraped, status, unprocessed_vars
        )
        print(df_formatted)
        if write_mode != "a+":
            df_formatted.to_csv(scraped_file_name, index=False, encoding="utf-8")
        else:
            if not os.path.isfile(scraped_file_name):
                df_formatted.to_csv(
                    scraped_file_name,
                    header=unprocessed_vars,
                    index=False,
                    encoding="utf-8",
                )
            else:  # else it exists so append without writing the header
                df_formatted.to_csv(
                    scraped_file_name,
                    mode=write_mode,
                    header=False,
                    index=False,
                    encoding="utf-8",
                )


#    for key, value in data_formatted.items():
#        print(key, value)
