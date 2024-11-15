import pandas as pd
import numpy as np

def transform_variables(df: pd.DataFrame) -> pd.DataFrame:
    df["log_income"] = np.log(df["person_income"])
    df["loan_grade_num"] = df["loan_grade"].map(
        {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
    )
    df["cb_person_default_on_file_bool"] = df["cb_person_default_on_file"] == "Y"
    return df[["person_age", "log_income", "person_emp_length", "cb_person_cred_hist_length",
                "loan_amnt", "loan_percent_income", "loan_grade_num",
               "person_home_ownership", "loan_intent",
               "cb_person_default_on_file_bool"]]

