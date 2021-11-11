#from src.utils.api import run_covid_api
import pandas as pd

# ====================== DEFINE FUNCTION ====================== #
def read_pickle(file_name: str) -> pd.DataFrame:
    return pd.read_pickle('Pickle/' + file_name)

# ====================== DATA.GO.KR API RUNNING ====================== #
# Korea Government API
#run_covid_api()
# ==================================================================== #

# ====================== LOAD PICKLES ====================== #
df_age_sex = read_pickle('covid_age_sex.pkl')
df_vaccines = read_pickle('covid_vaccines.pkl')
df_city = read_pickle('covid_city.pkl')
df_cases = read_pickle('covid_cases.pkl')