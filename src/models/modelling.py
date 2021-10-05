from src.utils.api import run_covid_api
from src.utils.api import run_youtube_chart
from src.utils.api import channel
import pandas as pd

# ====================== DEFINE FUNCTION ====================== #
def read_pickle(file_name: str) -> pd.DataFrame:
    return pd.read_pickle('Pickle/' + file_name)

# ====================== API RUNNING ====================== #
# Korea Government API
run_covid_api()
# Youtuve API 1 : Popular Chart
run_youtube_chart()
# Youtuve API 2 : Channel Search
channel = channel(cha_name='슈카월드')
channel.search()

# ====================== LOAD PICKLES ====================== #
# Korea Government Pickle Files
df_age_sex = read_pickle('covid_age_sex.pkl')
df_vaccines = read_pickle('covid_vaccines.pkl')
df_city = read_pickle('covid_city.pkl')
df_cases = read_pickle('covid_cases.pkl')
# Youtube Popular Chart Pickle Files
youtube_popular = read_pickle('youtube_popular.pkl')

# ====================== MODELLING ====================== #
