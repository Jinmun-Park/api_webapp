from src.utils.api import api_youtube_popular
from src.utils.api import channel
import pandas as pd

# ====================== DEFINE FUNCTION ====================== #
def read_pickle(file_name: str) -> pd.DataFrame:
    return pd.read_pickle('Pickle/' + file_name)

# ====================== YOUTUBE API RUNNING ====================== #
# Youtuve API 1 : Popular Chart
youtube_popular = api_youtube_popular(name='youtube_popular', environment='youtube', max_result=20)

# Youtuve API 2 : Channel Search
channel = channel(cha_name='슈카월드')
channel.search()
video_information = channel.video(channel_id='UCsJ6RuBiTVWRX156FVbeaGg')
video_information_sub = channel.title_find(find='코로나')
# ================================================================= #

# ====================== MODELLING ====================== #
