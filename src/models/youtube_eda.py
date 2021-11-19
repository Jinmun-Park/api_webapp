from src.utils.api import api_youtube_popular
from src.utils.api import channel_search
from src.utils.api import pickle_videos, pickle_videos_filter, pickle_videos_comments
from src.utils.api import globals_videos, globals_videos_filter, globals_videos_comments

import pandas as pd
import re
#import matplotlib.pyplot as plt
#from matplotlib import font_manager, rc
#font_path = "C:/Windows/Fonts/NGULIM.TTF"
#font = font_manager.FontProperties(fname=font_path).get_name()
#rc('font', family=font)
#from wordcloud import WordCloud
#from PyKomoran import *
from collections import Counter

# ====================== DEFINE FUNCTION ====================== #
def read_pickle(file_name: str) -> pd.DataFrame:
    return pd.read_pickle('Pickle/' + file_name)

# ====================== YOUTUBE API RUNNING ====================== #
# Youtube API 1 : Popular Chart
youtube_popular = api_youtube_popular(name='youtube_popular', max_result=20)

# Youtube API 2 : Channel video comments extraction and store in pickle format
search = channel_search('슈카월드')
vid = pickle_videos(type='sample', channel_id='UCsJ6RuBiTVWRX156FVbeaGg')
vid_filter = pickle_videos_filter(type='sample', find='중국')
vid_comments = pickle_videos_comments(type='sample', option='delete')

# Youtube API 3 : Channel video comments extraction in globals() foramt
search = channel_search('슈카월드')
vid = globals_videos(type='sample', channel_id='UCsJ6RuBiTVWRX156FVbeaGg')
vid_filter = globals_videos_filter(find='코로나')
vid_comments = globals_videos_comments(option='delete')

# ================================================================= #

# ====================== MODELLING ====================== #
# 1 Youtube_popular Analysis
"""
# sort by ViewCount
youtube_popular[['ViewCount', 'LikeCount', 'DislikeCount', 'FavoriteCount', 'CommentCount']] = youtube_popular[['ViewCount', 'LikeCount', 'DislikeCount', 'FavoriteCount', 'CommentCount']].apply(pd.to_numeric)
youtube_popular=youtube_popular.sort_values('ViewCount', ascending=False)

# Category
youtube_popular_category=youtube_popular['Wiki_Category_1'].value_counts(normalize=True)
youtube_popular_category=pd.DataFrame(youtube_popular_category)
youtube_popular_category.plot(kind='bar',color=['skyblue'])

# Channel
youtube_popular_channelTitle=youtube_popular['ChannelTitle'].value_counts(normalize=True)
youtube_popular_channelTitle=pd.DataFrame(youtube_popular_channelTitle)
youtube_popular_channelTitle.plot(kind='bar', color=['skyblue'])

# word cloud (title)
# create wordcloud object
wc = WordCloud(
    background_color='white',
    font_path="C:/Windows/Fonts/NGULIM.TTF",
    max_font_size=60)

komoran = Komoran(DEFAULT_MODEL['LIGHT'])
def wordcloud(x):
    text = " ".join(x)
    text_list = komoran.get_morphes_by_tags(text, tag_list=['NNP', 'NNG', 'VV', 'VA', 'SL']) #고유명사, 일반명사, 동사, 형용사, 외국어
    counts = Counter(text_list)
    tags = counts.most_common(50)
    cloud = wc.generate_from_frequencies(dict(tags))

    plt.figure(figsize=(12, 12))
    plt.imshow(
        cloud,
        interpolation="bilinear"
    )
    plt.axis("off")
    plt.show()

print(wordcloud(youtube_popular['VideoTitle']))
"""
