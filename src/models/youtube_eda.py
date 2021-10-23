from src.utils.api import api_youtube_popular
from src.utils.api import channel
import pandas as pd
import re
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/NGULIM.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
from wordcloud import WordCloud
from PyKomoran import *
from collections import Counter

# ====================== DEFINE FUNCTION ====================== #
def read_pickle(file_name: str) -> pd.DataFrame:
    return pd.read_pickle('Pickle/' + file_name)

# ====================== YOUTUBE API RUNNING ====================== #
# Youtuve API 1 : Popular Chart
youtube_popular = api_youtube_popular(name='youtube_popular', environment='youtube', max_result=20)

# Youtuve API 2 : Channel Search
channel = channel(cha_name='슈카월드')
channel.search()
video_info = channel.video(channel_id='UCsJ6RuBiTVWRX156FVbeaGg')
video_info_sub = channel.title_find(find='코로나')
video_comment = channel.comment()
# ================================================================= #

# ====================== MODELLING ====================== #
# 1 Youtube_popular Analysis

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




# 2 Channel Analysis