# eda
import pandas as pd
import datetime as dt
def read_pickle(file_name: str) -> pd.DataFrame:
    return pd.read_pickle('Pickle/' + file_name)

df = read_pickle('youtube_popular.pkl')
df = df[['VideoTitle', 'ChannelTitle', 'PublishedAt', 'ViewCount', 'LikeCount', 'DislikeCount', 'CommentCount', 'Wiki_Category_1']]
df.columns = ['동영상', '채널명', '날짜', '조회수', '좋아요수', '싫어요수', '댓글수', '카테고리']

df['날짜'] = pd.to_datetime(df['날짜'], infer_datetime_format=True)
df['날짜'] = df['날짜'].dt.strftime("%Y-%m-%d")

df[['조회수', '좋아요수', '싫어요수', '댓글수']] = df[['조회수', '좋아요수', '싫어요수', '댓글수']].apply(pd.to_numeric)
#df = df.sort_values('ViewCount', ascending=False)

df['카테고리'].replace('Entertainment', '엔터테인먼트', inplace=True)
df['카테고리'].replace('Music', '뮤직', inplace=True)
df['카테고리'].replace('Hip_hop_music', '뮤직', inplace=True)
df['카테고리'].replace('Food', '푸드', inplace=True)
df['카테고리'].replace('Video_game_culture', '게임', inplace=True)
df['카테고리'].replace('Lifestyle_(sociology)', '라이프스타일', inplace=True)
df['카테고리'].replace('Association_football', '스포츠', inplace=True)

df.index = df.index+1


# Analysis
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/NGULIM.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

df_category = df['카테고리'].value_counts(normalize=True)
df_category = pd.DataFrame(df_category)

# flask
from flask import Blueprint, render_template
bp = Blueprint('ranking', __name__, url_prefix='/ranking')

@bp.route('/', methods=["GET"])
def list_index():
    labels = [i for i in df_category.index]
    values = [i for i in df_category.카테고리]
    return render_template('ranking.html', data=df, titles=df.columns.values, labels=labels, values=values)
