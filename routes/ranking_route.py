import pandas as pd
import datetime as dt

# ====================== Upload ====================== #
def read_pickle(file_name: str) -> pd.DataFrame:
    return pd.read_pickle('Pickle/' + file_name)
df = read_pickle('youtube_popular.pkl')

# ====================== EDA ====================== #
# rename cols
df = df[['VideoTitle', 'VideoId', 'ChannelTitle', 'PublishedAt', 'ViewCount', 'LikeCount', 'DislikeCount', 'CommentCount', 'Wiki_Category_1']]
df.columns = ['동영상', '동영상아이디', '채널명', '날짜', '조회수', '좋아요수', '싫어요수', '댓글수', '카테고리']
# date
df['날짜'] = pd.to_datetime(df['날짜'], infer_datetime_format=True)
df['날짜'] = df['날짜'].dt.strftime("%Y-%m-%d")
# change data types
df[['조회수', '좋아요수', '싫어요수', '댓글수']] = df[['조회수', '좋아요수', '싫어요수', '댓글수']].apply(pd.to_numeric)
# change category names
df['카테고리'].replace('Entertainment', '엔터테인먼트', inplace=True)
df['카테고리'].replace('Music', '뮤직', inplace=True)
df['카테고리'].replace('Hip_hop_music', '뮤직', inplace=True)
df['카테고리'].replace('Food', '푸드', inplace=True)
df['카테고리'].replace('Video_game_culture', '게임', inplace=True)
df['카테고리'].replace('Lifestyle_(sociology)', '라이프스타일', inplace=True)
df['카테고리'].replace('Association_football', '스포츠', inplace=True)

df.index = df.index+1

# ====================== Analysis ====================== #
# category
df_category = df['카테고리'].value_counts(normalize=True)
df_category = pd.DataFrame(df_category)
# channeltitle
df_channeltitle = df['채널명'].value_counts(normalize=True)
df_channeltitle = pd.DataFrame(df_channeltitle)

# ====================== Flask ====================== #
from flask import Blueprint, render_template
bp = Blueprint('ranking', __name__, url_prefix='/ranking')

@bp.route('/', methods=["GET"])
def list():
    video_ids = df.동영상아이디
    return render_template('ranking.html', data=df, video_ids=video_ids, titles=df.columns.values)

@bp.route('/category', methods=["GET"])
def category():
    category = [i for i in df_category.index]
    category_rate = [i for i in df_category.카테고리]
    return render_template('category.html', data=df, category=category, category_rate=category_rate)

@bp.route('/channel', methods=["GET"])
def channel():
    channeltitle = [i for i in df_channeltitle.index]
    channeltitle_rate = [i for i in df_channeltitle.채널명]
    return render_template('channel.html', data=df, channeltitle=channeltitle, channeltitle_rate=channeltitle_rate)