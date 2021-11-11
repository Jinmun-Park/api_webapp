import pandas as pd
from src.utils.api import gcp_sql_connection
from src.utils.api import gcp_sql_ip_connection
import datetime as dt

# ====================== FUNCTION SETUP ====================== #
def read_pickle(file_name: str) -> pd.DataFrame:
    return pd.read_pickle('Pickle/' + file_name)

def flask_popular_chart():
    # ====================== READ FILES ====================== #
    # Read pickle files
    #df = read_pickle('youtube_popular.pkl')
    #df = gcp_sql_connection()
    df = gcp_sql_ip_connection()

    # ================== DATA PREPROCESSING ================== #
    # Rename columns (English - Korean)
    df = df[['video_title', 'video_id', 'channel_title', 'published_at', 'view_count', 'like_count', 'dislike_count', 'comment_count', 'wiki_category']]
    df.columns = ['동영상', '동영상아이디', '채널명', '날짜', '조회수', '좋아요수', '싫어요수', '댓글수', '카테고리']
    # Converting Field type : Date
    df['날짜'] = pd.to_datetime(df['날짜'], infer_datetime_format=True)
    df['날짜'] = df['날짜'].dt.strftime("%Y-%m-%d")
    # Converting Field type : Numeric
    df[['조회수', '좋아요수', '싫어요수', '댓글수']] = df[['조회수', '좋아요수', '싫어요수', '댓글수']].apply(pd.to_numeric)
    # Converting Field type : Category
    df['카테고리'].replace('Entertainment', '엔터테인먼트', inplace=True)
    df['카테고리'].replace(['Music','Hip_hop_music','Electronic_music'], '뮤직', inplace=True)
    df['카테고리'].replace('Food', '푸드', inplace=True)
    df['카테고리'].replace('Video_game_culture', '게임', inplace=True)
    df['카테고리'].replace('Lifestyle_(sociology)', '라이프스타일', inplace=True)
    df['카테고리'].replace(['Sport', 'Association_football'], '스포츠', inplace=True)
    df['카테고리'].replace('Society', '사회', inplace=True)
    df['카테고리'].replace('Health', '건강', inplace=True)
    # Counting index
    df.index = df.index+1

    # ================== FIELD ANALYSIS ================== #
    # Category
    df_category = df['카테고리'].value_counts(normalize=True)
    df_category = pd.DataFrame(df_category)
    # Channeltitle
    df_channeltitle = df['채널명'].value_counts(normalize=True)
    df_channeltitle = pd.DataFrame(df_channeltitle)

    return df, df_category, df_channeltitle

# ====================== Flask ====================== #
from flask import Blueprint, render_template
df, df_category, df_channeltitle = flask_popular_chart()
bp = Blueprint('ranking', __name__, url_prefix='/ranking')

@bp.route('/', methods=["GET"])
def list():
    video_ids = df.동영상아이디
    return render_template('ranking.html', data=df, video_ids=video_ids, titles=['동영상', '채널명', '날짜', '조회수', '좋아요수', '싫어요수', '댓글수', '카테고리'])

@bp.route('/category', methods=["GET"])
def category():
    category = [i for i in df_category.index]
    category_rate = [i for i in df_category.카테고리]
    category_channel = df.groupby(['카테고리', '채널명']).sum().sort_values('조회수', ascending=False).reset_index()
    return render_template('category.html', data=df, category=category, category_rate=category_rate, category_channel=category_channel)

@bp.route('/channel', methods=["GET"])
def channel():
    channeltitle = [i for i in df_channeltitle.index]
    channeltitle_rate = [i for i in df_channeltitle.채널명]
    channeltitle = df.groupby('채널명').sum().sort_values('조회수', ascending=False).reset_index()
    return render_template('channel.html', data=df, channeltitle=channeltitle)