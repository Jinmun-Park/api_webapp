import pandas as pd
from src.utils.api import read_pickle, gcp_sql_pull
import datetime as dt


# ====================== FUNCTION SETUP ====================== #
def flask_popular_chart():
    # ====================== READ FILES ====================== #
    # Read pickle files
    #df = read_pickle('youtube_popular.pkl')
    df = gcp_sql_pull()
    #df = gcp_sql_ip_connection()

    # ================== DATA PREPROCESSING ================== #
    # Select columns
    df = df[['video_title', 'video_id', 'channel_title', 'published_at', 'view_count', 'like_count', 'dislike_count', 'comment_count', 'wiki_category']]
    # Rename columns (English - Korean)
    df.rename(columns={'video_title': '동영상',
                       'video_id': '동영상아이디',
                       'channel_title': '채널명',
                       'published_at': '날짜',
                       'view_count': '조회수',
                       'like_count': '좋아요수',
                       'dislike_count': '싫어요수',
                       'comment_count': '댓글수',
                       'wiki_category': '카테고리'
                       }, inplace=True)
    # Reset index
    df = df.reset_index(drop=True)
    # Converting field type : Date
    df['날짜'] = pd.to_datetime(df['날짜'], infer_datetime_format=True)
    df['날짜'] = df['날짜'].dt.strftime("%Y-%m-%d")
    # Converting field type : Numeric
    df[['조회수', '좋아요수', '싫어요수', '댓글수']] = df[['조회수', '좋아요수', '싫어요수', '댓글수']].apply(pd.to_numeric)
    # Converting field type : Category
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
bp = Blueprint('chart', __name__, url_prefix='/chart')

@bp.route('/', methods=["GET"])
def chart():
    video_ids = df.동영상아이디
    return render_template('chart.html', data=df, video_ids=video_ids, titles=['동영상', '채널명', '날짜', '조회수', '좋아요수', '싫어요수', '댓글수', '카테고리'])


