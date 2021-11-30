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
    df = df[['video_title', 'video_id', 'channel_title', 'published_at', 'view_count', 'like_count', 'comment_count', 'wiki_category']]
    # Rename columns (English - Korean)
    df.rename(columns={'video_title': '동영상',
                       'video_id': '동영상아이디',
                       'channel_title': '채널명',
                       'published_at': '날짜',
                       'view_count': '조회수',
                       'like_count': '좋아요수',
                       'comment_count': '댓글수',
                       'wiki_category': '카테고리'
                       }, inplace=True)
    # Reset index
    df = df.reset_index(drop=True)
    # Converting field type : Date
    df['날짜'] = pd.to_datetime(df['날짜'], infer_datetime_format=True)
    df['날짜'] = df['날짜'].dt.strftime("%Y-%m-%d")
    # Converting field type : Numeric
    df[['조회수', '좋아요수', '댓글수']] = df[['조회수', '좋아요수', '댓글수']].apply(pd.to_numeric)
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

    return df

def flask_chart_analysis():

    df = flask_popular_chart()

    # ================== FIELD ANALYSIS ================== #
    # Category Count
    df_category = df['카테고리'].value_counts(normalize=True) * 100
    df_category = pd.DataFrame(df_category)

    # Channeltitle Count
    df_channeltitle = df['채널명'].value_counts(normalize=True) * 100
    df_channeltitle = pd.DataFrame(df_channeltitle)

    # Category Numeric Sum
    df_category_view = pd.DataFrame(df.groupby(['카테고리'])['조회수'].sum())
    df_category_view_per = pd.DataFrame((df_category_view['조회수'] / df_category_view['조회수'].sum()) * 100)
    df_category_like = pd.DataFrame(df.groupby(['카테고리'])['좋아요수'].sum())
    df_category_like_per = pd.DataFrame((df_category_like['좋아요수'] / df_category_like['좋아요수'].sum()) * 100)
    df_category_comment = pd.DataFrame(df.groupby(['카테고리'])['댓글수'].sum())
    df_category_comment_per = pd.DataFrame((df_category_comment['댓글수'] / df_category_comment['댓글수'].sum()) * 100)


    return df, df_category, df_channeltitle, df_category_view_per, df_category_like_per, df_category_comment_per

# ====================== Flask ====================== #
from flask import Blueprint, render_template
df = flask_popular_chart()
bp = Blueprint('chart', __name__, url_prefix='/chart')

@bp.route('/', methods=["GET"])
def chart():
    video_ids = df.동영상아이디
    return render_template('chart.html', data=df, video_ids=video_ids, titles=['동영상', '동영상 제목', '채널명', '날짜', '조회수', '좋아요수', '댓글수', '카테고리'])


