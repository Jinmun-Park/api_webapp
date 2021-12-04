import pandas as pd
import datetime as dt
from src.utils.api import flask_category, flask_channel

# ====================== Flask Blueprint ====================== #
from flask import Blueprint, render_template
df, df_category, df_channeltitle, df_category_view_per, df_category_like_per, df_category_comment_per, df_top_channel, df_top_category, df_top_comment = flask_category(command='daily')
bp = Blueprint('latest_trend', __name__, url_prefix='/latest_trend')

# ====================== Flask Route ====================== #
@bp.route('/', methods=["GET"])
def latest_trend():
    # Chart figures
    category = [i for i in df_category.index]
    category_rate = [i for i in df_category.카테고리]
    category_view_per = [i for i in df_category_view_per.조회수]
    category_like_per = [i for i in df_category_like_per.좋아요수]
    category_comment_per = [i for i in df_category_comment_per.댓글수]
    # Table
    category_channel = df.groupby(['카테고리', '채널명']).sum().sort_values('조회수', ascending=False).reset_index()
    category_channel['조회수'] = category_channel['조회수'].map("{:,}".format)
    category_channel['좋아요수'] = category_channel['좋아요수'].map("{:,}".format)
    category_channel['댓글수'] = category_channel['댓글수'].map("{:,}".format)
    # Chart Basic information figures
    category_top_channel = df_top_channel.채널명.iloc[0]
    category_top_category = df_top_category.카테고리.iloc[0]
    category_top_comment = format(df_top_comment.댓글수.iloc[0], ",")
    return render_template('latest_trend.html', category=category,
                           category_rate=category_rate,
                           category_channel=category_channel,
                           category_view_per=category_view_per,
                           category_like_per=category_like_per,
                           category_comment_per=category_comment_per,
                           category_top_channel=category_top_channel,
                           category_top_category=category_top_category,
                           category_top_comment=category_top_comment
                           )

@bp.route('/category', methods=["GET"])
def trend_category():
    # Chart figures
    category = [i for i in df_category.index]
    category_rate = [i for i in df_category.카테고리]
    category_view_per = [i for i in df_category_view_per.조회수]
    category_like_per = [i for i in df_category_like_per.좋아요수]
    category_comment_per = [i for i in df_category_comment_per.댓글수]
    # Table
    category_channel = df.groupby(['카테고리', '채널명']).sum().sort_values('조회수', ascending=False).reset_index()
    category_channel['조회수'] = category_channel['조회수'].map("{:,}".format)
    category_channel['좋아요수'] = category_channel['좋아요수'].map("{:,}".format)
    category_channel['댓글수'] = category_channel['댓글수'].map("{:,}".format)
    # Chart basic information figures
    category_top_channel = df_top_channel.채널명.iloc[0]
    category_top_category = df_top_category.카테고리.iloc[0]
    category_top_comment = format(df_top_comment.댓글수.iloc[0], ",")
    return render_template('category.html',
                           category=category, category_rate=category_rate, category_channel=category_channel,
                           category_view_per=category_view_per, category_like_per=category_like_per, category_comment_per=category_comment_per,
                           category_top_channel=category_top_channel, category_top_category=category_top_category,category_top_comment=category_top_comment
                           )

@bp.route('/channel', methods=["GET"])
def trend_channel():
    global flask_channel
    flask_channel = flask_channel(command='daily')
    # Channel names
    channel_label = [i for i in flask_channel.채널명]
    channel_view = [i for i in (flask_channel.채널총조회수)/1000]
    channel_subs = [i for i in flask_channel.채널구독수]
    # Channel basic information figures
    top_channel = flask_channel[flask_channel['채널총조회수'] == flask_channel['채널총조회수'].max()].채널명.iloc[0]
    top_channel_num = format(int(flask_channel[flask_channel['채널총조회수'] == flask_channel['채널총조회수'].max()].채널총조회수.iloc[0]), ",")
    top_subs = flask_channel[flask_channel['채널구독수'] == flask_channel['채널구독수'].max()].채널명.iloc[0]
    top_subsl_num = format(int(flask_channel[flask_channel['채널구독수'] == flask_channel['채널구독수'].max()].채널구독수.iloc[0]), ",")
    latest_channel = flask_channel.sort_values(by='채널개설날짜').tail(1).채널명.iloc[0]
    latest_channel_num = flask_channel.sort_values(by='채널개설날짜').tail(1).채널개설날짜.iloc[0]
    return render_template('channel.html',
                           channel_label=channel_label, channel_view=channel_view, channel_subs=channel_subs,
                           top_channel=top_channel, top_channel_num=top_channel_num,
                           top_subs=top_subs, top_subsl_num=top_subsl_num,
                           latest_channel=latest_channel, latest_channel_num=latest_channel_num
                           )