import pandas as pd
import datetime as dt
from src.utils.api import flask_category, flask_channel, flask_timeframe

# ====================== Flask Blueprint ====================== #
from flask import Blueprint, render_template
df, df_category, df_channeltitle, df_category_view_per, df_category_like_per, df_category_comment_per, df_top_channel, df_top_category, df_top_comment = flask_category(command='15days')
flask_channel = flask_channel(command='15days')
tf_list_channel, tf_channel, tf_sum, tf_avg, tf_sum_category = flask_timeframe(command='15days')
bp = Blueprint('15day_trend', __name__, url_prefix='/15day_trend')

# ======================== Flask Route ======================== #
@bp.route('/', methods=["GET"])
def second_trend():
    # Count rows
    count_db = len(df.index)
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
    return render_template('second_trend.html',
                           count_db=count_db,
                           category=category,
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
    # Count rows
    count_db = len(df.index)
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
    return render_template('second_trend_category.html',
                           count_db=count_db,
                           category=category,
                           category_rate=category_rate,
                           category_channel=category_channel,
                           category_view_per=category_view_per,
                           category_like_per=category_like_per,
                           category_comment_per=category_comment_per,
                           category_top_channel=category_top_channel,
                           category_top_category=category_top_category,
                           category_top_comment=category_top_comment
                           )

@bp.route('/channel', methods=["GET"])
def trend_channel():
    '''
    from src.utils.api import flask_channel
    global flask_channel
    flask_channel = flask_channel(command='15days')
    '''
    # Channel Top 20 & Bottom 20 names
    channel_top_label = [i for i in flask_channel.head(20).채널명]
    channel_top_view = [i for i in (flask_channel.head(20).채널총조회수)/1000]
    channel_top_subs = [i for i in (flask_channel.head(20).채널구독수)]
    channel_btm_label = [i for i in flask_channel.tail(20).채널명]
    channel_btm_view = [i for i in (flask_channel.tail(20).채널총조회수)/1000]
    channel_btm_subs = [i for i in (flask_channel.tail(20).채널구독수)]
    # Top channel information figures
    top_channel_select = flask_channel[flask_channel['채널총조회수'] == flask_channel['채널총조회수'].max()]
    top_channel = top_channel_select.채널명.iloc[0]
    top_channel_num = format(int(top_channel_select.채널총조회수.iloc[0]), ",")
    # Top channel top right information
    top_channel_url = top_channel_select.썸네일.iloc[0]
    top_channel_videoid = top_channel_select.동영상아이디.iloc[0]
    top_channel_channelid = top_channel_select.채널아이디.iloc[0]
    top_channel_publish = top_channel_select.채널개설날짜.iloc[0]
    top_channel_cateogry = top_channel_select.카테고리.iloc[0]
    top_channel_like = format(int(top_channel_select.좋아요수.iloc[0]), ",")
    top_channel_comment = format(int(top_channel_select.댓글수.iloc[0]), ",")
    # Top channel subscriptions
    top_subs_num = format(int(flask_channel[flask_channel['채널구독수'] == flask_channel['채널구독수'].max()].채널구독수.iloc[0]), ",")
    top_subs = flask_channel[flask_channel['채널구독수'] == flask_channel['채널구독수'].max()].채널명.iloc[0]
    # Latest published channel
    latest_channel_select = flask_channel.sort_values(by='채널개설날짜').tail(1)
    latest_channel = latest_channel_select.채널명.iloc[0]
    latest_channel_num = latest_channel_select.채널개설날짜.iloc[0]
    # Top channel top right information
    latest_channel_url = latest_channel_select.썸네일.iloc[0]
    latest_channel_videoid = latest_channel_select.동영상아이디.iloc[0]
    latest_channel_channelid = latest_channel_select.채널아이디.iloc[0]
    #latest_channel_publish = latest_channel_select.채널개설날짜.iloc[0]
    latest_channel_cateogry = latest_channel_select.카테고리.iloc[0]
    latest_channel_like = format(int(latest_channel_select.좋아요수.iloc[0]), ",")
    latest_channel_comment = format(int(latest_channel_select.댓글수.iloc[0]), ",")
    # trend_channel_chart
    channel_table = flask_channel[['동영상', '날짜', '채널명',  '채널개설날짜', '카테고리',  '채널총조회수', '채널구독수', '채널비디오수']]
    return render_template('second_trend_channel.html',
                           # top & btm 20 channel
                           channel_top_label=channel_top_label,
                           channel_top_view=channel_top_view,
                           channel_top_subs=channel_top_subs,
                           channel_btm_label=channel_btm_label,
                           channel_btm_view=channel_btm_view,
                           channel_btm_subs=channel_btm_subs,
                           # top 1 channel
                           top_channel=top_channel,
                           top_channel_num=top_channel_num,
                           top_channel_url=top_channel_url,
                           top_channel_videoid=top_channel_videoid,
                           top_channel_channelid=top_channel_channelid,
                           top_channel_publish= top_channel_publish,
                           top_channel_cateogry=top_channel_cateogry,
                           top_channel_like=top_channel_like,
                           top_channel_comment=top_channel_comment,
                           top_subs=top_subs,
                           top_subs_num=top_subs_num,
                           # latest channel
                           latest_channel=latest_channel,
                           latest_channel_num=latest_channel_num,
                           latest_channel_url=latest_channel_url,
                           latest_channel_videoid=latest_channel_videoid,
                           latest_channel_channelid=latest_channel_channelid,
                           latest_channel_cateogry=latest_channel_cateogry,
                           latest_channel_like=latest_channel_like,
                           latest_channel_comment=latest_channel_comment,
                           channel_table=channel_table
                           )

@bp.route('/timeframe', methods=["GET"])
def trend_timeframe():
    tf_top_url = tf_channel.썸네일.iloc[0]
    tf_top_videoid = tf_channel.동영상아이디.iloc[0]
    tf_top_channelid = tf_channel.채널아이디.iloc[0]
    tf_top_publish = tf_channel.채널개설날짜.iloc[0]
    tf_top_cateogry = tf_channel.카테고리.iloc[0]
    tf_top_like = format(int(tf_channel.좋아요수.iloc[0]), ",")
    tf_top_comment = format(int(tf_channel.댓글수.iloc[0]), ",")
    # Timeframe sum figures
    tf_sum_date = [i for i in tf_sum.차트일자]
    tf_sum_view = [i for i in (tf_sum.조회수)/100]
    tf_sum_like = [i for i in (tf_sum.좋아요수)/10]
    tf_sum_comm = [i for i in (tf_sum.댓글수)]
    # Timeframe average figures
    tf_avg_date = [i for i in tf_avg.차트일자]
    tf_avg_view = [i for i in (tf_avg.조회수)/100]
    tf_avg_like = [i for i in (tf_avg.좋아요수)/10]
    tf_avg_comm = [i for i in (tf_avg.댓글수)]
    # Timeframe category group sum figures
    tf_cate_date = [i for i in tf_sum_category.차트일자.unique()]
    tf_cate_001 = [i for i in tf_sum_category[tf_sum_category['카테고리'] == '엔터테인먼트'].조회수]
    tf_cate_002 = [i for i in tf_sum_category[tf_sum_category['카테고리'] == '뮤직'].조회수]
    tf_cate_003 = [i for i in tf_sum_category[tf_sum_category['카테고리'] == '푸드'].조회수]
    tf_cate_004 = [i for i in tf_sum_category[tf_sum_category['카테고리'] == '게임'].조회수]
    tf_cate_005 = [i for i in tf_sum_category[tf_sum_category['카테고리'] == '라이프스타일'].조회수]
    tf_cate_006 = [i for i in tf_sum_category[tf_sum_category['카테고리'] == '스포츠'].조회수]
    tf_cate_007 = [i for i in tf_sum_category[tf_sum_category['카테고리'] == '사회'].조회수]
    tf_cate_008 = [i for i in tf_sum_category[tf_sum_category['카테고리'] == '건강'].조회수]
    # Table
    tf_list_channel
    # Category Radar
    tf_radar = tf_sum_category.groupby(['카테고리'])[['조회수', '좋아요수', '댓글수']].sum().reset_index()
    tf_radar_cate = [i for i in tf_radar.카테고리]
    tf_radar_view = [i for i in tf_radar.조회수/100]
    tf_radar_like = [i for i in tf_radar.좋아요수/10]
    tf_radar_comm = [i for i in tf_radar.댓글수]
    return render_template('second_trend_timeframe.html',
                           tf_top_url=tf_top_url,
                           tf_top_videoid=tf_top_videoid,
                           tf_top_channelid=tf_top_channelid,
                           tf_top_publish=tf_top_publish,
                           tf_top_cateogry=tf_top_cateogry,
                           tf_top_like=tf_top_like,
                           tf_top_comment=tf_top_comment,
                           # sum figures
                           tf_sum_date=tf_sum_date,
                           tf_sum_view=tf_sum_view,
                           tf_sum_like=tf_sum_like,
                           tf_sum_comm=tf_sum_comm,
                           # average figures
                           tf_avg_date=tf_avg_date,
                           tf_avg_view=tf_avg_view,
                           tf_avg_like=tf_avg_like,
                           tf_avg_comm=tf_avg_comm,
                           # sum figures
                           tf_cate_date=tf_cate_date,
                           tf_cate_001=tf_cate_001,
                           tf_cate_002=tf_cate_002,
                           tf_cate_003=tf_cate_003,
                           tf_cate_004=tf_cate_004,
                           tf_cate_005=tf_cate_005,
                           tf_cate_006=tf_cate_006,
                           tf_cate_007=tf_cate_007,
                           tf_cate_008=tf_cate_008,
                           # table
                           tf_list_channel=tf_list_channel,
                           # radar
                           tf_radar_cate=tf_radar_cate,
                           tf_radar_view=tf_radar_view,
                           tf_radar_like=tf_radar_like,
                           tf_radar_comm=tf_radar_comm
                           )