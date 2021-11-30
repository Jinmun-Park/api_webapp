import pandas as pd
from src.utils.api import read_pickle, gcp_sql_pull
import datetime as dt
from routes.chart_route import flask_chart_analysis

from flask import Blueprint, render_template
df, df_category, df_channeltitle, df_category_view_per, df_category_like_per, df_category_comment_per = flask_chart_analysis()
bp = Blueprint('latest_trend', __name__, url_prefix='/latest_trend')

@bp.route('/', methods=["GET"])
def latest_trend():
    return render_template('latest_trend.html')

@bp.route('/category', methods=["GET"])
def trend_category():
    category = [i for i in df_category.index]
    category_rate = [i for i in df_category.카테고리]
    category_channel = df.groupby(['카테고리', '채널명']).sum().sort_values('조회수', ascending=False).reset_index()
    category_view = [i for i in df_category_view_per.조회수]
    category_like = [i for i in df_category_like_per.좋아요수]
    category_comment = [i for i in df_category_comment_per.댓글수]
    return render_template('category.html', category=category,
                           category_rate=category_rate,
                           category_channel=category_channel,
                           category_view=category_view,
                           category_like=category_like,
                           category_comment=category_comment
                           )

@bp.route('/channel', methods=["GET"])
def channel():
    channeltitle = [i for i in df_channeltitle.index]
    channeltitle_rate = [i for i in df_channeltitle.채널명]
    channeltitle = df.groupby('채널명').sum().sort_values('조회수', ascending=False).reset_index()
    return render_template('channel.html', data=df, channeltitle=channeltitle)