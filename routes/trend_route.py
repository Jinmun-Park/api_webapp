import pandas as pd
from src.utils.api import read_pickle, gcp_sql_pull
import datetime as dt
from routes.chart_route import df, df_category, df_channeltitle

from flask import Blueprint, render_template
bp = Blueprint('latest_trend', __name__, url_prefix='/latest_trend')

@bp.route('/', methods=["GET"])
def latest_trend():
    return render_template('latest_trend.html')

@bp.route('/category', methods=["GET"])
def trend_category():
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