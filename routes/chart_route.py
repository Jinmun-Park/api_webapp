import pandas as pd
from src.utils.api import flask_chart
import datetime as dt

# ====================== Flask Blueprint ====================== #
from flask import Blueprint, render_template
df = flask_chart(command='daily')
# Comma Separator
df['조회수'] = df['조회수'].map("{:,}".format)
df['좋아요수'] = df['좋아요수'].map("{:,}".format)
df['댓글수'] = df['댓글수'].map("{:,}".format)
# Blueprint
bp = Blueprint('chart', __name__, url_prefix='/chart')

# ====================== Flask Route ====================== #
@bp.route('/', methods=["GET"])
def chart():
    video_ids = df.동영상아이디
    return render_template('chart.html', data=df, video_ids=video_ids, titles=['동영상', '동영상 제목', '채널명', '날짜', '조회수', '좋아요수', '댓글수', '카테고리'])


