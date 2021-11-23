from src.models.bert_load import run_predict
from flask import Blueprint, render_template, request, redirect
from src.utils.api import channel_search
from src.utils.api import pickle_videos, pickle_videos_filter, pickle_videos_comments
from src.utils.api import globals_videos, globals_videos_filter, globals_videos_comments
import pandas as pd
import pickle

bp = Blueprint('search', __name__, url_prefix='/search')

@bp.route('/', methods=["GET"])
def search():
    return render_template('search.html')

@bp.route('/search_result', methods=["GET"])
def search_result():
    chanel_name = request.args.get('cha_name')
    search = channel_search(chanel_name)
    return render_template('search_result.html', title=chanel_name, data=search, titles=['channel_id', 'published_at', 'channel_title', 'view_count','subscriber_count', 'video_count'])

@bp.route('/video_result', methods=["GET"])
def video_result():
    channel_id = request.args.get('channel_id')
    vid = pickle_videos(type='sample', channel_id=channel_id)
    # vid = globals_videos(type='sample', channel_id=channel_id)
    # vid = pd.read_pickle('Pickle/video_sample_info.pkl')
    return render_template('video_result.html', title=channel_id, data=vid, titles=['video_id', 'video_title', 'published_at', 'view_count', 'like_count', 'dislike_count', 'favorite_count', 'comment_count', 'wiki_category'])

@bp.route('/filter_result', methods=["GET"])
def video_filter():
    find = request.args.get('find')
    vid_filter = pickle_videos_filter(type='sample', find=find)
    # vid_filter = globals_videos_filter(find)
    # vid_filter = pd.read_pickle('Pickle/video_info_filter.pkl')
    return render_template('video_result.html', title=find, data=vid_filter, titles=['video_id', 'video_title', 'published_at', 'view_count', 'like_count', 'dislike_count', 'favorite_count', 'comment_count', 'wiki_category'])

# @bp.route('/comment_result', methods=["GET"])
# def video_comment():
#     # vid_comments = pickle_videos_comments(type='sample', option='delete')
#     data = pd.read_pickle('Pickle/video_comment.pkl')
#     return render_template('video_result.html', tables=[data.to_html(classes='data')])






