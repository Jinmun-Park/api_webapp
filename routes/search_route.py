from src.models.bert_load import run_predict
from flask import Blueprint, render_template, request
from src.utils.api import channel_search, channel_videos, channel_videos_filter, channel_videos_comments

bp = Blueprint('search', __name__, url_prefix='/search')

@bp.route('/', methods=["GET"])
def search():
    return render_template('search.html')

@bp.route('/search_result', methods=["GET"])
def search_result():
    cha_name = request.args.get('cha_name')
    search = channel_search(cha_name)
    return render_template('search_result.html', tables=[search.to_html(classes='data')])

@bp.route('/video_result', methods=["GET"])
def video_result():
    channel_id = request.args.get('channel_id')
    vid = channel_videos(channel_id)
    return render_template('search_result.html', tables=[vid.to_html(classes='data')])

@bp.route('/filter_result', methods=["GET"])
def video_filter():
    find = request.args.get('find')
    vid_filter = channel_videos_filter(find)
    vid_comments = channel_videos_comments()
    return render_template('search_result.html', tables=[vid_comments.to_html(classes='data')])

# data = pd.read_pickle('video_comment.pkl')

# @bp.route('/', methods=["GET"])
# def predict():
#     return render_template('sentiment.html')
#
# @bp.route('/공포', methods=["GET"])
# def predict_공포():
#     result = run_predict()
#     공포 = result[result['emotion'] == '공포']
#     # 혐오 = result[result['emotion'] == '혐오']
#     # 분노 = result[result['emotion'] == '분노']
#     return render_template('sentiment.html', tables=[공포.to_html(classes='data')], titles=공포.columns.values)




