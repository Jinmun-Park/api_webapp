from flask import Blueprint, render_template, request, redirect
from src.models.bert_load import run_predict
from src.utils.api import channel_search
from src.utils.api import pickle_videos, pickle_videos_filter, pickle_videos_comments
from src.utils.api import globals_videos, globals_videos_filter, globals_videos_comments
from src.utils.api import remove_cleantext

import pandas as pd

# Tokenization
from kiwipiepy import Kiwi
from wordcloud import WordCloud
from collections import Counter
import re
import string
from matplotlib import rc
rc('font', family='NanumBarunGothic')

# ====================== Flask Blueprint ====================== #
bp = Blueprint('dashboard', __name__, url_prefix='/dashboard')

# wordcloud object
# Kiwi Tokenization
kiwi = Kiwi()
def wordcloud(x):
    text = " ".join(x)
    text_list = kiwi.tokenize(text)

    # Form & Tag split
    split_form = pd.DataFrame([item[0] for item in text_list])
    split_form.rename(columns = {0 : 'form'}, inplace = True)
    split_tag = pd.DataFrame([item[1] for item in text_list])
    split_tag.rename(columns={0: 'tag'}, inplace=True)
    text_token = pd.concat([split_form, split_tag], axis=1)

    # Remove tags
    tag = ['NNG', 'NNP', 'NNB', 'NR', 'NP']
    text_token = text_token[text_token['tag'].isin(tag)]

    # Remove manual token
    removal_manual = ['ᆷ', 'ᆼ', 'ᆫ', 'ᆸ니다', '에', '는', '의', '이', '를', '하', '을', '한', '가', '과', '다', '지', '고', '로', '은']
    text_token = text_token[~text_token['form'].isin(removal_manual)]

    # Clean text
    text_token["form"] = text_token["form"].apply(
        lambda x: remove_cleantext(x))
    # Choose nouns > 1
    text_token = text_token[
        text_token['form'].apply(lambda x: len(x) > 1)]

    # Most common words
    counts = Counter(text_token['form'])
    tags = counts.most_common(50)

    # Wordcloud
    wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white", font_path="static/gulim.ttc").generate_from_frequencies(dict(tags))
    wordcloud.to_file("static/images/wc_01.png")

# ====================== Flask Route ====================== #
@bp.route('/', methods=["GET"])
def search():
    return render_template('dashboard.html')

@bp.route('/channel_result', methods=["GET"])
def search_result():
    chanel_name = request.args.get('cha_name')
    search = channel_search(chanel_name)
    return render_template('channel_result.html', title=chanel_name, data=search, titles=['channel_id', 'published_at', 'channel_title', 'view_count','subscriber_count', 'video_count'])

@bp.route('/video_result', methods=['GET', 'POST'])
def video_result():
    if request.method == "POST":
        channel_id = request.form.get("channel_id")
        vid = pickle_videos(type='sample', channel_id=channel_id)
        vid[['view_count', 'like_count', 'comment_count']] = vid[['view_count', 'like_count', 'comment_count']].apply(pd.to_numeric)
        vid_view = vid.sort_values(by="view_count", ascending=False)
        top_view_vid = vid_view['video_title'].iloc[0]
        vid_category = vid['wiki_category'].value_counts().reset_index()
        top_category_vid = vid_category['index'].iloc[0]
        vid_comment = vid.sort_values(by="comment_count", ascending=False)
        top_comment_vid = vid_comment['video_title'].iloc[0]

        # vid = globals_videos(type='sample', channel_id=channel_id)
        # vid = pd.read_pickle('Pickle/video_sample_info.pkl')
        return render_template('video_result.html',
                               title=channel_id,
                               vid=vid,
                               titles=['video_id', 'video_title', 'published_at', 'view_count', 'like_count', 'comment_count', 'wiki_category'],
                               top_view_vid=top_view_vid,
                               top_category_vid=top_category_vid,
                               top_comment_vid=top_comment_vid
                               )

@bp.route('/filter_result', methods=["GET"])
def video_filter():
    find = request.args.get('find')
    vid_filter = pickle_videos_filter(type='sample', find=find)
    vid_filter[['view_count', 'like_count', 'comment_count']] = vid_filter[['view_count', 'like_count', 'comment_count']].apply(pd.to_numeric)
    vid_counts = vid_filter.shape[0]
    vid_view = vid_filter.sort_values(by="view_count", ascending=False)
    top_view_vid = vid_view['video_title'].iloc[0]
    vid_category = vid_filter['wiki_category'].value_counts().reset_index()
    top_category_vid = vid_category['index'].iloc[0]
    vid_comment = vid_filter.sort_values(by="comment_count", ascending=False)
    top_comment_vid = vid_comment['video_title'].iloc[0]

    # vid_filter = globals_videos_filter(find)
    # vid_filter = pd.read_pickle('Pickle/video_info_filter.pkl')
    return render_template('keyword_result.html',
                           title=find,
                           vid=vid_filter,
                           titles=['video_id', 'video_title', 'published_at', 'view_count', 'like_count', 'comment_count', 'wiki_category'],
                           vid_counts=vid_counts,
                           top_view_vid=top_view_vid,
                           top_category_vid=top_category_vid,
                           top_comment_vid=top_comment_vid
                           )

@bp.route('/comment_result', methods=["GET"])
def video_comment():
    vid_comments = pickle_videos_comments(type='sample', option='save')
    wordcloud(vid_comments['comment'])
    return render_template('comment_result.html', data=vid_comments, titles=['comment_id', 'comment', 'author', 'like_count', 'published_at','reply_count'])

@bp.route('/sentiment', methods=["GET"])
def predict():
    result = run_predict()
    result_emotion = result['emotion'].value_counts(normalize=True)
    result_emotion = pd.DataFrame(result_emotion)
    emotions = [i for i in result_emotion.index]
    emotions_rate = [i for i in result_emotion.emotion]
    return render_template('sentiment.html', data=result, result_emotion=result_emotion, emotions=emotions, emotions_rate=emotions_rate, titles=['emotion', 'comment'])







