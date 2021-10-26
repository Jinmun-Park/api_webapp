from flask import Blueprint, render_template
import pandas as pd

def read_pickle(file_name: str) -> pd.DataFrame:
    return pd.read_pickle('Pickle/' + file_name)

bp = Blueprint('ranking', __name__, url_prefix='/ranking')

@bp.route('/', methods=["GET"])
def list_index():
    df = read_pickle('youtube_popular.pkl')
    df = df[['VideoTitle', 'ChannelTitle', 'Tags']]
    return render_template('ranking.html', data=df, tables=[df.to_html(classes='data')], titles=df.columns.values)