from src.models.bert_load import run_predict
from flask import Blueprint, render_template
import pandas as pd

bp = Blueprint('sentiment', __name__, url_prefix='/sentiment')

@bp.route('/', methods=["GET"])
def predict():
    result = run_predict()
    result_emotion = result['emotion'].value_counts(normalize=True)
    result_emotion = pd.DataFrame(result_emotion)
    emotions = [i for i in result_emotion.index]
    emotions_rate = [i for i in result_emotion.emotion]
    return render_template('sentiment.html', data=result, result_emotion=result_emotion, emotions=emotions, emotions_rate=emotions_rate, titles=['emotion', 'comment'])

# @bp.route('/공포', methods=["GET"])
# def predict_공포():
#     result = run_predict()
#     공포 = result[result['emotion'] == '공포']
#     # 혐오 = result[result['emotion'] == '혐오']
#     # 분노 = result[result['emotion'] == '분노']
#     return render_template('sentiment.html', tables=[공포.to_html(classes='data')], titles=공포.columns.values)