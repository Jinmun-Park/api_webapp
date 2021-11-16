from src.models.bert_load import run_predict
from flask import Blueprint, render_template

bp = Blueprint('sentiment', __name__, url_prefix='/sentiment')

@bp.route('/', methods=["GET"])
def predict():
    return render_template('sentiment.html')

@bp.route('/공포', methods=["GET"])
def predict_공포():
    result = run_predict()
    공포 = result[result['emotion'] == '공포']
    # 혐오 = result[result['emotion'] == '혐오']
    # 분노 = result[result['emotion'] == '분노']
    return render_template('sentiment.html', tables=[공포.to_html(classes='data')], titles=공포.columns.values)



