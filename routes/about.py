from flask import Blueprint, render_template
bp = Blueprint('about', __name__, url_prefix='/about')

@bp.route('/', methods=["GET"])
def about():
    return render_template('about.html')

@bp.route('/jinmunpark', methods=["GET"])
def jmp_profile():
    return render_template('jmp_profile.html')

@bp.route('/eunjeongpark', methods=["GET"])
def ejp_profile():
    return render_template('ejp_profile.html')