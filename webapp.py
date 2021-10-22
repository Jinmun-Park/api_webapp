'''
set FLASK_APP=webapp.py
$env:FLASK_APP = "webapp"
$env:FLASK_ENV = "development"
set FLASK_DEBUG=1
flask run
'''

from flask import Flask, render_template

app = Flask(__name__, template_folder='templates')
#app.run(debug=True, use_debugger=False)

@app.route('/')
def home_page():
    return render_template('home.html')