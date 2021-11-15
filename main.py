'''
set FLASK_APP=main.py
$env:FLASK_APP = "main"
$env:FLASK_ENV = "development"
set FLASK_DEBUG=1
flask run
'''

from flask import Flask, render_template

def create_app():
    app = Flask(__name__)

    from routes.ranking_route import bp as ranking_bp
    from routes.sentiment_route import bp as sentiment_bp
    app.register_blueprint(ranking_bp)
    app.register_blueprint(sentiment_bp)

    @app.route('/')
    def home_page():
        return render_template('index.html')
    return app

app = create_app()

if __name__ == '__main__':
    HOST = '0.0.0.0'
    PORT = 8080
    app.run(HOST, PORT, debug=True)