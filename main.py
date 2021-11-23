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

    from routes.chart_route import bp as chart_bp
    from routes.dashboard_route import bp as dashboard_bp
    from routes.trend_route import bp as trend_bp
    app.register_blueprint(chart_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(trend_bp)

    @app.route('/')
    def home_page():
        return render_template('index.html')
    return app

app = create_app()

if __name__ == '__main__':
    HOST = '0.0.0.0'
    PORT = 8080
    app.run(HOST, PORT, debug=True)