# Web Application using Google API
This project is licensed under the terms of the GNU General Public License v3.0
### Contributor
* Jinmun Park : pjm9827@gmail.com <br/>
* Eunjeong Park : pej111797@gmail.com

### API Resources
| Source | URL | Description |
| --- | --- | --- |
| GoogleAPI | https://developers.google.com/youtube/v3 | Youtube Comments |
| 공공데이터 | https://www.data.go.kr/index.do | Covid 19/ Accident cases | 

### FIRST STAGE (15'SEPT'2021 ~ 20'OCT'2021) :

| PLAN &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;| DATE &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;| STATUS &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;| DESCRIPTION &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; |
| --- | --- | --- | --- |
| API SETUP | 01'OCT'2021 | COMPLETED | API Connection |
| DATA MAPPING | 6'OCT'2021 | COMPLETED | Mapping API |
| POSTGRESSQL | 8'OCT'2021 | COMPLETED | Postgresql DB|
| HEROKU   | OCT | COMPLETED | Autopush |

### SECOND STAGE (20'OCT'2021 ~ ) :

| PLAN &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;| DATE &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;| STATUS &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;| DESCRIPTION &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; |
| --- | --- | --- | --- |
| KOBERT | 25'OCT'2021 | COMPLETED | Sentiment Analysis |
| Google Cloud | - | IN PROGRESS | Replace Heroku |


### SUPPORT
| Source | URL | Description |
| --- | --- | --- |
| Google | https://developers.google.com/youtube/v3 | API Documentation |
| Google | https://support.google.com/youtube/contact/yt_api_form | API Extension |
| Kobert | https://github.com/SKTBrain/KoBERT | Korean Bert |

## PROJECT NOTE
### API SETUP
pip install --upgrade google-api-python-client

### Requirements.txt
pip freeze > requirements.txt <br/>
pip install -r requirements.txt 

### FLASK RUN
set FLASK_APP=webapp.py <br/>
$env:FLASK_APP = "webapp" <br/>
$env:FLASK_ENV = "development" <br/>
set FLASK_DEBUG=1 <br/>
flask run <br/>

### NOTE
시각화 ( 지금 한국에서 가장 유명한 유튜브 영상 top 50 리스트 , 어떤 카테고리 , 채널명 유명한지 .. ) <br/>
title word cloud 시각화 -> '오징어게임 ... ' <br/>
댓글 감성분석 <br/>
