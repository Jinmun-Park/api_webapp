# Web Application using Google API

## Contributor
* Jinmun Park : pjm9827@gmail.com <br/>
* Eunjeong Park : pej111797@gmail.com

## API Resources
| Source | URL | Description |
| --- | --- | --- |
| GoogleAPI | https://developers.google.com/youtube/v3 | Pulling youtube Channel/Videos/Comments |
| 공공데이터포탈 | https://www.data.go.kr/index.do | Pulling Covid 19/ Accident cases | 

## FIRST STAGE (15'SEPT'2021 ~ 15'OCT'2021) :
| PLAN | PERIOD | STATUS | DESCRIPTION |
| --- | --- | --- | --- |
| API SETUP | 01'OCT'2021 | COMPLETED | Setting Up API Connection |
| DATA MAPPING | 10'OCT'2021 | IN PROGRESS | Mapping list of data into dataframe |

##
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
