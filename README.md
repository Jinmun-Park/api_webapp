# Web Application using Google API
This project is licensed under the terms of the GNU General Public License v3.0
## Contributor
* Jinmun Park :  <br/>
* Eunjeong Park : 

## API Resources
| Source | URL | Description |
| --- | --- | --- |
| GoogleAPI | https://developers.google.com/youtube/v3 | Pulling youtube Channel/Videos/Comments |
| 공공데이터포탈 | https://www.data.go.kr/index.do | Pulling Covid 19/ Accident cases | 

## FIRST STAGE (15'SEPT'2021 ~ 15'OCT'2021) :
| PLAN | DATE | STATUS | DESCRIPTION |
| --- | --- | --- | --- |
| API SETUP | 01'OCT'2021 | COMPLETED | Setting Up API Connection |
| DATA MAPPING | 6'OCT'2021 | COMPLETED | Mapping listS of data |
| POSTGRESSQL | 8'OCT'2021 | COMPLETED | Connecting popular chart to Postgresql |
| JENKINS   | - | IN PROGRESS | Setup auto delivery function to push popular chart into Postgresql |

## SUPPORT
| Source | URL | Description |
| --- | --- | --- |
| Google | https://developers.google.com/youtube/v3 | API Documentation |
| Google | https://support.google.com/youtube/contact/yt_api_form | API Extension | 

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
유튜브 API -> 데이터 시각화(title word cloud 시각화...), 댓글 감성분석 <br/>
코로나19 데이터 -> 코로나19 예측 모델링 <br/>
