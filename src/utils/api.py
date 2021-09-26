# ====================== API READER SETUP ====================== #
from urllib.request import Request, urlopen
from urllib.parse import urlencode, quote_plus, unquote
from pandas import json_normalize
import json
import requests #XML DECODING
import xmltodict
# ====================== YAML READER SETUP ====================== #
import yaml
import os
from datetime import datetime

# ====================== RETRIEVING DATA FROM API ====================== #
def api_encodetype(name, environment):
    """
    :param name: main name of YAML configuration
    :param environment: api
    :return:
    """
    starttime = datetime.now()
    print(starttime)
    # ====================== CONFIGURATION.YAML READING ====================== #
    try:
        with open('config/credentials.yaml') as stream:
            credential = yaml.safe_load(stream)
    except yaml.YAMLError as e:
        print("Failed to parse credentials " + e.__str__())
    except Exception as e:
        print("Failed to parse credentials " + e.__str__())
    # ====================== Retrieving API and store in DF  ======================#
    env_cred = credential[name][environment]
    url = env_cred['api_url']
    service_key = env_cred['api_key']
    queryparams = '?' + urlencode({quote_plus('serviceKey'): service_key
                                   })
    try:
        response = urlopen(url + unquote(queryparams))
        json_api = response.read().decode("utf-8")
        json_file = json.loads(json_api)
        print(name.upper() + ' Successfully initiated API Connection')
    except:
        print(name.upper() + ' Failed to initiated API Connection')

    df = json_normalize(json_file['data'])

    # ====================== Export to Pickle  ======================#
    try:
        if not os.path.exists('Pickle/' + name + '.pkl'):
            try:
                os.makedirs('Pickle')
            except FileExistsError:
                pass
            df.to_pickle('Pickle/' + name + '.pkl')
    except Exception as e:
        print('Failed to load ' + name.upper() + e.__str__())
    else:
        print('Successfully loaded ' + name.upper())

    endtime = datetime.now()
    print(endtime)
    timetaken = endtime - starttime
    print('Time taken : ' + timetaken.__str__())

def api_decodetype(name, environment, startdate):
    """
    :param name: main name of YAML configuration
    :param environment: api
    :return:
    """
    starttime = datetime.now()
    print(starttime)
    # ====================== CONFIGURATION.YAML READING ====================== #
    try:
        with open('config/credentials.yaml') as stream:
            credential = yaml.safe_load(stream)
    except yaml.YAMLError as e:
        print("Failed to parse credentials " + e.__str__())
    except Exception as e:
        print("Failed to parse credentials " + e.__str__())
    # ====================== Retrieving API and store in DF  ======================#
    env_cred = credential[name][environment]
    url = env_cred['api_url']
    service_key = env_cred['api_key']
    queryparams = '?' + urlencode({quote_plus('serviceKey'): service_key,
                                   quote_plus('startCreateDt'): startdate})
    try:
        res = requests.get(url + queryparams)
        result = xmltodict.parse(res.text)
        json_file = json.loads(json.dumps(result))
        print(name.upper() + ' Successfully initiated API Connection')
    except:
        print(name.upper() + ' Failed to initiated API Connection')

    df = json_normalize(json_file['response']['body']['items']['item'])
    # ====================== Export to Pickle  ======================#
    try:
        if not os.path.exists('Pickle/' + name + '.pkl'):
            try:
                os.makedirs('Pickle')
            except FileExistsError:
                pass
            df.to_pickle('Pickle/' + name + '.pkl')
    except Exception as e:
        print('Failed to load ' + name.upper() + e.__str__())
    else:
        print('Successfully loaded ' + name.upper())

    endtime = datetime.now()
    print(endtime)
    timetaken = endtime - starttime
    print('Time taken : ' + timetaken.__str__())

# ====================== API RUNNING ====================== #
def run_api():
    api_encodetype(name='covid_vaccines', environment='api')
    api_decodetype(name='covid_age_sex', environment='api', startdate='20200210')
    api_decodetype(name='covid_city', environment='api', startdate='20200210')
    api_decodetype(name='covid_cases', environment='api', startdate='20200210')


