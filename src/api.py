# ====================== API READER ====================== #
from urllib.request import Request, urlopen
from urllib.parse import urlencode, quote_plus, unquote
import pandas as pd
from pandas import json_normalize
import json
# ====================== YAML READER ====================== #
import yaml
import os
from datetime import datetime

def run_api(name, environment):
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
    endtime = datetime.now()
    print(endtime)
    timetaken = endtime - starttime
    print('Time taken : ' + timetaken.__str__())

    return df

covid_vaccines = run_api(name='covid_vaccines', environment='api')
