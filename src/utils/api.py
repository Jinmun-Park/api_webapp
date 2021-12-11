# ====================== LIBRARY SETUP ====================== #
# YOUTUBE API SETUP
from googleapiclient.discovery import build #pip install google-api-python-client
from pandas import json_normalize
# KOR GOV COVID API SETUP
#from urllib.request import Request, urlopen #공공데이터 API
#from urllib.parse import urlencode, quote_plus, unquote #공공데이터 API
#import json #공공데이터 API
#import requests #XML DECODING & 공공데이터 API
#import xmltodict ##공공데이터 API

# DIRECTORY SETUP
import os
# PICKLE SETUP
import pickle
# LOG/COLUMN SETUP
import pandas as pd
from datetime import datetime, date
import calendar
# GOOGLE SETUP
import google.cloud.secretmanager as secretmanager #pip install google-cloud-secret-manager
from google.cloud.sql.connector import connector #pip install cloud-sql-python-connector[pymysql] #FINAL LIBRARY

# Tokenization
from kiwipiepy import Kiwi
from wordcloud import WordCloud
from collections import Counter
import re
import string
from matplotlib import rc
rc('font', family='NanumBarunGothic')

# ====================== FUNCTION SETUP ====================== #
def secret_manager_setup():
    """
    DESCRIPTION : Retrieving Google Secret keys using Google Application Crendentials.json
    USAGE : Google Application Credentials setup
    """
    # GOOGLE CREDENTIALS & SECRET MANAGER
    project_id = "api-website-333307"
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "config/api-website-333307-2f9aa748b514.json"
    client = secretmanager.SecretManagerServiceClient()
    return project_id, client

def get_secrets(secret_request, project_id, client):
    """
    DESCRIPTION : Getting keys using 'secret_manager_setup' function
    USAGE : Google Secret Manager setup
    """
    name = f"projects/{project_id}/secrets/{secret_request}/versions/latest"
    response = client.access_secret_version(name=name)
    return response.payload.data.decode("UTF-8")

def keys():
    """
    DESCRIPTION : Getting Secret Keys using 'secret_manager_setup' and 'get_secrets' functions
    USAGE : Retrieving keys from Google Secret Manager
    """
    # GOOGLE SECRET MANAGER
    project_id, client = secret_manager_setup()
    # GCP CLOUD ENGINE
    connection_name = get_secrets("connection_name", project_id, client)
    query_string = dict({"unix_socket": "/cloudsql/{}".format(connection_name)})
    db_name = get_secrets("db_name", project_id, client)
    db_user = get_secrets("db_user", project_id, client)
    db_password = get_secrets("db_password", project_id, client)
    driver_name = 'mysql+pymysql'
    # GCP CLOUD PUBLIC IP
    public_ip = get_secrets("public_ip", project_id, client)
    # YOUTUBE API SERVICE KEY
    service_key = get_secrets("service_key", project_id, client)
    return connection_name, query_string, db_name, db_user, db_password, driver_name, public_ip, service_key

# PICKLE SETUP
def pickle_replace(name, file):
    """
    DESCRIPTION 1 : Creating Path and Pickle file if these are not existing.
    DESCROPTION 2 : If Pickle file is already existing in your directory, then it will be replaced with new pickle files.
    USAGE : Most API Calls in this project will use 'pickle_replace' function to store the dataframe in pickle format.
    """
    # Create pickle directory path if it does not exist
    try:
        if not os.path.exists('Pickle/'):
            try:
                os.makedirs('Pickle')
            except FileExistsError:
                pass
    except Exception as e:
        print('Failed to create directory (Pickle/..) ' + name.upper() + e.__str__())
    else:
        print('Successfully created directory (Pickle/..) ' + name.upper())
    # Create pickle file if the file is not existing (If the file exist, then write it again)
    try:
        if os.path.exists('Pickle/' + name + '.pkl'):
            with open('Pickle/' + name + '.pkl', 'wb') as f:
                pickle.dump(file, f)
        else:
            file.to_pickle('Pickle/' + name + '.pkl')
    except Exception as e:
        print('Failed to export(.pkl) ' + name.upper() + e.__str__())
    else:
        print('Successfully export(.pkl) ' + name.upper())

def read_pickle(file_name: str) -> pd.DataFrame:
    return pd.read_pickle('Pickle/' + file_name)

# ==================== RETRIEVING DATA FROM GOOGLE CLOUD SQL =================== #
def gcp_sql_pull(command):
    """
    SECTION : GCP, Chart
    DESCRIPTION : Pulling popular chart from GCP SQL database
    USAGE : command='daily' extracts daily popular chart, whereas command='accumulated' extracts accumulated chart.
    """
    # ======================== GOOGLE SECRET Reading ========================= #
    connection_name, query_string, db_name, db_user, db_password, driver_name, public_ip, service_key = keys()
    # ======================================================================== #

    # ============================ GCP Connection ============================ #
    # Create connection
    conn = connector.connect(
        connection_name,
        "pymysql",
        user=db_user,
        password=db_password,
        db=db_name
    )
    if command == 'daily':
        com = 'SELECT * FROM youtube_daily_chart'
    else:
        com = 'SELECT * FROM youtube_chart'

    # Connect & Execute
    try:
        cursor = conn.cursor()
        cursor.execute(com)
        result = cursor.fetchall()
        df = pd.DataFrame(result)
        df.columns = ['run_date', 'run_time', 'day', 'video_title', 'video_id', 'channel_title', 'channel_id', 'published_at',
                      'category_id', 'view_count', 'like_count', 'favorite_count', 'comment_count', 'for_kids',
                      'wiki_category', 'reg_category']
    except Exception as e:
        return 'Error: {}'.format(str(e))
    return df

# ====================== RETRIEVING DATA FROM YOUTUBE API ====================== #

def api_youtube_popular(name, max_result):
    """
    WARNING : 'api_youtube_popular' is no longer using. Query in Google SQL is scheduled to be pulled using Google Function.
    SECTION : chart
    DESCRIPTION : Retrieving 'mostPopular' videos in 'KR' region.
    USAGE : Store a number of results in pickle format.
    """
    # ====================== Setup ====================== #
    pd.options.mode.chained_assignment = None  # Off warning messages, default='warn'
    starttime = datetime.now()
    print(starttime)
    dictionary = {0: 'wiki_category'}
    dictionary_list = list(dictionary.values())
    pickle_name = name


    # ======================== GOOGLE SECRET Reading ========================= #
    connection_name, query_string, db_name, db_user, db_password, driver_name, public_ip, service_key = keys()
    youtube = build('youtube', 'v3', developerKey=service_key)
    # ======================================================================== #

    # ====================== Retrieving API and store in DF  ====================== #
    try:
        # YOUTUBE_VIDEO_LIST
        res_popular = youtube.videos().list(part=['snippet', 'statistics', 'status', 'topicDetails'],
                                            chart='mostPopular',
                                            maxResults=int(max_result), regionCode='KR').execute()
        df_popular = json_normalize(res_popular['items'])
        print('Videos API Connection has been successfully completed')

        # YOUTUBE_VIDEO_CATEGORY
        res_videocategory = youtube.videoCategories().list(part='snippet', regionCode='KR').execute()
        df_videocategory = json_normalize(res_videocategory['items'])
        df_videocategory = df_videocategory[['id', 'snippet.title']]
        print('VideoCategories API Connection has been successfully completed')
    except:
        print(name.upper() + ' has failed to open API Connection')

    # ====================== YOUTUBE_VIDEO_LIST : Data Mapping  ====================== #
    # Select Columns
    df_popular = df_popular[
        ['snippet.title', 'id', 'snippet.channelTitle', 'snippet.channelId', 'snippet.publishedAt',
         'snippet.categoryId',  # video().list(part='snippet')
         'statistics.viewCount', 'statistics.likeCount', 'statistics.favoriteCount',
         'statistics.commentCount',  # video().list(part='statistics')
         'topicDetails.topicCategories',  # video().list(part='topicDetails')
         'status.madeForKids']]

    # Rename Columns
    df_popular.rename(columns={'snippet.title': 'video_title',
                               'id': 'video_id',
                               'snippet.channelTitle': 'channel_title',
                               'snippet.channelId': 'channel_id',
                               'snippet.publishedAt': 'published_at',
                               'snippet.categoryId': 'category_id',
                               'statistics.viewCount': 'view_count',
                               'statistics.likeCount': 'like_count',
                               'statistics.favoriteCount': 'favorite_count',
                               'statistics.commentCount': 'comment_count',
                               'topicDetails.topicCategories': 'topic_categories',
                               'status.madeForKids': 'for_kids',
                               }, inplace=True)

    # Reset Index
    df_popular = df_popular.reset_index(drop=True)

    # Split TopicCategories URL
    catrgory_split = df_popular['topic_categories']
    catrgory_split = pd.DataFrame(catrgory_split)
    catrgory_split = catrgory_split['topic_categories'].apply(pd.Series).rename(columns=dictionary)

    # Filter columns based on the length
    dictionary_list = dictionary_list[0:len(catrgory_split.columns)]

    # Split WIKI_URL and pick up the last word (Filtering category)
    for i in range(len(catrgory_split.columns)):
        df = catrgory_split.iloc[:, i].str.split('/').apply(pd.Series).iloc[:, -1]
        df.columns = [i]
        catrgory_split[i] = df

    # Remove & Rename columns
    catrgory_split.drop(dictionary_list, axis=1, inplace=True)
    catrgory_split = catrgory_split.rename(columns=dictionary)
    catrgory_split = catrgory_split['wiki_category']

    # Merge & Rename columns
    df_popular = df_popular.merge(catrgory_split, left_index=True, right_index=True)
    del df_popular['topic_categories']
    print('Youtube Video List : Data mapping has been successfully completed')

    # ====================== YOUTUBE_VIDEO_CATEGORY : Data Mapping  ====================== #
    df_videocategory = df_videocategory[['id', 'snippet.title']]
    df_videocategory.rename(columns={'id': 'category_id',
                                     'snippet.title': 'reg_category'
                                     }, inplace=True)
    print('Youtube Video Category : Data mapping has been successfully completed')

    # ====================== MERGE : df_popular & df_videocategory ====================== #
    youtube_popular = df_popular.merge(df_videocategory, how='inner', on='category_id')

    # MYSQL insert 'run_date' and 'day'
    youtube_popular['run_date'] = date.today()
    youtube_popular['run_time'] = datetime.now().time().strftime('%H:%M:%S')
    youtube_popular['day'] = calendar.day_name[date.today().weekday()]
    # Move last column(run_date) to first sequence
    cols = youtube_popular.columns.tolist()
    cols = cols[-3:] + cols[:-3]
    youtube_popular = youtube_popular[cols]

    # ====================== Export to Pickle & Read  ====================== #
    pickle_replace(name=pickle_name, file=youtube_popular)
    youtube_popular = read_pickle('youtube_popular.pkl')
    # ====================================================================== #

    endtime = datetime.now()
    print(endtime)
    timetaken = endtime - starttime
    print('Time taken : ' + timetaken.__str__())

    return youtube_popular

def flask_chart(command):
    """
    SECTION : flask, channel, analysis
    DESCRIPTION : Select columns from popular chart and converting popular chart into Korean.
    USAGE : command='daily' extracts daily popular chart, whereas command='accumulated' extracts accumulated chart.
    """
    # ====================== READ FILES ====================== #
    # Read pickle files
    #df = read_pickle('youtube_popular.pkl')
    if command == 'daily':
        df = gcp_sql_pull(command='daily')
    elif command == '15days':
        df = gcp_sql_pull(command='accumulated')
        df = df.tail(300)
    else:
        df = gcp_sql_pull(command='accumulated')
        df = df.tail(600)
    #df = gcp_sql_ip_connection()

    # ================== DATA PREPROCESSING ================== #
    # Select columns
    df = df[['video_title', 'video_id', 'channel_title', 'published_at', 'view_count', 'like_count', 'comment_count', 'wiki_category']]
    # Rename columns (English - Korean)
    df.rename(columns={'video_title': '동영상',
                       'video_id': '동영상아이디',
                       'channel_title': '채널명',
                       'published_at': '날짜',
                       'view_count': '조회수',
                       'like_count': '좋아요수',
                       'comment_count': '댓글수',
                       'wiki_category': '카테고리'
                       }, inplace=True)
    # Reset index
    df = df.reset_index(drop=True)
    # Converting field type : Date
    df['날짜'] = pd.to_datetime(df['날짜'], infer_datetime_format=True)
    df['날짜'] = df['날짜'].dt.strftime("%Y-%m-%d")
    # Converting field type : Numeric
    df[['조회수', '좋아요수', '댓글수']] = df[['조회수', '좋아요수', '댓글수']].apply(pd.to_numeric)
    # Converting field type : Category
    df['카테고리'].replace('Entertainment', '엔터테인먼트', inplace=True)
    df['카테고리'].replace(['Music','Hip_hop_music','Electronic_music'], '뮤직', inplace=True)
    df['카테고리'].replace('Food', '푸드', inplace=True)
    df['카테고리'].replace('Video_game_culture', '게임', inplace=True)
    df['카테고리'].replace('Lifestyle_(sociology)', '라이프스타일', inplace=True)
    df['카테고리'].replace(['Sport', 'Association_football'], '스포츠', inplace=True)
    df['카테고리'].replace('Society', '사회', inplace=True)
    df['카테고리'].replace('Health', '건강', inplace=True)
    # Counting index
    df.index = df.index+1

    return df

def flask_category(command):
    """
    SECTION : flask, channel, analysis
    DESCRIPTION : Extending flask_chart() to category and converting them into Korean.
    USAGE : command='daily' extracts daily popular chart, whereas command='accumulated' extracts accumulated chart.
    """
    # ====================== READ FILES ====================== #
    if command == 'daily':
        df = flask_chart(command='daily')
    elif command == '15days':
        df = flask_chart(command='15days')
    else:
        df = flask_chart(command='30days')

    # ================== FIELD ANALYSIS ================== #
    # Category Count
    df_category = df['카테고리'].value_counts(normalize=True) * 100
    df_category = pd.DataFrame(df_category)

    # Channeltitle Count
    df_channeltitle = df['채널명'].value_counts(normalize=True) * 100
    df_channeltitle = pd.DataFrame(df_channeltitle)

    # Category Percentage
    df_category_view = pd.DataFrame(df.groupby(['카테고리'])['조회수'].sum())
    df_category_view_per = pd.DataFrame((df_category_view['조회수'] / df_category_view['조회수'].sum()) * 100)
    df_category_like = pd.DataFrame(df.groupby(['카테고리'])['좋아요수'].sum())
    df_category_like_per = pd.DataFrame((df_category_like['좋아요수'] / df_category_like['좋아요수'].sum()) * 100)
    df_category_comment = pd.DataFrame(df.groupby(['카테고리'])['댓글수'].sum())
    df_category_comment_per = pd.DataFrame((df_category_comment['댓글수'] / df_category_comment['댓글수'].sum()) * 100)

    # Chart Information
    df_top_channel = df[df['조회수'] == df['조회수'].max()] #df.채널명
    df_top_category = df_category_view_per[df_category_view_per['조회수'] == df_category_view_per['조회수'].max()].reset_index() #index
    df_top_comment = df[df['댓글수'] == df['댓글수'].max()] #df.댓글수

    return df, df_category, df_channeltitle, \
           df_category_view_per, df_category_like_per, df_category_comment_per, \
           df_top_channel, df_top_category, df_top_comment

def flask_channel(command):
    """
    SECTION : flask, channel, analysis
    DESCRIPTION : Extract channel information for flask app using popular trend from gcp_sqp_pull()
    USAGE 1 : Channel ID from popular chart will trigger to extract channel information
    USAGE 2 : command='daily' extracts daily popular chart, whereas command='accumulated' extracts accumulated chart.
    """
    pd.options.mode.chained_assignment = None  # Off warning messages, default='warn'
    starttime = datetime.now()
    print(starttime)

    # ======================== GOOGLE SECRET Reading ========================= #
    connection_name, query_string, db_name, db_user, db_password, driver_name, public_ip, service_key = keys()
    youtube = build('youtube', 'v3', developerKey=service_key)
    # ======================================================================== #

    # ======================  Retrieving API : YOUTUBE_CHANNEL_INFO  ====================== #
    if command == 'daily':
        chart = gcp_sql_pull(command='daily')
    elif command == '15days':
        chart = gcp_sql_pull(command='accumulate')
        # Convert view_count into int
        chart['view_count'] = chart['view_count'].astype(str).astype(int)
        # Sort by view_count
        chart = chart.tail(300).sort_values('view_count', ascending = False)
        # Remove duplicates
        chart = chart.drop_duplicates(subset='video_id', keep="last")
        high = chart.head(20)
        low = chart.tail(20)
        chart = pd.concat([high, low], ignore_index=True)
    else:
        chart = gcp_sql_pull(command='accumulate')
        # Convert view_count into int
        chart['view_count'] = chart['view_count'].astype(str).astype(int)
        # Sort by view_count
        chart = chart.tail(300).sort_values('view_count', ascending = False)
        # Remove duplicates
        chart = chart.drop_duplicates(subset='video_id', keep="last")
        high = chart.head(20)
        low = chart.tail(20)
        chart = pd.concat([high, low], ignore_index=True)

    # Running loop to get Channel_ID from 'df_channel_search'
    df_channel_info = pd.DataFrame()

    for id in chart['channel_id']:
        try:
            # YOUTUBE_CHANNEL_INFORMATION
            res_channel = youtube.channels().list(part=['snippet', 'statistics', 'contentDetails'],
                                                  id=id).execute()
            dataframe = json_normalize(res_channel['items'][0])
            df_channel_info = df_channel_info.append(dataframe)
            print('Channel ID API Connection : ' + id + '  has been successfully completed')
        except:
            print('Channel ID : ' + id + ' has failed to open API connection')

    # ====================== YOUTUBE_CHANNEL_INFO : Data Mapping  ====================== #
    # Select Columns
    df_channel_info = df_channel_info[['id',
                                       'snippet.thumbnails.medium.url',
                                       #'snippet.customUrl',
                                       'snippet.publishedAt',
                                       'statistics.viewCount',
                                       'statistics.subscriberCount',
                                       'statistics.videoCount']]

    # Rename Columns
    df_channel_info.rename(columns={'id': 'channel_id',
                                    'snippet.thumbnails.medium.url': 'thumbnails',
                                    #'snippet.customUrl': 'custom_url',
                                    'snippet.publishedAt': 'channel_published',
                                    'statistics.viewCount': 'channel_view',
                                    'statistics.subscriberCount': 'channel_subscribers',
                                    'statistics.videoCount': 'channel_videos'
                                    }, inplace=True)


    # ====================== MERGE : df_channel_search & df_channel_info ====================== #
    # Merge & Rename columns
    df = pd.DataFrame()
    df = chart.merge(df_channel_info, how='inner', on='channel_id')
    df = df.drop_duplicates()
    # ========================================================================================= #

    # Select columns
    df = df[['thumbnails', 'video_title', 'video_id', 'channel_id', 'channel_title', 'published_at', 'view_count', 'like_count', 'comment_count', 'wiki_category',
             'channel_published', 'channel_view', 'channel_subscribers', 'channel_videos']]
    # Rename columns (English - Korean)
    df.rename(columns={'thumbnails': '썸네일',
                       'video_title': '동영상',
                       'video_id': '동영상아이디',
                       'channel_id': '채널아이디',
                       'channel_title': '채널명',
                       'published_at': '날짜',
                       'view_count': '조회수',
                       'like_count': '좋아요수',
                       'comment_count': '댓글수',
                       'wiki_category': '카테고리',
                       'channel_published': '채널개설날짜',
                       'channel_view': '채널총조회수',
                       'channel_subscribers': '채널구독수',
                       'channel_videos': '채널비디오수'
                       }, inplace=True)
    # Reset index
    df = df.reset_index(drop=True)
    # Converting field type : Date
    df['날짜'] = pd.to_datetime(df['날짜'], infer_datetime_format=True)
    df['날짜'] = df['날짜'].dt.strftime("%Y-%m-%d")
    df['채널개설날짜'] = pd.to_datetime(df['채널개설날짜'], infer_datetime_format=True)
    df['채널개설날짜'] = df['채널개설날짜'].dt.strftime("%Y-%m-%d")
    # Nan Value
    df['좋아요수'] = df['좋아요수'].fillna(0)
    df['댓글수'] = df['댓글수'].fillna(0)
    df['채널총조회수'] = df['채널총조회수'].fillna(0)
    df['채널구독수'] = df['채널구독수'].fillna(0)
    # Converting field type : Numeric
    df[['조회수', '좋아요수', '댓글수', '채널총조회수', '채널구독수', '채널비디오수']] = df[['조회수', '좋아요수', '댓글수', '채널총조회수', '채널구독수', '채널비디오수']].apply(pd.to_numeric)
    # Converting field type : Category
    df['카테고리'].replace('Entertainment', '엔터테인먼트', inplace=True)
    df['카테고리'].replace(['Music','Hip_hop_music','Electronic_music'], '뮤직', inplace=True)
    df['카테고리'].replace('Food', '푸드', inplace=True)
    df['카테고리'].replace('Video_game_culture', '게임', inplace=True)
    df['카테고리'].replace('Lifestyle_(sociology)', '라이프스타일', inplace=True)
    df['카테고리'].replace(['Sport', 'Association_football'], '스포츠', inplace=True)
    df['카테고리'].replace('Society', '사회', inplace=True)
    df['카테고리'].replace('Health', '건강', inplace=True)
    # Counting index
    df.index = df.index+1

    # end time
    endtime = datetime.now()
    print(endtime)
    timetaken = endtime - starttime
    print('Time taken : ' + timetaken.__str__())

    return df

def remove_cleantext(text):
    '''Make text lowercase, remove text in square brackets,remove links,remove punctuation
    and remove words containing numbers.'''
    text = str(text).lower() # Lower
    text = re.sub('\[.*?\]', '', text) # SquareBracket
    text = re.sub('https?://\S+|www\.\S+', '', text) # URL
    text = re.sub('<.*?>+', '', text) # Bracket
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text) # Punctuation
    text = re.sub('\n', ' ', text) #
    text = re.sub('(?<=#)\w+', '', text) # Hash
    text = re.sub('[\w\.-]+@[\w\.-]+', '', text) # Email
    text = re.sub('[0-9]+', '', text) # Number
    text = text.strip()
    #text = text.strip() # Remove Space
    #text = text.split() # Remove Space
    return text

def tf_conversion(data):
    # ================== DATA PREPROCESSING ================== #
    # Select columns
    select = data[['run_date', 'video_title', 'video_id', 'channel_title', 'channel_id', 'published_at', 'view_count', 'like_count', 'comment_count', 'wiki_category']]
    # To avoid warning
    df = select.copy()
    # Rename columns (English - Korean)
    df.rename(columns={
        'run_date':'차트일자',
        'video_title':'동영상',
        'video_id':'동영상아이디',
        'channel_title':'채널명',
        'channel_id':'채널아이디',
        'published_at':'날짜',
        'view_count':'조회수',
        'like_count':'좋아요수',
        'comment_count':'댓글수',
        'wiki_category':'카테고리'
        }, inplace=True)
    # Reset index
    df = df.reset_index(drop=True)
    # Converting field type : Date
    df['차트일자'] = pd.to_datetime(df['차트일자'], infer_datetime_format=True)
    df['차트일자'] = df['차트일자'].dt.strftime("%Y-%m-%d")
    df['날짜'] = pd.to_datetime(df['날짜'], infer_datetime_format=True)
    df['날짜'] = df['날짜'].dt.strftime("%Y-%m-%d")
    # Converting field type : Numeric
    df[['조회수', '좋아요수', '댓글수']] = df[['조회수', '좋아요수', '댓글수']].apply(pd.to_numeric)
    # Converting field type : Category
    df['카테고리'].replace('Entertainment', '엔터테인먼트', inplace=True)
    df['카테고리'].replace(['Music','Hip_hop_music','Electronic_music'], '뮤직', inplace=True)
    df['카테고리'].replace('Food', '푸드', inplace=True)
    df['카테고리'].replace('Video_game_culture', '게임', inplace=True)
    df['카테고리'].replace('Lifestyle_(sociology)', '라이프스타일', inplace=True)
    df['카테고리'].replace(['Sport', 'Association_football'], '스포츠', inplace=True)
    df['카테고리'].replace('Society', '사회', inplace=True)
    df['카테고리'].replace('Health', '건강', inplace=True)
    # Counting index
    df.index = df.index+1

    return df

def flask_timeframe(command):
    """
    SECTION : flask, channel, analysis
    DESCRIPTION : Run analysis with gcp_sqp_pull()
    USAGE : Channel ID from popular chart will trigger to extract channel information
    """

    # =============================== GOOGLE SECRET Reading =============================== #
    connection_name, query_string, db_name, db_user, db_password, driver_name, public_ip, service_key = keys()
    youtube = build('youtube', 'v3', developerKey=service_key)
    # ===================================================================================== #

    # ======================  Retrieving API : YOUTUBE_CHANNEL_INFO  ====================== #
    if command == 'daily':
        chart = gcp_sql_pull(command='daily')
    elif command == '15days':
        chart = gcp_sql_pull(command='accumulated')
        # Convert view_count into int
        chart['view_count'] = chart['view_count'].astype(str).astype(int)
        # Sort by view_count
        chart = chart.tail(300).sort_values('view_count', ascending = False)
        chart = tf_conversion(data=chart)
        # With duplicated video ids & without (removed) duplicated video ids
        chart_repeat = chart
        chart = chart.drop_duplicates(subset='동영상아이디', keep="last")

    else:
        chart = gcp_sql_pull(command='accumulated')
        # Convert view_count into int
        chart['view_count'] = chart['view_count'].astype(str).astype(int)
        # Sort by view_count
        chart = chart.tail(600).sort_values('view_count', ascending = False)
        chart = tf_conversion(chart)
        # With duplicated video ids & without (removed) duplicated video ids
        chart_repeat = chart
        chart = chart.drop_duplicates(subset='동영상아이디', keep="last")

    # ======================  KIWI Tokeanization & Wordcloud  ====================== #
    # Kiwi Tokenization
    kiwi = Kiwi()
    text = " ".join(chart['동영상'])
    text_list = kiwi.tokenize(text)
    # Form & Tag split
    split_form = pd.DataFrame([item[0] for item in text_list])
    split_form.rename(columns = {0 : 'form'}, inplace = True)
    split_tag = pd.DataFrame([item[1] for item in text_list])
    split_tag.rename(columns={0: 'tag'}, inplace=True)
    text_token = pd.concat([split_form, split_tag], axis=1)

    # Remove tags
    tag = ['NNG', 'NNP', 'NNB', 'NR', 'NP']
    text_token = text_token[text_token['tag'].isin(tag)]
    # Remove manual token
    removal_manual = ['ᆷ', 'ᆼ', 'ᆫ', 'ᆸ니다', '에', '는', '의', '이', '를', '하', '을', '한', '가', '과', '다', '지', '고', '로', '은']
    text_token = text_token[~text_token['form'].isin(removal_manual)]
    # Clean text
    text_token["form"] = text_token["form"].apply(
        lambda x: remove_cleantext(x)
    )
    # Choose nouns > 1
    text_token = text_token[
        text_token['form'].apply(lambda x: len(x) > 1)
    ]

    # Most common words
    counts = Counter(text_token['form'])
    tags = counts.most_common(50)
    # Wordcloud
    wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white", font_path="static/gulim.ttc").generate_from_frequencies(dict(tags))
    wordcloud.to_file("static/images/wc_01.png")

    # ==========================  With duplicated video ids   ========================== #
    # 1. Best Channel - Connects to Youtube API for Channel information
    top_channel = chart_repeat[chart_repeat['조회수'] == chart_repeat['조회수'].max()]

    # 2. Calculate average view counts
    tf_sum = chart_repeat.groupby(['차트일자'])[['조회수', '좋아요수', '댓글수']].sum().sort_values('차트일자', ascending=True).reset_index()
    tf_avg = chart_repeat.groupby(['차트일자'])[['조회수', '좋아요수', '댓글수']].mean().sort_values('차트일자', ascending=True).reset_index()
    tf_sum_category = chart_repeat.groupby(['차트일자', '카테고리'])[['조회수', '좋아요수', '댓글수']].sum().sort_values('차트일자', ascending=True).reset_index()

    # 3. Best Channel in Time Period - Connects to Youtube API for Channel information
    top_cate_channel = chart_repeat[chart_repeat['조회수'] == chart_repeat.groupby('차트일자')['조회수'].transform('max')].sort_values('차트일자', ascending=True).reset_index()
    del top_cate_channel["index"]

    # ====================== YOUTUBE_CHANNEL_INFO : Top Channel  ====================== #
    # Running loop to get Channel_ID from 'df_channel_search'
    df_channel_info = pd.DataFrame()

    for id in top_channel['채널아이디']:
        try:
            # YOUTUBE_CHANNEL_INFORMATION
            res_channel = youtube.channels().list(part=['snippet', 'statistics', 'contentDetails'],
                                                  id=id).execute()
            dataframe = json_normalize(res_channel['items'][0])
            df_channel_info = df_channel_info.append(dataframe)
        except:
            print('Channel ID : ' + id + ' has failed to open API connection')

    # Select Columns
    df_channel_info = df_channel_info[['id',
                                       'snippet.thumbnails.medium.url',
                                       #'snippet.customUrl',
                                       'snippet.publishedAt',
                                       'statistics.viewCount',
                                       'statistics.subscriberCount',
                                       'statistics.videoCount']]

    # Rename Columns
    df_channel_info.rename(columns={'id': '채널아이디',
                                    'snippet.thumbnails.medium.url': '썸네일',
                                    #'snippet.customUrl': 'custom_url',
                                    'snippet.publishedAt': '채널개설날짜',
                                    'statistics.viewCount': '채널총조회수',
                                    'statistics.subscriberCount': '채널구독수',
                                    'statistics.videoCount': '채널비디오수'
                                    }, inplace=True)
    df_channel_info['채널개설날짜'] = pd.to_datetime(df_channel_info['채널개설날짜'], infer_datetime_format=True)
    df_channel_info['채널개설날짜'] = df_channel_info['채널개설날짜'].dt.strftime("%Y-%m-%d")

    # Merge with video information
    tf_channel = pd.DataFrame()
    tf_channel = top_channel.merge(df_channel_info, how='inner', on='채널아이디')

    # ====================== YOUTUBE_CHANNEL_INFO : Top Channel in time period  ====================== #
    # Running loop to get Channel_ID from 'df_channel_search'
    df_channel_info = pd.DataFrame()

    for id in top_cate_channel['채널아이디']:
        try:
            # YOUTUBE_CHANNEL_INFORMATION
            res_channel = youtube.channels().list(part=['snippet', 'statistics', 'contentDetails'],
                                                  id=id).execute()
            dataframe = json_normalize(res_channel['items'][0])
            df_channel_info = df_channel_info.append(dataframe)
            print('Channel ID API Connection : ' + id + '  has been successfully completed')
        except:
            print('Channel ID : ' + id + ' has failed to open API connection')

    # Select Columns
    df_channel_info = df_channel_info[['id',
                                       'snippet.thumbnails.medium.url',
                                       #'snippet.customUrl',
                                       'snippet.publishedAt',
                                       'statistics.viewCount',
                                       'statistics.subscriberCount',
                                       'statistics.videoCount']]

    # Rename Columns
    df_channel_info.rename(columns={'id': '채널아이디',
                                    'snippet.thumbnails.medium.url': '썸네일',
                                    #'snippet.customUrl': 'custom_url',
                                    'snippet.publishedAt': '채널개설날짜',
                                    'statistics.viewCount': '채널총조회수',
                                    'statistics.subscriberCount': '채널구독수',
                                    'statistics.videoCount': '채널비디오수'
                                    }, inplace=True)
    df_channel_info['채널개설날짜'] = pd.to_datetime(df_channel_info['채널개설날짜'], infer_datetime_format=True)
    df_channel_info['채널개설날짜'] = df_channel_info['채널개설날짜'].dt.strftime("%Y-%m-%d")

    # Merge with video information
    tf_list_channel = pd.DataFrame()
    tf_list_channel = top_cate_channel.merge(df_channel_info, how='inner', on='채널아이디').drop_duplicates().reset_index()
    tf_list_channel = tf_list_channel[['차트일자', '동영상', '채널명', '썸네일', '조회수', '좋아요수', '댓글수', '카테고리']]
    tf_list_channel.index = tf_list_channel.index+1

    return tf_list_channel, tf_channel, tf_sum, tf_avg, tf_sum_category

def channel_search(chanel_name):
    """
    SECTION : channel, comment
    DESCRIPTION : Print channel information using the key. The output is only stored in dataframe.
    USAGE : User can check statistics figures in each relevant channel and finally uses ChanneId.
    """
    pd.options.mode.chained_assignment = None  # Off warning messages, default='warn'
    starttime = datetime.now()
    print(starttime)

    # ======================== GOOGLE SECRET Reading ========================= #
    connection_name, query_string, db_name, db_user, db_password, driver_name, public_ip, service_key = keys()
    youtube = build('youtube', 'v3', developerKey=service_key)
    # ======================================================================== #

    try:
        # YOUTUBE_CHANNEL_SEARCH
        res_channel_search = youtube.search().list(part='snippet', type='channel', regionCode='KR', q=chanel_name,
                                                   order='videoCount', maxResults=20).execute()
        df_channel_search = json_normalize(res_channel_search['items'])
        print('Channel Search API Connection : ' + chanel_name + '  has been successfully completed')
    except:
        print('Channel Search : ' + chanel_name + ' has failed to open API connection')

    # ====================== YOUTUBE_CHANNEL_SEARCH : Data Mapping  ====================== #

    # Select Columns
    df_channel_search = df_channel_search[['id.channelId', 'snippet.publishedAt', 'snippet.title']]

    # Rename Columns
    df_channel_search.rename(columns={'id.channelId': 'channel_id',
                                      'snippet.publishedAt': 'published_at',
                                      'snippet.title': 'channel_title',
                                      'snippet.categoryId': 'category_id'
                                      }, inplace=True)

    # ======================  Retrieving API : YOUTUBE_CHANNEL_INFO  ====================== #

    df_channel_info = pd.DataFrame()

    # Running loop to get Channel_ID from 'df_channel_search'
    for id in df_channel_search['channel_id']:
        try:
            # YOUTUBE_CHANNEL_INFORMATION
            res_channel = youtube.channels().list(part=['snippet', 'statistics', 'contentDetails'],
                                                  id=id).execute()
            df = json_normalize(res_channel['items'][0])
            df_channel_info = df_channel_info.append(df)
            print('Channel ID API Connection : ' + id + '  has been successfully completed')
        except:
            print('Channel ID : ' + id + ' has failed to open API connection')

    # ====================== YOUTUBE_CHANNEL_INFO : Data Mapping  ====================== #
    # Select Columns
    df_channel_info = df_channel_info[['id',
                                       #'snippet.customUrl',
                                       'statistics.viewCount',
                                       'statistics.subscriberCount',
                                       'statistics.videoCount']]

    # Rename Columns
    df_channel_info.rename(columns={'id': 'channel_id',
                                    #'snippet.customUrl': 'custom_url',
                                    'statistics.viewCount': 'view_count',
                                    'statistics.subscriberCount': 'subscriber_count',
                                    'statistics.videoCount': 'video_count'
                                    }, inplace=True)

    # ====================== MERGE : df_channel_search & df_channel_info ====================== #
    # Merge & Rename columns
    df_channel = df_channel_search.merge(df_channel_info, how='inner', on='channel_id')
    # ========================================================================================= #

    # print(df_chanel.to_markdown())  # Choose index=TRUE / index=FALSE
    endtime = datetime.now()
    print(endtime)
    timetaken = endtime - starttime
    print('Time taken : ' + timetaken.__str__())

    return df_channel

def pickle_videos(type, channel_id):
    """
    SECTION : channel, videos, comment
    DESCRIPTION 1: This function uses youtube.search() and .videos(). Hence, all videos in playlist from the channel may not be extracted.
    DESCRIPTION 2: Create pickle files for all video information from the channel ID key in by users.
    USAGE 1 [type='user'] : User can extract all video IDs from the channel with some basic statistic figures.
    USAGE 2 [type='sample'] : User can extract only limited video (Default : 50 videos)
    """
    # ====================== Setup ====================== #
    pd.options.mode.chained_assignment = None  # Off warning messages, default='warn'
    starttime = datetime.now()
    print(starttime)
    dictionary = {0: 'wiki_category'}
    dictionary_list = list(dictionary.values())

    # ======================== GOOGLE SECRET Reading ========================= #
    connection_name, query_string, db_name, db_user, db_password, driver_name, public_ip, service_key = keys()
    youtube = build('youtube', 'v3', developerKey=service_key)
    # ======================================================================== #

    # STEP1 : Search VIDEO_ID using Search()
    # DESCRIPTION : 'sample' can extract one page token, whereas 'user' can extract all page tokens
    if type == 'user':
        print('Video Item : user is selected. All video ids will be extracted in page tokens')
        try:
            videos = []
            next_page_token = None
            while 1:
                res_video_id = youtube.search().list(part='id', type='video', channelId=channel_id, maxResults=50,
                                                     pageToken=next_page_token).execute()
                videos += res_video_id['items']
                next_page_token = res_video_id.get('nextPageToken')
                if next_page_token is None:
                    break
            # Extract VIDEO_IDs only
            video_ids = list(map(lambda x: x['id']['videoId'], videos))
            print('Video Item : video IDs has been successfully completed')
        except:
            print('Video Item : video IDs has failed to pull video IDs')

    else:
        print('Video Item : sample is selected. Video ids will be extracted in one page token only')
        try:
            videos = []
            res_video_id = youtube.search().list(part='id', type='video', channelId=channel_id, maxResults=50).execute()
            videos = res_video_id['items']
            # Extract VIDEO_IDs only
            video_ids = list(map(lambda x: x['id']['videoId'], videos))
            print('Video Item : video IDs has been successfully completed')
        except:
            print('Video Item : video IDs has failed to pull video IDs')

    # STEP2 : Use VIDEO_ID to extract VIDEO_INFORMATION
    try:
        df_video_info = pd.DataFrame()
        for id in video_ids:
            res_video_info = youtube.videos().list(part=['snippet', 'statistics', 'status', 'topicDetails'],
                                                   id=id,
                                                   regionCode='KR').execute()
            df = json_normalize(res_video_info['items'])
            df_video_info = df_video_info.append(df)
        print('Collecting Video Information has been successfully completed')
    except:
        print('Collecting Video Information has failed to extract')

    # ====================== YOUTUBE_CHANNEL_INFO : Data Mapping  ======================#
    # Select Columns
    df_video_info = df_video_info[[
        'id',
        'snippet.title', 'snippet.publishedAt',
        'statistics.viewCount', 'statistics.likeCount', 'statistics.favoriteCount',
        'statistics.commentCount',  # video().list(part='statistics')
        'topicDetails.topicCategories',  # video().list(part='topicDetails')
        'status.madeForKids']]

    # Rename Columns
    df_video_info.rename(columns={'id': 'video_id',
                                  'snippet.title': 'video_title',
                                  'snippet.publishedAt': 'published_at',
                                  'statistics.viewCount': 'view_count',
                                  'statistics.likeCount': 'like_count',
                                  'statistics.favoriteCount': 'favorite_count',
                                  'statistics.commentCount': 'comment_count',
                                  'topicDetails.topicCategories': 'topic_categories',
                                  'status.madeForKids': 'for_kids'
                                  }, inplace=True)

    # Reset Index
    df_video_info = df_video_info.reset_index()

    # Split TopicCategories URL
    catrgory_split = df_video_info['topic_categories']
    catrgory_split = pd.DataFrame(catrgory_split)
    catrgory_split = catrgory_split['topic_categories'].apply(pd.Series).rename(columns=dictionary)

    # Filter columns based on the length
    dictionary_list = dictionary_list[0:len(catrgory_split.columns)]

    # Split WIKI_URL and pick up the last word (Filtering category)
    for i in range(len(catrgory_split.columns)):
        df = catrgory_split.iloc[:, i].str.split('/').apply(pd.Series).iloc[:, -1]
        df.columns = [i]
        catrgory_split[i] = df

    # Remove & Rename columns
    catrgory_split.drop(dictionary_list, axis=1, inplace=True)
    catrgory_split = catrgory_split.rename(columns=dictionary)
    catrgory_split = catrgory_split['wiki_category']

    # Merge & Rename columns
    df_video_info = df_video_info.merge(catrgory_split, left_index=True, right_index=True)
    del df_video_info['topic_categories']
    del df_video_info['index']
    print('Video Information : Data mapping has been successfully completed')

    # ====================== Export to Pickle & Read  ====================== #
    if type == 'user':
        pickle_replace(name='video_user_info', file=df_video_info)
        video_info = read_pickle('video_user_info.pkl')
    else:
        pickle_replace(name='video_sample_info', file=df_video_info)
        video_info = read_pickle('video_sample_info.pkl')
    # ====================================================================== #

    endtime = datetime.now()
    print(endtime)
    timetaken = endtime - starttime
    print('Time taken : ' + timetaken.__str__())

    return video_info

def pickle_videos_filter(type, find):
    """
    SECTION : channel, videos, comment
    DESCRIPTION : Find substrings from the channel title to narrow down the search results.
    USAGE : Extraction of the comments takes less number of videos after the tile search.
    """
    # ====================== Setup ====================== #
    pd.options.mode.chained_assignment = None  # Off warning messages, default='warn'
    # =================================================== #

    # ====================== LOAD & FILTER PICKLES ====================== #
    # Choose pickle files based on the type condition
    if type == 'user':
        load_dir_path = 'Pickle/video_user_info.pkl'
        load_filename = 'video_user_info.pkl'
        save_name = 'video_filter_user_info'
        save_filename = 'video_filter_user_info.pkl'
    else:
        load_dir_path = 'Pickle/video_sample_info.pkl'
        load_filename = 'video_sample_info.pkl'
        save_name = 'video_filter_sample_info'
        save_filename = 'video_filter_sample_info.pkl'

    # Load video_info
    if os.path.exists(load_dir_path):
        video_info = read_pickle(load_filename)
        # Find substrings
        video_info_filter = video_info.loc[video_info['video_title'].str.contains(find, case=False)].reset_index(
            drop=True)
        # Export to Pickle & Read
        pickle_replace(name=save_name, file=video_info_filter)
        video_info_filter = read_pickle(save_filename)
        print('Video Filtering : Filtering titles from video_information has been successfully completed')
    else:
        print(load_dir_path + ' is not existing in the path')

    return video_info_filter

def pickle_videos_comments(type, option):
    """
    SECTION : channel, videos, comment
    DESCRIPTION : Extract comments from video_info_subs
    USAGE : The comment will be used to have sentiment analysis
    """
    # ====================== Setup ====================== #
    pd.options.mode.chained_assignment = None  # Off warning messages, default='warn'
    starttime = datetime.now()
    print(starttime)

    # Choose pickle files based on the type condition
    if type == 'user':
        load_filename = 'video_filter_user_info.pkl'
    else:
        load_filename = 'video_filter_sample_info.pkl'

    # ====================== Running ====================== #
    if os.path.exists('Pickle/' + load_filename):

        # ====================== Read Pickles (VideoIDs) ====================== #
        video_info_filter = read_pickle(load_filename)
        # ======================== GOOGLE SECRET Reading ========================= #
        connection_name, query_string, db_name, db_user, db_password, driver_name, public_ip, service_key = keys()
        youtube = build('youtube', 'v3', developerKey=service_key)
        # ======================================================================== #

        # ====================== Retrieving API and store in DF  ====================== #
        print('Video Comments : Starting to extract comments in the filtered video ids')
        df_comment = pd.DataFrame()
        try:
            for id in video_info_filter.video_id:
                # CommentThread based on relevance filter
                res_rel_comments = youtube.commentThreads().list(part='snippet', videoId=id, order='relevance',
                                                                 maxResults=25).execute()
                df = json_normalize(res_rel_comments['items'])
                df_comment = df_comment.append(df)
                # CommentThread based on order filter
                # res_ord_comments = youtube.commentThreads().list(part='snippet', videoId=id, order='time', maxResults=25).execute()
                # df = json_normalize(res_ord_comments['items'])
                # df_comment = df_comment.append(df, ignore_index=True)
            # Reset_index()
            df_comment = df_comment.reset_index(drop=True)
            print(str(len(df_comment)) + ' Comments has been successfully loaded')
        except:
            print('Video Comments : Comments has failed to load')

        # ====================== YOUTUBE_COMMENT : Data Mapping  ====================== #

        # Select Columns
        df_comment = df_comment[[
            'id',
            'snippet.topLevelComment.snippet.textOriginal',
            'snippet.topLevelComment.snippet.authorDisplayName',
            'snippet.topLevelComment.snippet.likeCount',
            'snippet.topLevelComment.snippet.publishedAt',
            'snippet.totalReplyCount'
        ]]

        # Rename Columns
        df_comment.rename(columns={'id': 'comment_id',
                                   'snippet.topLevelComment.snippet.textOriginal': 'comment',
                                   'snippet.topLevelComment.snippet.authorDisplayName': 'author',
                                   'snippet.topLevelComment.snippet.likeCount': 'like_count',
                                   'snippet.topLevelComment.snippet.publishedAt': 'published_at',
                                   'snippet.totalReplyCount': 'reply_count',
                                   }, inplace=True)

        # ====================== Export to Pickle & Read ====================== #
        # Read pickle files
        pickle_replace(name='video_comment', file=df_comment)
        video_comment = read_pickle('video_comment.pkl')
        print('Video Comments : Extracting comments from the filtered video ids has been successfully completed')

        # Delete Path
        if option == 'delete':
            os.remove('Pickle/' + load_filename)
            print("Delete [option] : Filtered video pickle file has been successfully deleted")
        else:
            print("Delete [none] : There are no changes in Pickle files")
        # ===================================================================== #

        return video_comment
    else:
        print("Warning : The file directory is not existing. Please execute channel search & video id extraction before you run this code")

    endtime = datetime.now()
    print(endtime)
    timetaken = endtime - starttime
    print('Time taken : ' + timetaken.__str__())

def globals_videos(type, channel_id):
    """
    SECTION : channel, videos, comment
    DESCRIPTION 1: This function uses youtube.search() and .videos(). Hence, all videos in playlist from the channel may not be extracted.
    DESCRIPTION 2: Create pickle files for all video information from the channel ID key in by users.
    USAGE 1 [type='user'] : User can extract all video IDs from the channel with some basic statistic figures.
    USAGE 2 [type='sample'] : User can extract only limited video (Default : 50 videos)
    """
    # ====================== Setup ====================== #
    pd.options.mode.chained_assignment = None  # Off warning messages, default='warn'
    starttime = datetime.now()
    print(starttime)
    dictionary = {0: 'wiki_category'}
    dictionary_list = list(dictionary.values())

    # ======================== GOOGLE SECRET Reading ========================= #
    connection_name, query_string, db_name, db_user, db_password, driver_name, public_ip, service_key = keys()
    youtube = build('youtube', 'v3', developerKey=service_key)
    # ======================================================================== #

    # STEP1 : Search VIDEO_ID using Search()
    # DESCRIPTION : 'sample' can extract one page token, whereas 'user' can extract all page tokens
    if type == 'user':
        print('Video Item : user is selected. All video ids will be extracted in page tokens')
        try:
            videos = []
            next_page_token = None
            while 1:
                res_video_id = youtube.search().list(part='id', type='video', channelId=channel_id, maxResults=50,
                                                     pageToken=next_page_token).execute()
                videos += res_video_id['items']
                next_page_token = res_video_id.get('nextPageToken')
                if next_page_token is None:
                    break
            # Extract VIDEO_IDs only
            video_ids = list(map(lambda x: x['id']['videoId'], videos))
            print('Video Item : video IDs has been successfully completed')
        except:
            print('Video Item : video IDs has failed to pull video IDs')

    else:
        print('Video Item : sample is selected. Video ids will be extracted in one page token only')
        try:
            videos = []
            res_video_id = youtube.search().list(part='id', type='video', channelId=channel_id, maxResults=50).execute()
            videos = res_video_id['items']
            # Extract VIDEO_IDs only
            video_ids = list(map(lambda x: x['id']['videoId'], videos))
            print('Video Item : video IDs has been successfully completed')
        except:
            print('Video Item : video IDs has failed to pull video IDs')

    # STEP2 : Use VIDEO_ID to extract VIDEO_INFORMATION
    try:
        df_video_info = pd.DataFrame()
        for id in video_ids:
            res_video_info = youtube.videos().list(part=['snippet', 'statistics', 'status', 'topicDetails'],
                                                   id=id,
                                                   regionCode='KR').execute()
            df = json_normalize(res_video_info['items'])
            df_video_info = df_video_info.append(df)
        print('Collecting Video Information has been successfully completed')
    except:
        print('Collecting Video Information has failed to extract')

    # ====================== YOUTUBE_CHANNEL_INFO : Data Mapping  ======================#
    # Select Columns
    df_video_info = df_video_info[[
        'id',
        'snippet.title', 'snippet.publishedAt',
        'statistics.viewCount', 'statistics.likeCount', 'statistics.favoriteCount',
        'statistics.commentCount',  # video().list(part='statistics')
        'topicDetails.topicCategories',  # video().list(part='topicDetails')
        'status.madeForKids']]

    # Rename Columns
    df_video_info.rename(columns={'id': 'video_id',
                                  'snippet.title': 'video_title',
                                  'snippet.publishedAt': 'published_at',
                                  'statistics.viewCount': 'view_count',
                                  'statistics.likeCount': 'like_count',
                                  'statistics.favoriteCount': 'favorite_count',
                                  'statistics.commentCount': 'comment_count',
                                  'topicDetails.topicCategories': 'topic_categories',
                                  'status.madeForKids': 'for_kids'
                                  }, inplace=True)

    # Reset Index
    df_video_info = df_video_info.reset_index()

    # Split TopicCategories URL
    catrgory_split = df_video_info['topic_categories']
    catrgory_split = pd.DataFrame(catrgory_split)
    catrgory_split = catrgory_split['topic_categories'].apply(pd.Series).rename(columns=dictionary)

    # Filter columns based on the length
    dictionary_list = dictionary_list[0:len(catrgory_split.columns)]

    # Split WIKI_URL and pick up the last word (Filtering category)
    for i in range(len(catrgory_split.columns)):
        df = catrgory_split.iloc[:, i].str.split('/').apply(pd.Series).iloc[:, -1]
        df.columns = [i]
        catrgory_split[i] = df

    # Remove & Rename columns
    catrgory_split.drop(dictionary_list, axis=1, inplace=True)
    catrgory_split = catrgory_split.rename(columns=dictionary)
    catrgory_split = catrgory_split['wiki_category']

    # Merge & Rename columns
    vid = df_video_info.merge(catrgory_split, left_index=True, right_index=True)
    del vid['topic_categories']
    del vid['index']
    print('Video Information : Data mapping has been successfully completed')

    endtime = datetime.now()
    print(endtime)
    timetaken = endtime - starttime
    print('Time taken : ' + timetaken.__str__())

    return vid

def globals_videos_filter(find):
    """
    SECTION : channel, videos, comment
    DESCRIPTION : Find substrings from the channel title to narrow down the search results.
    USAGE : Extraction of the comments takes less number of videos after the tile search.
    """
    # ====================== Setup ====================== #
    pd.options.mode.chained_assignment = None  # Off warning messages, default='warn'
    # =================================================== #

    # ====================== Load & Return Globals() ====================== #
    # Load video_info
    if 'vid' in globals():
        # Load the global file in the current running environment
        vid = globals()['vid']
        # Find substrings
        vid_filter = vid.loc[vid['video_title'].str.contains(find, case=False)].reset_index(
            drop=True)
        print('Video Filtering : Filtering titles from video_information has been successfully completed')
    else:
        print('vid globals() is not existing in the path')

    return vid_filter

def globals_videos_comments(option):
    """
    SECTION : channel, videos, comment
    DESCRIPTION : Extract comments from video_info_subs
    USAGE : The comment will be used to have sentiment analysis
    """
    # ====================== Setup ====================== #
    pd.options.mode.chained_assignment = None  # Off warning messages, default='warn'
    starttime = datetime.now()
    print(starttime)

    # ====================== Running ====================== #
    if 'vid_filter' in globals():
        # Load the global file in the current running environment
        vid_filter = globals()['vid_filter']
        load_filename = 'vid_filter'

        # ======================== GOOGLE SECRET Reading ========================= #
        connection_name, query_string, db_name, db_user, db_password, driver_name, public_ip, service_key = keys()
        youtube = build('youtube', 'v3', developerKey=service_key)
        # ======================================================================== #

        # ====================== Retrieving API and store in DF  ====================== #
        print('Video Comments : Starting to extract comments in the filtered video ids')
        df_comment = pd.DataFrame()
        try:
            for id in vid_filter.video_id:
                # CommentThread based on relevance filter
                res_rel_comments = youtube.commentThreads().list(part='snippet', videoId=id, order='relevance',
                                                                 maxResults=25).execute()
                df = json_normalize(res_rel_comments['items'])
                df_comment = df_comment.append(df)
                # CommentThread based on order filter
                # res_ord_comments = youtube.commentThreads().list(part='snippet', videoId=id, order='time', maxResults=25).execute()
                # df = json_normalize(res_ord_comments['items'])
                # df_comment = df_comment.append(df, ignore_index=True)
            # Reset_index()
            df_comment = df_comment.reset_index(drop=True)
            print(str(len(df_comment)) + ' Comments has been successfully loaded')
        except:
            print('Video Comments : Comments has failed to load')

        # ====================== YOUTUBE_COMMENT : Data Mapping  ====================== #

        # Select Columns
        vid_comments = df_comment[[
            'id',
            'snippet.topLevelComment.snippet.textOriginal',
            'snippet.topLevelComment.snippet.authorDisplayName',
            'snippet.topLevelComment.snippet.likeCount',
            'snippet.topLevelComment.snippet.publishedAt',
            'snippet.totalReplyCount'
        ]]

        # Rename Columns
        vid_comments.rename(columns={'id': 'comment_id',
                                   'snippet.topLevelComment.snippet.textOriginal': 'comment',
                                   'snippet.topLevelComment.snippet.authorDisplayName': 'author',
                                   'snippet.topLevelComment.snippet.likeCount': 'like_count',
                                   'snippet.topLevelComment.snippet.publishedAt': 'published_at',
                                   'snippet.totalReplyCount': 'reply_count',
                                   }, inplace=True)

        # ====================== Load & Return Globals() ====================== #
        # Check
        print('Video Comments : Extracting comments from the filtered video ids has been successfully completed')

        # Delete Path
        if option == 'delete':
            del globals()[load_filename]
            print("Delete [option] : Filtered Globals() variable has been successfully deleted")
        else:
            print("Delete [none] : There are no changes in Globals() variable in the running environment")

        return vid_comments
        # ===================================================================== #
    else:
        print("Warning : The file directory is not existing. Please execute channel search & video id extraction before you run this code")

    endtime = datetime.now()
    print(endtime)
    timetaken = endtime - starttime
    print('Time taken : ' + timetaken.__str__())

def channel_playlist(channel_id):
    """
    %%% WARNING : Channel with massive videos may result to reach your quota given by Google %%%
    SECTION : channel, playlist, comment
    DESCRIPTION : Create pickle files for all video information from the playlist ID key in by users.
    USAGE : User can have a list of video IDs from the channel with some basic statistic figures.
    """
    # ====================== Setup ====================== #
    pd.options.mode.chained_assignment = None  # Off warning messages, default='warn'
    starttime = datetime.now()
    print(starttime)
    dictionary = {0: 'wiki_category_1', 1: 'wiki_category_2', 2: 'wiki_category_3', 3: 'wiki_category_4',
                  4: 'wiki_category_5', 5: 'wiki_category_6'}
    dictionary_list = list(dictionary.values())

    # ======================== GOOGLE SECRET Reading ========================= #
    connection_name, query_string, db_name, db_user, db_password, driver_name, public_ip, service_key = keys()
    youtube = build('youtube', 'v3', developerKey=service_key)
    # ======================================================================== #

    # STEP1 : Collect PLAYLIST_ID using CHANNELS()
    try:
        res_playlist = youtube.channels().list(part='contentDetails', id=channel_id).execute()
        playlist_id = res_playlist['items'][0]['contentDetails']['relatedPlaylists']['uploads']
        print('Channel PlaylistID API Connection has been successfully completed')
    except:
        print('Channel PlaylistID API Connection has failed to initiate')

    # STEP2 : Get VIDEO_ID from PLAYLIST_ID
    try:
        videos = []
        next_page_token = None
        while 1:
            res_video_id = youtube.playlistItems().list(part='snippet', playlistId=playlist_id, maxResults=50,
                                                        pageToken=next_page_token).execute()
            videos += res_video_id['items']
            next_page_token = res_video_id.get('nextPageToken')
            if next_page_token is None:
                break
        # Extract VIDEO_IDs only
        video_ids = list(map(lambda x: x['snippet']['resourceId']['videoId'], videos))
        print('Playlist Item : video IDs has been successfully completed')
    except:
        print('Playlist Item : video IDs has failed to pull video IDs')

    # STEP3 : Use VIDEO_ID to extract VIDEO_INFORMATION
    try:
        df_video_info = pd.DataFrame()
        for id in video_ids:
            res_video_info = youtube.videos().list(part=['snippet', 'statistics', 'status', 'topicDetails'],
                                                   id=id,
                                                   regionCode='KR').execute()
            df = json_normalize(res_video_info['items'])
            df_video_info = df_video_info.append(df)
        print('Collecting Video Information has been successfully completed')
    except:
        print('Collecting Video Information has failed to extract')

    # ====================== YOUTUBE_CHANNEL_INFO : Data Mapping  ====================== #
    # Select Columns
    df_video_info = df_video_info[[
        'id',
        'snippet.title', 'snippet.publishedAt',
        'statistics.viewCount', 'statistics.likeCount', 'statistics.favoriteCount',
        'statistics.commentCount',  # video().list(part='statistics')
        'topicDetails.topicCategories',  # video().list(part='topicDetails')
        'status.madeForKids']]

    # Rename Columns
    df_video_info.rename(columns={'id': 'video_id',
                                  'snippet.title': 'video_title',
                                  'snippet.publishedAt': 'published_at',
                                  'statistics.viewCount': 'view_count',
                                  'statistics.likeCount': 'like_count',
                                  'statistics.favoriteCount': 'favorite_count',
                                  'topicDetails.topicCategories': 'topic_categories',
                                  'status.madeForKids': 'for_kids'
                                  }, inplace=True)

    # Reset Index
    df_video_info = df_video_info.reset_index()

    # Split TopicCategories URL
    catrgory_split = df_video_info['topic_categories']
    catrgory_split = pd.DataFrame(catrgory_split)
    catrgory_split = catrgory_split['topic_categories'].apply(pd.Series).rename(columns=dictionary)

    # Filter columns based on the length
    dictionary_list = dictionary_list[0:len(catrgory_split.columns)]

    # Split WIKI_URL and pick up the last word (Filtering category)
    for i in range(len(catrgory_split.columns)):
        df = catrgory_split.iloc[:, i].str.split('/').apply(pd.Series).iloc[:, -1]
        df.columns = [i]
        catrgory_split[i] = df

    # Remove & Rename columns
    catrgory_split.drop(dictionary_list, axis=1, inplace=True)
    catrgory_split = catrgory_split.rename(columns=dictionary)

    # Merge & Rename columns
    df_video_info = df_video_info.merge(catrgory_split, left_index=True, right_index=True)
    del df_video_info['topic_categories']
    del df_video_info['index']
    print('Youtube Video Information : Data mapping has been successfully completed')

    # ====================== Export to Pickle & Read  ====================== #
    pickle_replace(name='playlist_info', file=df_video_info)
    playlist_info = read_pickle('playlist_info.pkl')
    # ====================================================================== #

    endtime = datetime.now()
    print(endtime)
    timetaken = endtime - starttime
    print('Time taken : ' + timetaken.__str__())

    return playlist_info

def channel_playlist_filter(find):
    """
    SECTION : channel, playlist, comment
    DESCRIPTION : Find substrings from the channel title to narrow down the search results.
    USAGE : Extraction of the comments takes less number of videos after the tile search.
    """
    # ====================== Setup ====================== #
    pd.options.mode.chained_assignment = None  # Off warning messages, default='warn'
    # =================================================== #

    # ====================== LOAD & FILTER PICKLES ====================== #
    # Load playlist_info
    if os.path.exists('Pickle/playlist_info.pkl'):
        playlist_info = read_pickle('playlist_info.pkl')
        # Find substrings
        playlist_info_filter = playlist_info.loc[
            playlist_info['VideoTitle'].str.contains(find, case=False)].reset_index(drop=True)
        # Export to Pickle & Read
        pickle_replace(name='playlist_info_filter', file=playlist_info_filter)
        playlist_info_filter = read_pickle('playlist_info_filter.pkl')
        print('Title Searching : filtering titles from playlist_info has been successfully completed')
    else:
        print('playlist_info.pkl is not exist in the path')

    return playlist_info_filter

def channel_playlist_comments():
    """
    SECTION : channel, playlist, comment
    DESCRIPTION : Extract comments from playlist_info_sub
    USAGE : The comment will be used to have sentiment analysis
    """
    # ====================== Setup ====================== #
    pd.options.mode.chained_assignment = None  # Off warning messages, default='warn'
    starttime = datetime.now()
    print(starttime)

    # ====================== Read Pickles (VideoIDs) ====================== #
    playlist_info_filter = read_pickle('playlist_info_filter.pkl')
    # ===================================================================== #

    # ======================== GOOGLE SECRET Reading ========================= #
    query_string, db_name, db_user, db_password, driver_name, service_key = keys()
    youtube = build('youtube', 'v3', developerKey=service_key)
    # ======================================================================== #

    # ====================== Retrieving API and store in DF  ====================== #
    print('Playlist Comments : Starting to extract comments in the filtered video ids')
    df_comment = pd.DataFrame()
    try:
        for id in playlist_info_filter.video_id:
            # CommentThread based on relevance filter
            res_rel_comments = youtube.commentThreads().list(part='snippet', videoId=id, order='relevance',
                                                             maxResults=25).execute()
            df = json_normalize(res_rel_comments['items'])
            df_comment = df_comment.append(df)
            # CommentThread based on order filter
            res_ord_comments = youtube.commentThreads().list(part='snippet', videoId=id, order='time',
                                                             maxResults=25).execute()
            df = json_normalize(res_ord_comments['items'])
            df_comment = df_comment.append(df, ignore_index=True)
        # Reset_index()
        df_comment = df_comment.reset_index(drop=True)
        print(str(len(df_comment)) + ' Comments has been successfully loaded')
    except:
        print('Comments has failed to load')

    # ====================== YOUTUBE_COMMENT : Data Mapping  ====================== #

    # Select Columns
    df_comment = df_comment[[
        'id',
        'snippet.topLevelComment.snippet.textOriginal',
        'snippet.topLevelComment.snippet.authorDisplayName',
        'snippet.topLevelComment.snippet.likeCount',
        'snippet.topLevelComment.snippet.publishedAt',
        'snippet.totalReplyCount'
    ]]

    # Rename Columns
    df_comment.rename(columns={'id': 'comment_id',
                               'snippet.topLevelComment.snippet.textOriginal': 'comment',
                               'snippet.topLevelComment.snippet.authorDisplayName': 'author',
                               'snippet.topLevelComment.snippet.likeCount': 'like_count',
                               'snippet.topLevelComment.snippet.publishedAt': 'published_at',
                               'snippet.totalReplyCount': 'reply_count',
                               }, inplace=True)

    # ====================== Export to Pickle & Read ====================== #
    pickle_replace(name='playlist_comment', file=df_comment)
    playlist_comment = read_pickle('playlist_comment.pkl')
    # ===================================================================== #

    print('Playlist Comments : Extracting comments from the filtered video ids has been successfully completed')
    endtime = datetime.now()
    print(endtime)
    timetaken = endtime - starttime
    print('Time taken : ' + timetaken.__str__())

    return playlist_comment

# ====================== RETRIEVING DATA FROM YOUTUBE API ====================== #


