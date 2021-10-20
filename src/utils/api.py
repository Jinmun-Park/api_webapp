# ====================== LIBRARY SETUP ====================== #
# API READER SETUP
from googleapiclient.discovery import build #GOOGLE API
from urllib.request import Request, urlopen
from urllib.parse import urlencode, quote_plus, unquote
from pandas import json_normalize
import json
import requests #XML DECODING
import xmltodict
# YAML READER SETUP
import yaml
import os
from datetime import datetime
# PICKLE SETUP
import pickle
# LOG SETUP
import pandas as pd
from tabulate import tabulate #Future Markdown

# ====================== FUNCTION SETUP ====================== #
def credential_yaml():
    try:
        with open('config/credentials.yaml') as stream:
            credential = yaml.safe_load(stream)
            print('Sucessfully imported credential.yaml file')
    except yaml.YAMLError as e:
        print("Failed to parse credentials " + e.__str__())
    except Exception as e:
        print("Failed to parse credentials " + e.__str__())
    return credential

def picke_replace(name, file):
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
    # Create pickle file
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

# ====================== RETRIEVING DATA FROM KOREA GOVERNMENT API ====================== #
def api_encodetype(name, environment):
    """
    :param name: main name of YAML configuration
    :param environment: api
    :return:
    """
    pickle_name = name
    starttime = datetime.now()
    print(starttime)

    # ====================== CONFIGURATION.YAML READING ====================== #
    credential = credential_yaml()
    # ======================================================================== #

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
        print(name.upper() + ' API Connection has been successfully completed')
    except:
        print(name.upper() + ' has failed to open API Connection')

    df = json_normalize(json_file['data'])

    # ====================== Export to Pickle  ======================#
    picke_replace(name=pickle_name, file=df)
    # ===============================================================#

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
    pickle_name = name
    starttime = datetime.now()
    print(starttime)

    # ====================== CONFIGURATION.YAML READING ====================== #
    credential = credential_yaml()
    # ======================================================================== #

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
        print(name.upper() + ' API Connection has been successfully completed')
    except:
        print(name.upper() + ' has failed to open API Connection')

    df = json_normalize(json_file['response']['body']['items']['item'])

    # ====================== Export to Pickle  ======================#
    picke_replace(name=pickle_name, file=df)
    # ===============================================================#

    endtime = datetime.now()
    print(endtime)
    timetaken = endtime - starttime
    print('Time taken : ' + timetaken.__str__())

# ====================== RETRIEVING DATA FROM YOUTUBE API ====================== #

def api_youtube_popular(name, environment, max_result):

    # ====================== Setup ====================== #
    pd.options.mode.chained_assignment = None  # Off warning messages, default='warn'
    starttime = datetime.now()
    print(starttime)
    dictionary = {0: 'Wiki_Category_1', 1: 'Wiki_Category_2', 2: 'Wiki_Category_3', 3: 'Wiki_Category_4',
                  4: 'Wiki_Category_5', 5: 'Wiki_Category_6'}
    dictionary_list = list(dictionary.values())
    pickle_name = name

    # ====================== CONFIGURATION.YAML Reading ====================== #
    credential = credential_yaml()
    # ======================================================================== #

    # ====================== Retrieving API and store in DF  ======================#
    env_cred = credential[name][environment]
    service_key = env_cred['api_key']
    youtube = build('youtube', 'v3', developerKey=service_key)

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

    # ====================== YOUTUBE_VIDEO_LIST : Data Mapping  ======================#

    # Select Columns
    df_popular = df_popular[
        ['snippet.title', 'id', 'snippet.channelTitle', 'snippet.channelId', 'snippet.publishedAt', 'snippet.tags',
         'snippet.categoryId',  # video().list(part='snippet')
         'statistics.viewCount', 'statistics.likeCount', 'statistics.dislikeCount', 'statistics.favoriteCount',
         'statistics.commentCount',  # video().list(part='statistics')
         'topicDetails.topicCategories',  # video().list(part='topicDetails')
         'status.madeForKids']]

    # Rename Columns
    df_popular.rename(columns={'snippet.title': 'VideoTitle',
                               'id': 'VideoId',
                               'snippet.channelTitle': 'ChannelTitle',
                               'snippet.channelId': 'ChannelId',
                               'snippet.publishedAt': 'PublishedAt',
                               'snippet.tags': 'Tags',
                               'snippet.categoryId': 'CategoryId',
                               'statistics.viewCount': 'ViewCount',
                               'statistics.likeCount': 'LikeCount',
                               'statistics.dislikeCount': 'DislikeCount',
                               'statistics.favoriteCount': 'FavoriteCount',
                               'statistics.commentCount': 'CommentCount',
                               'topicDetails.topicCategories': 'TopicCategories',
                               'status.madeForKids': 'ForKids',
                               }, inplace=True)

    # Reset Index
    df_popular = df_popular.reset_index(drop=True)

    # Split TopicCategories URL
    catrgory_split = df_popular['TopicCategories']
    catrgory_split = pd.DataFrame(catrgory_split)
    catrgory_split = catrgory_split['TopicCategories'].apply(pd.Series).rename(columns=dictionary)

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
    df_popular = df_popular.merge(catrgory_split, left_index=True, right_index=True)
    del df_popular['TopicCategories']
    print('Youtube Video List : Data mapping has been successfully completed')

    # ====================== YOUTUBE_VIDEO_CATEGORY : Data Mapping  ======================#
    df_videocategory = df_videocategory[['id', 'snippet.title']]
    df_videocategory.rename(columns={'id': 'CategoryId',
                                     'snippet.title': 'Reg_Category'
                                     }, inplace=True)
    print('Youtube Video Category : Data mapping has been successfully completed')

    # ====================== MERGE : df_popular & df_videocategory ====================== #
    df_popular = df_popular.merge(df_videocategory, how='inner', on='CategoryId')

    # ====================== Export to Pickle & Read  ======================#
    picke_replace(name=pickle_name, file=df_popular)
    youtube_popular = read_pickle('youtube_popular.pkl')
    # ======================================================================#

    endtime = datetime.now()
    print(endtime)
    timetaken = endtime - starttime
    print('Time taken : ' + timetaken.__str__())

    return youtube_popular

class channel:
    def __init__(self, cha_name):
        self.cha_name = cha_name

    def search(self):
        """
        NAME : channel.search()
        DESCRIPTION : Print channel information from the channel id key in __init__(cha_name)
        USAGE : User can check statistics figures in each relevant channel and finally uses ChanneId.
        """
        pd.options.mode.chained_assignment = None  # Off warning messages, default='warn'
        starttime = datetime.now()
        print(starttime)

        # ====================== CONFIGURATION.YAML Reading ====================== #
        credential = credential_yaml()
        name = 'youtube_popular'
        environment = 'youtube'
        # ======================================================================== #

        # ====================== Retrieving API and store in DF  ======================#
        env_cred = credential[name][environment]
        service_key = env_cred['api_key']
        youtube = build('youtube', 'v3', developerKey=service_key)

        try:
            # YOUTUBE_CHANNEL_SEARCH
            res_channel_search = youtube.search().list(part='snippet', type='channel', regionCode='KR', q=self.cha_name,
                                                       order='videoCount', maxResults=20).execute()
            df_channel_search = json_normalize(res_channel_search['items'])
            print('Channel Search API Connection : ' + self.cha_name + '  has been successfully completed')
        except:
            print('Channel Search : ' + self.cha_name + ' has failed to open API connection')

        # ====================== YOUTUBE_CHANNEL_SEARCH : Data Mapping  ======================#

        # Select Columns
        df_channel_search = df_channel_search[['id.kind', 'id.channelId', 'snippet.publishedAt', 'snippet.title']]

        # Rename Columns
        df_channel_search.rename(columns={'id.kind': 'Type',
                                          'id.channelId': 'ChannelId',
                                          'snippet.publishedAt': 'PublishedAt',
                                          'snippet.title': 'ChannelTitle',
                                          'snippet.categoryId': 'CategoryId'
                                          }, inplace=True)

        # ======================  Retrieving API : YOUTUBE_CHANNEL_INFO  ======================#

        df_channel_info = pd.DataFrame()

        # Running loop to get Channel_ID from 'df_channel_search'
        for id in df_channel_search['ChannelId']:
            try:
                # YOUTUBE_CHANNEL_INFORMATION
                res_channel = youtube.channels().list(part=['snippet', 'statistics', 'contentDetails'],
                                                      id=id).execute()
                df = json_normalize(res_channel['items'][0])
                df_channel_info = df_channel_info.append(df)
                print('Channel ID API Connection : ' + id + '  has been successfully completed')
            except:
                print('Channel ID : ' + id + ' has failed to open API connection')

        # ====================== YOUTUBE_CHANNEL_INFO : Data Mapping  ======================#
        # Select Columns
        df_channel_info = df_channel_info[['id',
                                           'snippet.customUrl',
                                           'statistics.viewCount', 'statistics.subscriberCount',
                                           'statistics.videoCount']]

        # Rename Columns
        df_channel_info.rename(columns={'id': 'ChannelId',
                                        'snippet.customUrl': 'CustomUrl',
                                        'statistics.viewCount': 'ViewCount',
                                        'statistics.subscriberCount': 'SubscriberCount',
                                        'statistics.videoCount': 'VideoCount'
                                        }, inplace=True)

        # ====================== MERGE : df_channel_search & df_channel_info ====================== #
        # Merge & Rename columns
        df_chanel = df_channel_search.merge(df_channel_info, how='inner', on='ChannelId')
        # ========================================================================================= #

        print(df_chanel.to_markdown())  # Choose index=TRUE / index=FALSE
        endtime = datetime.now()
        print(endtime)
        timetaken = endtime - starttime
        print('Time taken : ' + timetaken.__str__())

    def video(self, channel_id):
        """
        NAME : channel.video()
        DESCRIPTION : Create pickle files for all video information from the channel ID key in by users.
        USAGE : User can have a list of video IDs from the channel with some basic statistic figures.
        """
        # ====================== Setup ====================== #
        pd.options.mode.chained_assignment = None  # Off warning messages, default='warn'
        starttime = datetime.now()
        print(starttime)
        dictionary = {0: 'Wiki_Category_1', 1: 'Wiki_Category_2', 2: 'Wiki_Category_3', 3: 'Wiki_Category_4',
                      4: 'Wiki_Category_5', 5: 'Wiki_Category_6'}
        dictionary_list = list(dictionary.values())

        # ====================== CONFIGURATION.YAML Reading ====================== #
        credential = credential_yaml()
        name = 'youtube_popular'
        environment = 'youtube'
        # ======================================================================== #

        # ====================== Retrieving API and store in DF  ======================#
        env_cred = credential[name][environment]
        service_key = env_cred['api_key']
        youtube = build('youtube', 'v3', developerKey=service_key)

        # STEP1 : PLAYLIST_ID
        try:
            res_playlist = youtube.channels().list(part='contentDetails', id=channel_id).execute()
            playlist_id = res_playlist['items'][0]['contentDetails']['relatedPlaylists']['uploads']
            print('Channel PlaylistID API Connection : ' + self.cha_name + ' has been successfully completed')
        except:
            print('Channel PlaylistID API Connection : ' + self.cha_name + ' has failed to initiate')

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

        # ====================== YOUTUBE_CHANNEL_INFO : Data Mapping  ======================#
        # Select Columns
        df_video_info = df_video_info[[
            'id',
            'snippet.title', 'snippet.publishedAt',
            'statistics.viewCount', 'statistics.likeCount', 'statistics.dislikeCount', 'statistics.favoriteCount','statistics.commentCount',  # video().list(part='statistics')
            'topicDetails.topicCategories',  # video().list(part='topicDetails')
            'status.madeForKids']]

        # Rename Columns
        df_video_info.rename(columns={'id': 'VideoId',
                                   'snippet.title': 'VideoTitle',
                                   'snippet.publishedAt': 'PublishedAt',
                                   'statistics.viewCount': 'ViewCount',
                                   'statistics.likeCount': 'LikeCount',
                                   'statistics.dislikeCount': 'DislikeCount',
                                   'statistics.favoriteCount': 'FavoriteCount',
                                   'topicDetails.topicCategories': 'TopicCategories',
                                   'status.madeForKids': 'ForKids',
                                   }, inplace=True)

        # Reset Index
        df_video_info = df_video_info.reset_index()

        # Split TopicCategories URL
        catrgory_split = df_video_info['TopicCategories']
        catrgory_split = pd.DataFrame(catrgory_split)
        catrgory_split = catrgory_split['TopicCategories'].apply(pd.Series).rename(columns=dictionary)

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
        del df_video_info['TopicCategories']
        del df_video_info['index']
        print('Youtube Video Information : Data mapping has been successfully completed')

        # ====================== Export to Pickle & Read  ======================#
        picke_replace(name='video_info', file=df_video_info)
        video_info = read_pickle('video_info.pkl')
        # ======================================================================#

        endtime = datetime.now()
        print(endtime)
        timetaken = endtime - starttime
        print('Time taken : ' + timetaken.__str__())

        return video_info

    def title_find(self, find):
        """
        :param find: Find substrings from the channel title.
        NAME : channel.title_find()
        DESCRIPTION : Find substrings from the channel title to narrow down the search results.
        USAGE : Extraction of the comments takes less number of videos after the tile search.
        """
        # ====================== Setup ====================== #
        pd.options.mode.chained_assignment = None  # Off warning messages, default='warn'

        # ====================== CONFIGURATION.YAML Reading ====================== #
        credential = credential_yaml()
        name = 'youtube_popular'
        environment = 'youtube'
        # ======================================================================== #

        # ====================== LOAD PICKLES ====================== #
        video_info = read_pickle('video_info.pkl')

        # ====================== FIND SUBSTRINGS ====================== #
        video_info_sub = video_info.loc[video_info['VideoTitle'].str.contains(find, case=False)].reset_index(drop=True)

        # ====================== Export to Pickle & Read ======================#
        picke_replace(name='video_info_subs', file=video_info_sub)
        video_info_sub = read_pickle('video_info_subs.pkl')
        # =====================================================================#

        return video_info_sub

    def comment(self):
        """
        NAME : channel.comment()
        DESCRIPTION : Extract comments from video_info_subs
        USAGE : The comment will be used to have sentiment analysis
        """
        # ====================== Setup ====================== #
        pd.options.mode.chained_assignment = None  # Off warning messages, default='warn'
        starttime = datetime.now()
        print(starttime)

        # ====================== CONFIGURATION.YAML Reading ====================== #
        credential = credential_yaml()
        name = 'youtube_popular'
        environment = 'youtube'
        # ======================================================================== #

        # ====================== Read Pickels (VideoIDs) ======================#
        video_info_sub = read_pickle('video_info_subs.pkl')
        # =====================================================================#

        # ====================== Retrieving API and store in DF  ======================#
        env_cred = credential[name][environment]
        service_key = env_cred['api_key']
        youtube = build('youtube', 'v3', developerKey=service_key)

        # ====================== Retrieving API and store in DF  ======================#

        df_comment = pd.DataFrame()
        try:
            for id in video_info_sub.VideoId:
                # CommentThread based on relevance filter
                res_rel_comments = youtube.commentThreads().list(part='snippet', videoId=id, order='relevance', maxResults=25).execute()
                df = json_normalize(res_rel_comments['items'])
                df_comment = df_comment.append(df)
                # CommentThread based on order filter
                res_ord_comments = youtube.commentThreads().list(part='snippet', videoId=id, order='time', maxResults=25).execute()
                df = json_normalize(res_ord_comments['items'])
                df_comment = df_comment.append(df, ignore_index=True)
            # Reset_index()
            df_comment = df_comment.reset_index(drop=True)
            print(str(len(df_comment)) + ' Comments has been successfully loaded')
        except:
            print('Comments has failed to load')

        # ====================== YOUTUBE_COMMENT : Data Mapping  ======================#

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
        df_comment.rename(columns={'id': 'CommentId',
                                   'snippet.topLevelComment.snippet.textOriginal': 'Comment',
                                   'snippet.topLevelComment.snippet.authorDisplayName': 'Author',
                                   'snippet.topLevelComment.snippet.likeCount': 'LikeCount',
                                   'snippet.topLevelComment.snippet.publishedAt': 'PublishedAt',
                                   'snippet.totalReplyCount': 'ReplyCount',
                                   }, inplace=True)

        # ====================== Export to Pickle & Read ======================#
        picke_replace(name='video_comment', file=df_comment)
        video_comment = read_pickle('video_comment.pkl')
        # =====================================================================#

        endtime = datetime.now()
        print(endtime)
        timetaken = endtime - starttime
        print('Time taken : ' + timetaken.__str__())

        return video_comment

####################################################################################################
# ====================== Setup ====================== #
        pd.options.mode.chained_assignment = None  # Off warning messages, default='warn'
        starttime = datetime.now()
        print(starttime)

        # ====================== CONFIGURATION.YAML Reading ====================== #
        credential = credential_yaml()
        name = 'youtube_popular'
        environment = 'youtube'
        # ======================================================================== #

        # ====================== Retrieving API and store in DF  ======================#
        env_cred = credential[name][environment]
        service_key = env_cred['api_key']
        youtube = build('youtube', 'v3', developerKey=service_key)

        #
        res_activities = youtube.channelSections().list(part=['snippet','contentDetails','id'], channelId='UCsJ6RuBiTVWRX156FVbeaGg').execute()
        df = json_normalize(res_rel_comments['items'])


# ====================== API RUNNING ====================== #
def run_covid_api():
    # DATA_GO_KR
    api_encodetype(name='covid_vaccines', environment='data_go_kr')
    api_decodetype(name='covid_age_sex', environment='data_go_kr', startdate='20200210')
    api_decodetype(name='covid_city', environment='data_go_kr', startdate='20200210')
    api_decodetype(name='covid_cases', environment='data_go_kr', startdate='20200210')

def run_youtube_chart():
    # YOUTUBE_POPULAR_CHART
    api_youtube_popular(name='youtube_popular', environment='youtube', max_result=20)


