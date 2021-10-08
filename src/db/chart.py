from src.utils.api import api_youtube_popular
from configparser import ConfigParser
from sqlalchemy import create_engine
import psycopg2
from datetime import datetime, date
import calendar

def config_list(filename='config/database.ini', section='postgresql_list'):
    # create a parser
    parser = ConfigParser()
    # read config file
    parser.read(filename)

    # get section, default to postgresql
    db = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db[param[0]] = param[1]
    else:
        raise Exception('Section {0} not found in the {1} file'.format(section, filename))

    return db

def config_url(filename='config/database.ini', section='postgresql_url'):
    # create a parser
    parser = ConfigParser()
    # read config file
    parser.read(filename)

    # get section, default to postgresql
    db = {}
    if parser.has_section(section):
        params = parser.items(section)
        for param in params:
            db = param[0]+':'+param[1]
    else:
        raise Exception('Section {0} not found in the {1} file'.format(section, filename))

    return db

def connect():
    """ Connect to the PostgreSQL database server """
    conn = None
    try:
        # read connection parameters
        params = config_list()

        # connect to the PostgreSQL server
        print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(**params)

        # create a cursor
        cur = conn.cursor()

        # execute a statement
        print('PostgreSQL database version:')
        cur.execute('SELECT version()')

        # display the PostgreSQL database server version
        db_version = cur.fetchone()
        print(db_version)

        # close the communication with the PostgreSQL
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print('Database connection closed.')

def chart_export(key):

    starttime = datetime.now()
    print(starttime)
    youtube_popular = api_youtube_popular(name='youtube_popular', environment='youtube', max_result=20)

    # Move last column(run_date) to first sequence
    youtube_popular['run_date'] = date.today()
    youtube_popular['day'] = calendar.day_name[date.today().weekday()]
    cols = youtube_popular.columns.tolist()
    cols = cols[-2:] + cols[:-2]
    youtube_popular = youtube_popular[cols]

    if key == 'new':
        # CASE 1 : Push DF to Table
        params = config_url()
        engine = create_engine(params)
        youtube_popular.to_sql('popular_chart', engine, index=False)

    elif key == 'update':
        # CASE 2 : Append DF to Table
        params = config_url()
        engine = create_engine(params)
        youtube_popular.to_sql('popular_chart', engine, if_exists='append', index=False)

    endtime = datetime.now()
    print(endtime)
    timetaken = endtime - starttime
    print('Time taken : ' + timetaken.__str__())

