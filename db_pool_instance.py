from dbutils.pooled_db import PooledDB
import pymysql
from config import *

db_pool = PooledDB(
    creator=pymysql,
    maxconnections=10,
    mincached=5,
    host=HOST,
    port=PORT,
    user=USER,
    password=PASSWORD,
    database=DATABASE,
    charset='utf8',
)
