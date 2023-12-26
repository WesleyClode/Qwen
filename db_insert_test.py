from logging import getLogger, Formatter, INFO
from db_connector import DatabaseLogHandler

# 设置日志配置
logger = getLogger('api_log')
logger.setLevel(INFO)
formatter = Formatter('%(asctime)s - %(name)s - %(levelname)s %(message)s')
# 写日志到数据库中
dblog = DatabaseLogHandler()
dblog.setFormatter(formatter)
logger.addHandler(dblog)
user_id = f"wurj"
user_ask = f"user:你好"
assistant = f"assistant:你好，我能为你做些什么？"
session_id = "a-s-d"
msg_id = "zxc"
msg_index = 2

user_msg = {'user_id': user_id, 'chat_msg': user_ask, 'session_id': session_id, 'msg_id': msg_id,
            'msg_index': msg_index}

reply_msg = {'user_id': user_id, 'chat_msg': assistant, 'session_id': session_id, 'msg_id': msg_id,
             'msg_index': msg_index+1}

logger.info(user_msg)
logger.info(reply_msg)
