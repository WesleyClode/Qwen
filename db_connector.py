import logging
from config import *
from db_pool_instance import db_pool


class DatabaseLogHandler(logging.Handler):

    def __init__(self):
        self.pool = db_pool
        self.table = TABLE
        logging.Handler.__init__(self)
        # self.db = pymysql.connect(
        #     host=create,
        #     port=port,
        #     user=name,
        #     passwd=password,
        #     db=createID)  # 连接地址，登陆名，密码，数据库标示
        # self.table = table  # 表名
        # self.cursor = self.db.cursor()
        # self.db.commit()

    def emit(self, record):
        print(record)

        msg = record.getMessage()
        msg = eval(msg)
        keys = list(msg.keys())
        values = list(msg.values())
        if "chat_msg" in keys:
            for i in range(len(keys)):
                keys[i] = '`' + keys[i] + '`'
            keys = ','.join(keys)
            values = f"{values[0], values[1], values[2], values[3], values[4]}"
            sql = f"INSERT INTO {self.table}({keys}) VALUES {values}"
        else:
            sql = f"UPDATE {self.table} SET {keys[3]}={values[3]} WHERE {keys[0]}='{values[0]}' AND {keys[1]}='{values[1]}' AND {keys[2]}='{values[2]}'"

        print(sql)
        conn = self.pool.connection()
        cursor = conn.cursor()
        try:
            cursor.execute(sql)
            conn.commit()
            print(f"{sql} executed successfully!")
        except Exception as err:
            print('SQL执行错误', err)
        finally:
            cursor.close()
            conn.close()
