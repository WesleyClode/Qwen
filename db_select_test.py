import copy

from db_pool_instance import db_pool

pool = db_pool


def fetch(sql):
    conn = pool.connection()
    cursor = conn.cursor()
    cursor.execute(sql)
    result = cursor.fetchall()
    cursor.close()
    conn.close()
    return result


sql = f"SELECT user_id , session_id FROM log WHERE user_id = 'wurj' GROUP BY session_id"

result = fetch(sql)

session_ids = []
msg_ids = []
msg_indexes = []
chat_msgs = []
for row in result:
    session_ids.append(row[1])
    sql = f'''SELECT user_id, session_id, msg_id, msg_index, chat_msg FROM log WHERE user_id = '{row[0]}' AND session_id = '{row[1]}'
    ORDER BY msg_id'''
    ordered_result = fetch(sql)
    msg_id = []
    msg_index = []
    chat_msg = []
    for item in ordered_result:
        msg_id.append(item[2])
        msg_index.append(item[3])
        chat_msg.append(item[4])
    msg_ids.append(msg_id)
    msg_indexes.append(msg_index)
    chat_msgs.append(chat_msg)

print(session_ids)
print(msg_ids)
print(msg_indexes)
print(chat_msgs)

