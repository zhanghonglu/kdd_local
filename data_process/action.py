from math import radians, cos, sin, asin, sqrt
import time
import random
import os.path
import numpy as np
import pandas as pd

order_file_all = []   #全局读取的txt的文件
for day in range(1, 31):
    print("读取订单{}文件".format(day))
    if day < 10:
        day_str = '0' + str(day)
    else:
        day_str = str(day)
    name = "../res/total_ride_request/order_201611" + day_str
    my_path = os.path.abspath(os.path.dirname(__file__))
    file_name = os.path.join(my_path, name)
    order_file_all.append(pd.read_csv(file_name,dtype={"order_id": str, "start_time": np.int32, "end_time": np.int32, "on_lg": np.float64,"on_lt": np.float64, "off_lg": np.float64, "off_lt": np.float64,"reward": np.float64, "start_time_day": np.int32, "end_time_day": np.int32}))


def haversine_distance(lon1, lat1, lon2, lat2): # 经度1，纬度1，经度2，纬度2 （十进制度数）
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # 将十进制度数转化为弧度
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371 # 地球平均半径，单位为公里
    return c * r


def get_action(time, s_lg,  s_lt,time_limit=900,distance_limit=2.5,order_num = 10):#action的选取
    actions = pd.DataFrame()
    distance = distance_limit
    time_limit_1 = time_limit
    if(s_lg==0 or s_lt==0):
        return pd.DataFrame(np.zeros((10, 10)))

    while(len(actions)< order_num ):
        day = random.randint(0,29)
        global order_file_all
        order_file = order_file_all[day]
        actions = order_file[order_file.apply(lambda x:haversine_distance(s_lg, s_lt,x.on_lg,x.on_lt)<= distance,axis=1)& (order_file.start_time_day >= time)&(order_file.start_time_day <= (time+time_limit_1))].copy()
        # print("当前动作数量{} 时间限制{} 距离限制{}".format(len(actions), time_limit_1, distance))
        distance = distance + 1
        time_limit_1 =time_limit_1 + 180
        if time+time_limit_1>=86400:
            return pd.DataFrame(np.zeros((10, 10)))


    actions["distance"] = actions.apply(lambda x: haversine_distance(s_lg, s_lt, x.on_lg, x.on_lt), axis=1).copy()
    actions["s_t"] = time
    actions["s_lg"] = s_lg
    actions["s_lt"] = s_lt
    actions = actions.sort_values(by=['distance', 'start_time_day']).head(order_num)
    return actions.loc[:, ['s_t','s_lg','s_lt','start_time_day', 'end_time_day', 'on_lg', 'on_lt','off_lg', 'off_lt', 'reward']]



#



# print(get_action(35197,104.09464,30.703971000000003 ,6000,100))
#



#     """
# #     根据两个点的经纬度求两点之间的距离
# #     :param Lng_A:  经度1
# #     :param Lat_A:   维度1
# #     :param Lng_B:  经度2
# #     :param Lat_B:   维度2
# #     :return:  单位米
# #     """



# print(haversine_distance(117.113011,36.706185,117.112368,36.691557 ))
# 将数据处理为分钟数，从当天零点开始算的分钟数

def time_data1(time_sj):  # 传入单个时间比如'2019-8-01 00:00:00'，类型为str
    data_sj = time.strptime(time_sj, "%Y-%m-%d %H:%M:%S")  # 定义格式
    time_int = int(time.mktime(data_sj))
    return time_int  # 返回传入时间的时间戳，类型为int

# for i in range(1, 31):
# #     if i < 10:
# #         day = '0'+str(i)
# #     else:
# #         day = str(i)
# #     file_name = 'E:\\SDU\\KDD-CUP\\res\\order_01-30.zip\\total_ride_request\\order_201611'+day
# #     order = pd.read_csv(file_name, encoding='UTF-8',dtype ={"order_id":str,"start_time":np.int64,"end_timed_day":np.int64,"on_lg":np.float64,"on_lt":np.float64, "off_lg":np.float64,"off_lt":np.float64,"reward":np.float64})
# #     # order["start_time"].apply()
#     # time_base = '2016-11-'+day+' 00:00:00'
#     # order["start_time_day"] = order["start_time"].apply(lambda x:x-time_data1(time_base))
#     # order["end_time_day"] = order["end_time"].apply(lambda x:x-time_data1(time_base))
#     # order["start_time_day"] = order["start_time"].apply(lambda x:x-time_data1(time_base))
#     # order.to_csv(file_name)
#     print(order.dtypes)


