# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 13:11:04 2020

@author: FMENG
"""

##catboost测试
import pandas as pd
import copy
import numpy as np
from math import isnan
import catboost as cb
from catboost import Pool
import lightgbm as lgb

pd.set_option('display.max_columns',None)
pd.set_option('display.width',None)



train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test_noLabel.csv')
print("====================初始数据========================")
print(train.info())
print(test.info())


# 房屋朝向的汉字编码处理
#房屋朝向 共有64种不同的不同的结果，对相同的房屋朝向进行编码
train_vec = copy.deepcopy(train['房屋朝向'])
train_vec_dic = list(set(train_vec))
#print("train_vec_dic",train_vec_dic)
num_train_vec_dic = len(train_vec_dic)
encode_train_vec_dic = list(np.arange(num_train_vec_dic))
dic_enc_train_vec = dict(map(lambda x, y: [x, y], train_vec_dic, encode_train_vec_dic))
encode_train_vec = []
for i in train_vec:
    temp = dic_enc_train_vec[i]
    encode_train_vec.append(temp)

test_vec = copy.deepcopy(test['房屋朝向'])
test_vec_dic = list(set(test_vec))
te_vec_new = []
for i in test_vec_dic:
    if i not in train_vec_dic:
        te_vec_new.append(i)
    

num_te_vec_new = len(te_vec_new)
enc_te_vec_new = list(np.arange(num_train_vec_dic, num_train_vec_dic+num_te_vec_new))
dic_te_vec_new = dict(map(lambda x, y: [x, y], te_vec_new, enc_te_vec_new))
dic_te_vec = dict(dic_enc_train_vec, **dic_te_vec_new)
encode_test_vec = []
for i in test_vec:
    temp = dic_te_vec[i]
    encode_test_vec.append(temp)
train['RoomDir1'] = encode_train_vec
test['RoomDir1'] = encode_test_vec
del dic_enc_train_vec, dic_te_vec, dic_te_vec_new, enc_te_vec_new, encode_test_vec
del encode_train_vec, encode_train_vec_dic, i, num_te_vec_new, num_train_vec_dic
del te_vec_new, temp, test_vec, test_vec_dic, train_vec, train_vec_dic


train_vec = list(copy.deepcopy(train['房屋朝向']))
diract = []
for i in train_vec:
    temp = i.split()
    temp = temp[0]
    diract.append(temp)
Diract = []
for i in diract:
    if i == '南':
        temp = 0
    elif i == '东南':
        temp = 1
    elif i == '东':
        temp = 2
    elif i == '西南':
        temp = 3
    elif i == '北':
        temp = 4
    elif i == '西':
        temp = 5
    elif i == '东北':
        temp = 6
    elif i == '西北':
        temp = 7
    Diract.append(temp)
train.drop('房屋朝向',axis=1, inplace=True)
train['RoomDir2'] = Diract
del train_vec, diract, i, temp, Diract
test_vec = list(copy.deepcopy(test['房屋朝向']))
diract = []
for i in test_vec:
    temp = i.split()
    temp = temp[0]
    diract.append(temp)
Diract = []
for i in diract:
    if i == '南':
        temp = 0
    elif i == '东南':
        temp = 1
    elif i == '东':
        temp = 2
    elif i == '西南':
        temp = 3
    elif i == '北':
        temp = 4
    elif i == '西':
        temp = 5
    elif i == '东北':
        temp = 6
    elif i == '西北':
        temp = 7
    Diract.append(temp)
test.drop('房屋朝向',axis=1, inplace=True)
test['RoomDir2'] = Diract
del test_vec, diract, i, temp, Diract


"""
地铁线路存在104761条缺失值，value_counts()以后发现，
存在1、2、3、4、5共五组数据，因此我认为缺失值为没有地铁线路存在，填充0.
"""
train_subway = list(copy.deepcopy(train['地铁线路']))
test_subway = list(copy.deepcopy(test['地铁线路']))
Train_subway = []
for i in train_subway:
    if isnan(i):
        Train_subway.append(0)
    else:
        Train_subway.append(i)
Test_subway = []
for i in test_subway:
    if isnan(i):
        Test_subway.append(0)
    else:
        Test_subway.append(i)

train.drop('地铁线路',axis=1, inplace=True)
train['SubwayLine'] = Train_subway
test.drop('地铁线路',axis=1, inplace=True)
test['SubwayLine'] = Test_subway
del Test_subway, Train_subway, i, test_subway, train_subway

"""
地铁站点同样存在104761条缺失值，观察数据发现，其值为0.1-11.9之间的小数
在这里，我认为缺失值是没有地铁站店存在，因此，我将缺失值填充为0
"""

train_station = list(copy.deepcopy(train['地铁站点']))
test_station = list(copy.deepcopy(test['地铁站点']))
Train_station = []
for i in train_station:
    if isnan(i):
        Train_station.append(0)
    else:
        Train_station.append(i)
Test_station = []
for i in test_station:
    if isnan(i):
        Test_station.append(0)
    else:
        Test_station.append(i)
train.drop('地铁站点',axis=1, inplace=True)
train['SubwaySta'] = Train_station
test.drop('地铁站点',axis=1, inplace=True)
test['SubwaySta'] = Test_station
del Test_station, Train_station, i, test_station, train_station

"""
#区、位置处理、小区房屋出租数量
print(train['区'].isnull().sum())
print(train['位置'].isnull().sum())
print(train['小区房屋出租数量'].isnull().sum())
print(train['距离'].isnull().sum())

print(train['区'].value_counts())
print(train['位置'].value_counts())
print(train['小区房屋出租数量'].value_counts())
print(train['距离'].value_counts())

小区房屋出租数量缺失了31条，不多，不影响主要结果,我用上一个的值填充。
"""
train = train.sort_values(by=['小区名', '楼层', '时间'], ascending=(True, True, True))
train_num_house = list(copy.deepcopy(train['小区房屋出租数量']))
Time = list(copy.deepcopy(train['时间']))
count = []
num = len(train_num_house)
for i in range(num):
    if isnan(train_num_house[i]):
        count.append(count[i-1])
    else:
        count.append(train_num_house[i])
train.drop('小区房屋出租数量',axis=1, inplace=True)
train.drop('时间',axis=1, inplace=True)
train['RentRoom'] = count
train['Time'] = Time
test = test.sort_values(by=['小区名', '楼层', '时间'], ascending=(True, True, True))
test_num_house = list(copy.deepcopy(test['小区房屋出租数量']))
Time = list(copy.deepcopy(test['时间']))
count = []
num = len(test_num_house)
for i in range(num):
    if isnan(test_num_house[i]):
        count.append(count[i-1])
    else:
        count.append(test_num_house[i])
test.drop('小区房屋出租数量',axis=1, inplace=True)
test.drop('时间',axis=1, inplace=True)
test['RentRoom'] = count
test['Time'] = Time
del count, i, num, test_num_house, train_num_house


"""
区的缺失值较少，值为0-14之间的整数，但是唯独缺少了5，因此填充为5。
位置值为0-152，唯独缺少76，填充76

"""
train_province = list(copy.deepcopy(train['区']))
test_province = list(copy.deepcopy(test['区']))
Train_province = []
for i in train_province:
    if isnan(i):
        Train_province.append(5)
    else:
        Train_province.append(i)
Test_province = []
for i in test_province:
    if isnan(i):
        Test_province.append(5)
    else:
        Test_province.append(i)
train.drop('区',axis=1, inplace=True)
train['Region'] = Train_province
test.drop('区',axis=1, inplace=True)
test['Region'] = Test_province
del Test_province, Train_province, i, test_province, train_province


train_location = list(copy.deepcopy(train['位置']))
test_location = list(copy.deepcopy(test['位置']))
Train_location = []
for i in train_location:
    if isnan(i):
        Train_location.append(76)
    else:
        Train_location.append(i)
Test_location = []
for i in test_location:
    if isnan(i):
        Test_location.append(76)
    else:
        Test_location.append(i)
train.drop('位置',axis=1, inplace=True)
train['Local'] = Train_location
test.drop('位置',axis=1, inplace=True)
test['Local'] = Test_location
del Test_location, Train_location, i, test_location, train_location



train_sub = []
train_sub = pd.DataFrame(train_sub)
test_sub = []
test_sub = pd.DataFrame(test_sub)
train_sub['小区名'] = list(copy.deepcopy(train['小区名']))
train_sub['距离'] = list(copy.deepcopy(train['距离']))
train_sub['地铁线路'] = list(copy.deepcopy(train['SubwayLine']))
train_sub['地铁站点'] = list(copy.deepcopy(train['SubwaySta']))

test_sub['小区名'] = list(copy.deepcopy(test['小区名']))
test_sub['距离'] = list(copy.deepcopy(test['距离']))
test_sub['地铁线路'] = list(copy.deepcopy(test['SubwayLine']))
test_sub['地铁站点'] = list(copy.deepcopy(test['SubwaySta']))


train_sub = pd.concat([train_sub, test_sub], ignore_index=True)
train_sub = train_sub.drop_duplicates()
train_sub = train_sub.sort_values(by=['小区名'], ascending=(True))
train_sub = train_sub.dropna(axis=0, how='any')
del test_sub

#根据小区名对距离进行调整，同一小区认定距离相同。
train_sub_name = train_sub['小区名']
train_sub_dis = train_sub['距离']
dic_name_dis = dict(map(lambda x, y: [x, y], train_sub_name, train_sub_dis))
#print("dic_name_dis",dic_name_dis)
train_distance = list(copy.deepcopy(train['距离']))
train_name = list(copy.deepcopy(train['小区名']))
num = len(train_name)
distance = []
for i in range(num):
    if (train_name[i] in dic_name_dis.keys()) and (isnan(train_distance[i])):
        distance.append(dic_name_dis[train_name[i]])
    else:
        distance.append(train_distance[i])
train.drop('距离',axis=1, inplace=True)
train['SubwayDis'] = distance
test_distance = list(copy.deepcopy(test['距离']))
test_name = list(copy.deepcopy(test['小区名']))
num = len(test_name)
distance = []
for i in range(num):
    if (test_name[i] in dic_name_dis.keys()) and (isnan(test_distance[i])):
        distance.append(dic_name_dis[test_name[i]])
    else:
        distance.append(test_distance[i])
test.drop('距离',axis=1, inplace=True)
test['SubwayDis'] = distance
del dic_name_dis, distance, i, num, test_distance, test_name, train_distance
del train_name, train_sub_dis, train_sub_name

#根据小区名对地铁线路进行调整,同一个小区的地铁线路是相同的
train_sub_name = train_sub['小区名']
train_sub_subway = train_sub['地铁线路']
dic_name_subway = dict(map(lambda x, y: [x, y], train_sub_name, train_sub_subway))
#print("dic_name_subway",dic_name_subway)
train_name = list(copy.deepcopy(train['小区名']))
train_subway = list(copy.deepcopy(train['SubwayLine']))
num = len(train_name)
subway = []
for i in range(num):
    if (train_name[i] in dic_name_subway.keys()) and (train_subway[i]==0):
        subway.append(dic_name_subway[train_name[i]])
    else:
        subway.append(train_subway[i])
train.drop('SubwayLine',axis=1, inplace=True)
train['SubwayLine'] = subway
test_name = list(copy.deepcopy(test['小区名']))
test_subway = list(copy.deepcopy(test['SubwayLine']))
num = len(test_name)
subway = []
for i in range(num):
    if (test_name[i] in dic_name_subway.keys()) and (test_subway[i]==0):
        subway.append(dic_name_subway[test_name[i]])
    else:
        subway.append(test_subway[i])
test.drop('SubwayLine',axis=1, inplace=True)
test['SubwayLine'] = subway
del dic_name_subway, i, num, subway, test_name, test_subway, train_name
del train_sub_name, train_sub_subway, train_subway

#根据小区名对地铁站点进行调整,同一个小区的地铁站点是相同的
train_sub_name = train_sub['小区名']
train_sub_position = train_sub['地铁站点']
dic_name_position = dict(map(lambda x, y: [x, y], train_sub_name, train_sub_position))
#print("dic_name_position",dic_name_position)
train_name = list(copy.deepcopy(train['小区名']))
train_position = list(copy.deepcopy(train['SubwaySta']))
num = len(train_name)
position = []
for i in range(num):
    if (train_name[i] in dic_name_position.keys()) and (train_position[i]==0):
        position.append(dic_name_position[train_name[i]])
    else:
        position.append(train_position[i])
train.drop('SubwaySta',axis=1, inplace=True)
train['SubwaySta'] = position
test_name = list(copy.deepcopy(test['小区名']))
test_position = list(copy.deepcopy(test['SubwaySta']))
num = len(test_name)
position = []
for i in range(num):
    if (test_name[i] in dic_name_position.keys()) and (test_position[i]==0):
        position.append(dic_name_position[test_name[i]])
    else:
        position.append(test_position[i])
test.drop('SubwaySta',axis=1, inplace=True)
test['SubwaySta'] = position
del dic_name_position, i, num, position, test_name, test_position, train_name
del train_position, train_sub_name, train_sub_position, train_sub

#共89122个距离为空值，对距离为空的补全
train_distance = list(copy.deepcopy(train['SubwayDis']))
distance = []
for i in train_distance:
    if isnan(i):
        distance.append(0)
    else:
        distance.append(i)
train.drop('SubwayDis',axis=1, inplace=True)
train['SubwayDis'] = distance
test_distance = list(copy.deepcopy(test['SubwayDis']))
distance = []
for i in test_distance:
    if isnan(i):
        distance.append(0)
    else:
        distance.append(i)
test.drop('SubwayDis',axis=1, inplace=True)
test['SubwayDis'] = distance
del distance, i, test_distance, train_distance

#居住状态为空的认定为0
train_live = list(copy.deepcopy(train['居住状态']))
live = []
for i in train_live:
    if isnan(i):
        live.append(0)
    else:
        live.append(i)
train.drop('居住状态',axis=1, inplace=True)
train['RentStatus'] = live
test_live = list(copy.deepcopy(test['居住状态']))
live = []
for i in test_live:
    if isnan(i):
        live.append(0)
    else:
        live.append(i)
test.drop('居住状态',axis=1, inplace=True)
test['RentStatus'] = live
del i, live, test_live, train_live

#装修状态为空填充为0
train_zx = list(copy.deepcopy(train['装修情况']))
zx = []
for i in train_zx:
    if isnan(i):
        zx.append(0)
    else:
        zx.append(i)
train.drop('装修情况',axis=1, inplace=True)
train['RenoCond'] = zx
test_zx = list(copy.deepcopy(test['装修情况']))
zx = []
for i in test_zx:
    if isnan(i):
        zx.append(0)
    else:
        zx.append(i)
test.drop('装修情况',axis=1, inplace=True)
test['RenoCond'] = zx
del i, test_zx, train_zx, zx


#print(train['出租方式'].isnull().sum())  缺失172309
#print(train['出租方式'].value_counts())  只有0和1

#对出租方式为空的，填充为2
train_cz = list(copy.deepcopy(train['出租方式']))
cz = []
for i in train_cz:
    if isnan(i):
        cz.append(2)
    else:
        cz.append(i)
train.drop('出租方式',axis=1, inplace=True)
train['RentType'] = cz
test_cz = list(copy.deepcopy(test['出租方式']))
cz = []
for i in test_cz:
    if isnan(i):
        cz.append(2)
    else:
        cz.append(i)
test.drop('出租方式',axis=1, inplace=True)
test['RentType'] = cz
del cz, i, test_cz, train_cz


train_ws = list(copy.deepcopy(train['卧室数量']))
train_t = list(copy.deepcopy(train['厅的数量']))
train_w = list(copy.deepcopy(train['卫的数量']))
total = []
num = len(train_ws)
for i in range(num):
    temp = train_ws[i] + train_t[i] + train_w[i]
    total.append(temp)
train['AllRoom'] = total
train['BedRoom'] = train_ws
train['LivingRoom'] = train_t
train['BathRoom'] = train_w
train.drop('卧室数量',axis=1, inplace=True)
train.drop('厅的数量',axis=1, inplace=True)
train.drop('卫的数量',axis=1, inplace=True)

test_ws = list(copy.deepcopy(test['卧室数量']))
test_t = list(copy.deepcopy(test['厅的数量']))
test_w = list(copy.deepcopy(test['卫的数量']))
total = []
num = len(test_ws)
for i in range(num):
    temp = test_ws[i] + test_t[i] + test_w[i]
    total.append(temp)
test['AllRoom'] = total
test['BedRoom'] = test_ws
test['LivingRoom'] = test_t
test['BathRoom'] = test_w
test.drop('卧室数量',axis=1, inplace=True)
test.drop('厅的数量',axis=1, inplace=True)
test.drop('卫的数量',axis=1, inplace=True)
del i, num, temp, test_t, test_w, test_ws, train_t, train_w, train_ws, total


train_ws = list(copy.deepcopy(train['BedRoom']))
train_w = list(copy.deepcopy(train['BathRoom']))
total = []
num = len(train_ws)
for i in range(num):
    temp = train_ws[i] + train_w[i]
    total.append(temp)
train['BBRoom'] = total
test_ws = list(copy.deepcopy(test['BedRoom']))
test_w = list(copy.deepcopy(test['BathRoom']))
total = []
num = len(test_ws)
for i in range(num):
    temp = test_ws[i] + test_w[i]
    total.append(temp)
test['BBRoom'] = total
del i, num, temp, test_w, test_ws, train_w, train_ws, total

#房屋面积有一个为空
train_acer = list(copy.deepcopy(train['房屋面积']))
acer = []
for i in train_acer:
    if i == 0:
        acer.append(1)
    else:
        acer.append(i)
train.drop('房屋面积',axis=1, inplace=True)
train['RoomArea'] = acer
test_acer = list(copy.deepcopy(test['房屋面积']))
acer = []
for i in test_acer:
    if i == 0:
        acer.append(1)
    else:
        acer.append(i)
test.drop('房屋面积',axis=1, inplace=True)
test['RoomArea'] = acer
del acer, i, test_acer, train_acer

# 卧室占整个房间的比例

#卧室均面积
bed_sub_all = 0.3
train_sq = list(copy.deepcopy(train['RoomArea']))
train_ws = list(copy.deepcopy(train['BedRoom']))
ws_sq = []
num = len(train_sq)

for i in range(num):
    temp = (train_sq[i] * bed_sub_all) / (train_ws[i] + 1)
    ws_sq.append(temp)
train['AveBedArea'] = ws_sq

test_sq = list(copy.deepcopy(test['RoomArea']))
test_ws = list(copy.deepcopy(test['BedRoom']))
ws_sq = []
num = len(test_sq)
for i in range(num):
    temp = (test_sq[i] * bed_sub_all) / (test_ws[i] + 1)
    ws_sq.append(temp)
test['AveBedArea'] = ws_sq
del train_sq, train_ws, ws_sq, i, temp, num, test_sq, test_ws, bed_sub_all

#卫的均面积
train_sq = list(copy.deepcopy(train['RoomArea']))
train_w = list(copy.deepcopy(train['BathRoom']))
w_sq = []
num = len(train_sq)
for i in range(num):
    temp = (train_sq[i] / 14) / (train_w[i] + 1)
    w_sq.append(temp)
train['AveBathArea'] = w_sq
test_sq = list(copy.deepcopy(test['RoomArea']))
test_w = list(copy.deepcopy(test['BathRoom']))
w_sq = []
num = len(test_sq)
for i in range(num):
    temp = (test_sq[i] / 14) / (test_w[i] + 1)
    w_sq.append(temp)
test['AveBathArea'] = w_sq
del train_sq, train_w, w_sq, temp, i, num, test_sq, test_w

#卧室总面积
train_wss = list(copy.deepcopy(train['AveBedArea']))
train_ws = list(copy.deepcopy(train['BedRoom']))
ws_s = []
num = len(train_wss)
for i in range(num):
    temp = train_wss[i] * train_ws[i]
    ws_s.append(temp)
train['AllBedArea'] = ws_s
test_wss = list(copy.deepcopy(test['AveBedArea']))
test_ws = list(copy.deepcopy(test['BedRoom']))
ws_s = []
num = len(test_wss)
for i in range(num):
    temp = test_wss[i] * test_ws[i]
    ws_s.append(temp)
test['AllBedArea'] = ws_s
del train_wss, train_ws, ws_s, num, i, temp, test_wss, test_ws


#print(train['总楼层'].isnull().sum()) 
#print(train['总楼层'].value_counts()) 
#print(train['楼层'].isnull().sum()) 
#print(train['楼层'].value_counts())   
"""
原数据给了0、1、2来描述楼的高低程度，首先对其归一化并进行量化，然后与总楼层相乘可以推断出房屋所在的大体楼层.
"""

total_floor = list(copy.deepcopy(train['总楼层']))
Total_floor = []
for i in total_floor:
    if i == 0:
        Total_floor.append(1)
    else:
        Total_floor.append(i)
train.drop('总楼层',axis=1, inplace=True)
train['TolHeight'] = Total_floor
total_floor = list(copy.deepcopy(test['总楼层']))
Total_floor = []
for i in total_floor:
    if i == 0:
        Total_floor.append(1)
    else:
        Total_floor.append(i)
test.drop('总楼层',axis=1, inplace=True)
test['TolHeight'] = Total_floor
del Total_floor, i, total_floor



def fun(p):
    if p == 0:
        r = 0
    elif p == 1:
        r = 0.3333
    elif p == 2:
        r = 0.6666
    return r

train['楼层'] = train['楼层'].apply(lambda x :fun(x))
test['楼层'] = test['楼层'].apply(lambda x :fun(x))


total_floor = list(copy.deepcopy(train['TolHeight']))
floor = list(copy.deepcopy(train['楼层']))
num = len(total_floor)
Floor = []
for i in range(num):
    temp = total_floor[i] * floor[i]
    Floor.append(temp)
train['Floor'] = Floor
train.drop('楼层',axis=1, inplace=True)
train['Height'] = floor
total_floor = list(copy.deepcopy(test['TolHeight']))
floor = list(copy.deepcopy(test['楼层']))
num = len(total_floor)
Floor = []
for i in range(num):
    temp = total_floor[i] * floor[i]
    Floor.append(temp)
test['Floor'] = Floor
test.drop('楼层',axis=1, inplace=True)
test['Height'] = floor
del total_floor, floor, Floor, num, i, temp



"""
小区名在整个特征里面其实重要性非常大，因为同一个小区的房屋租金价格基本相似，而不同小区之间的价格差别很大.
训练集中共存在5547个小区，做one-hot处理或者直接不处理都是不可行的。one-hot处理后特征爆炸，直接不处理更不妥，因其为分类变量，无大小关系之分，必须要进行处理。
因此，根据小区名与其他特征的关系重新构造了新特征
小区名与该小区的出租房屋面积、小区卧室出租数量、小区厅的出租数量、小区卫的出租数量、小区总楼层之间构造特征
"""
"""
items = ['RoomArea','BedRoom','LivingRoom','BathRoom','TolHeight']
for item in items:
    Neigh = train.groupby('小区名',as_index=False)[item].agg({item+"Ne":'mean'})
    train = train.merge(Neigh,on='小区名',how='left')
#去掉小区名
train.drop(['小区名'],axis=1,inplace=True)
for item in items:
    Neigh = test.groupby('小区名',as_index=False)[item].agg({item+"Ne":'mean'})
    test = test.merge(Neigh,on='小区名',how='left')
#去掉小区名
test.drop(['小区名'],axis=1,inplace=True)
print(train.head())
   """

print(train.head())
#print(test.head())
train.to_csv('data/train1.csv',encoding='utf-8',index=False)
test.to_csv('data/test1.csv',encoding='utf-8',index=False)






























