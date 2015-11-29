# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 17:25:42 2015

@author: Ophidian
"""
import time
import numpy as np
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression

def toDatetime(ori_timestamp):
    """
    convert timestamp to datetime
    """
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ori_timestamp))

def toTimestamp(ori_datetime):
    """
    convert datetime to timestamp
    """
    return time.mktime( time.strptime(ori_datetime,"%Y-%m-%d %H:%M:%S") )
    
def accumulateRecord(intervals, acc_header, acc_records, acc_counts, user_id , one_visit):
    
    record_interval = 0
    for i, timestamp in enumerate( intervals ):
        if one_visit["timestamp"] < timestamp:
            record_interval = i
            break
            
    if record_interval == 0:
        return

    if user_id not in acc_records:
        acc_records[user_id] = {}
        acc_counts[user_id] = {}
        
    if record_interval not in acc_records[user_id]:
        acc_records[user_id][record_interval] = np.zeros( len(acc_header) )
        acc_records[user_id][record_interval][0] = record_interval
        acc_counts[user_id][record_interval] = 0
    
    acc_counts[user_id][record_interval] += 1
    
    for i,key in enumerate( one_visit ):
        if key=="timestamp":
            continue
        field_val = one_visit[key]
        if field_val in acc_header:
            acc_records[user_id][record_interval][ acc_header.index(field_val) ] +=1

train_path = "D:/Dropbox/pageview_data/train.csv"
tpl_path = "D:/Dropbox/pageview_data/submission.csv"
result_path = "D:/Dropbox/pageview_data/my_submission.csv"

header = [
    "url_hash",
    "resolution",
    "browser",
    "os",
    "device_marketing",
    "device_brand",
    "cookie_pta",
    "date",
    "author_id",
    "category_id",
    "referrer_venue"
    ]

resolution_label=[
    "high",
    "low"
    ]
    
intervals = [
    toTimestamp("2014-11-02 00:00:00"),
    toTimestamp("2014-11-09 00:00:00"),
    toTimestamp("2014-11-16 00:00:00"),
    toTimestamp("2014-11-23 00:00:00"),
    toTimestamp("2014-11-30 00:00:00")
    ]

start_tmp = time.strptime("2011-11-01 00:00:00","%Y-%m-%d %H:%M:%S" )
train_start = time.mktime( start_tmp )

#parsed visit record, group by user
user_record = {}

#parsed label of visit fields
browser_label = []
os_label =[]
category_label = []
refer_label = []

with open(train_path , "r", encoding = "utf8") as train_file:
    for i,line in enumerate( train_file.readlines() ):
        if i==0:
            continue;   #skip header
        single_visit = line.strip().split(',')

        #about label
        if single_visit[2] not in browser_label:
            browser_label.append( single_visit[2] )
        if single_visit[3] not in os_label:
            os_label.append( single_visit[3] )
        if single_visit[10] not in refer_label:
            refer_label.append( single_visit[10] )
            
        #group resolution
        width_height = single_visit[1].strip().split("x")
        width = int(width_height[0])
        height = int(width_height[1])
        if width*height > 1024*768:
            current_res_label = resolution_label[0]
        else:
            current_res_label = resolution_label[1]

        #about user
        user_id = single_visit[6]
        if user_id not in user_record:
            user_record[ user_id ] = []
        
        wanted_info = {
            "timestamp": int(single_visit[7]),
            "resolution":current_res_label,
            "browser":single_visit[2],
            "os":single_visit[3],
            "refer":single_visit[10]
            }
        user_record[ user_id ].append( wanted_info )

"""        
print( "resolution_label count: " + str(len(resolution_label)) )
print( resolution_label )
print( "browser_label count: " + str(len(browser_label)) )
print( browser_label )
print( "os_label count: " + str(len(os_label)) )
print( os_label )
print( "refer_label count: " + str(len(refer_label)) )
print( refer_label )
"""

#make train/predit data header
acc_header = ["interval"] + resolution_label + browser_label + os_label + refer_label
print( "ACC_HEADER:" )
print( acc_header )

#make train data
visit_info_acc = {}
visit_count_acc = {}
total_visit = {}
for i,user_id in enumerate( user_record ):
    total_visit[user_id] = len( user_record[user_id] )
    for j,visit_record in enumerate( user_record[user_id] ):
        if visit_record["timestamp"] < intervals[0]:
            continue
        accumulateRecord( intervals, acc_header, visit_info_acc, visit_count_acc, user_id, visit_record )
""" well ... just *7/30
with open(tpl_path , "r", encoding = "utf8") as tpl_file, open(result_path , "w", encoding = "utf8") as result_file:
    for i,line in enumerate( tpl_file.readlines() ):
        if i==0:
            result_file.write(line)
            continue
        tmp = line.strip().split(",")
        user_id = tmp[0]
        result_file.write( user_id + "," + str(total_visit[user_id]*7/30) + "\n" )
"""
#make train_input and predit_data for each user
train_x = []
train_y = []
predit_info = {}
for i,user_id in enumerate( visit_info_acc ):
    predit_user_info = np.zeros( len(acc_header) )
    for j,interval in enumerate( visit_info_acc[user_id] ):
        train_x.append(visit_info_acc[user_id][interval])
        train_y.append( visit_count_acc[user_id][interval] )
        predit_user_info = predit_user_info + visit_info_acc[user_id][interval]
        #print(visit_info_acc[user_id][interval])
        #print(visit_count_acc[user_id][interval])
    predit_user_info = predit_user_info/4
    predit_user_info[0] = 5
    predit_info[ user_id ] = predit_user_info
    
    #print(predit_user_info)

print("start training")
clf = SVR()
#clf = LogisticRegression()
clf.fit(train_x, train_y)
print("end training")

print("start prediting")
with open(tpl_path , "r", encoding = "utf8") as tpl_file, open(result_path , "w", encoding = "utf8") as result_file:
    for i,line in enumerate( tpl_file.readlines() ):
        if i==0:
            result_file.write(line)
            continue
        tmp = line.strip().split(",")
        user_id = tmp[0]
        
        if user_id not in predit_info:
            result_file.write( user_id + ",0\n" )
            continue

        predit_result = clf.predict( [ predit_info[user_id] ] )
        result_file.write( user_id + "," + str( predit_result[0] ) + "\n" )
