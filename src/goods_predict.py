
# coding: utf-8

# In[100]:


import numpy as np
import pandas as pd
import re
import os
from datetime import datetime, date
from chinese_calendar import is_workday, is_holiday
from datetime import timedelta
from scipy.optimize import root,fsolve
import math
import chinese_calendar as calendar  # 中国日历，依赖库


path=open("./train/发货记录.csv",encoding="utf-8")
send_goods=pd.read_csv(path,engine="python")
send_goods.infer_objects()
send_goods["day"]=send_goods["day"].astype(int)
send_goods["start_city_id"]=send_goods["start_city_id"].astype(int)


# In[127]:
def getDetailHoliday(date):
    april_last = datetime.strptime(str(date),'%Y%m%d')
    on_holiday, holiday_name = calendar.get_holiday_detail(april_last)
    if(on_holiday==True  and  holiday_name is not None):
        return 1
    else:
        return 0
#由日期获得星期
def  getWeekday(date):
    date_time = datetime.strptime(date,'%Y%m%d')
    return date_time.weekday()
def getHoliday(date):
    return is_holiday(datetime.strptime(str(date),'%Y%m%d'))
def toNormalDate(date):
    return date[0:4]+"-"+date[4:6]+"-"+date[6:8]
def getBeforeDay(date,day):
    time=datetime.strptime(date,'%Y%m%d')-timedelta(days=day)
    return time.strftime('%Y-%m-%d')
def  containSum(sum_array,arr):
    count=0
    for a in arr:
        count+=sum_array.count(a)
    return count
def getBeforeDayNew(date,day):
    time=datetime.strptime(str(date),'%Y%m%d')-timedelta(days=day)
    return time.strftime('%Y%m%d')
fn=[0,0,0,0,0,0,0,0,0,0,0,0,0,0]
def fun(x):
    sum_x=0
    for i in fn:
        sum_x+=(i-x)**2
    return math.sqrt(sum_x)
def getLossMin(f):
    global fn
    fn=f
    sol_root = root(fun,[0])
    fn=[0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    return sol_root.x
feature_columns=[
    "sendGood_sum",
    "truck_length_other","truck_length_3_9","truck_length_9_15","truck_length_15_over",  
    "truck_weight_other","truck_weight_5_10","truck_weight_10_15","truck_weight_15_20","truck_weight_20_over",

    "truck_type_-1","truck_type_0","truck_type_1","truck_type_2","truck_type_3","truck_type_4","truck_type_5","truck_type_6",
    "truck_type_7","truck_type_8","truck_type_9","truck_type_10","truck_type_11","truck_type_12","truck_type_13","truck_type_14",
    "truck_type_15",
    "handling_type_0","handling_type_1","handling_type_2","handling_type_3","handling_type_4","handling_type_5","handling_type_6"    
]

#提取邻近周期发货记录特征
def extractSendGoodFe(day,city_code):  
    feature={}
    goodss=send_goods[(send_goods["day"]<day)&
                      (send_goods["day"]>=int(getBeforeDayNew(day,7)))&
                (send_goods["start_city_id"]>=city_code) & (send_goods["start_city_id"]<=city_code+99)
                ]
    goodss["start_city_id"]=goodss["start_city_id"].astype(int)
    ll=len(goodss)
    feature["sendGood_sum"]=ll
    
    #统计卡车长度特征
    goodss["truck_length"]=goodss["truck_length"].apply(lambda x: x.split(",")[0])   
    goodss["truck_length"]=goodss["truck_length"].astype(float)   
    
    feature["truck_length_other"]= goodss[(goodss["truck_length"]<3)].shape[0]
    feature["truck_length_3_9"]=goodss[(goodss["truck_length"]>=3) & (goodss["truck_length"]<9)].shape[0]
    feature["truck_length_9_15"]=goodss[(goodss["truck_length"]>=9) & (goodss["truck_length"]<15)].shape[0]
    feature["truck_length_15_over"]=goodss[(goodss["truck_length"]>=15)].shape[0]
    goodss["truck_weight"]=goodss["truck_weight"].astype(float)
    
    feature["truck_weight_other"]=goodss[(goodss["truck_length"]<5)].shape[0]
    feature["truck_weight_5_10"]=goodss[(goodss["truck_weight"]>=5)& (goodss["truck_length"]<10)].shape[0]
    feature["truck_weight_10_15"]=goodss[(goodss["truck_weight"]>=10) & (goodss["truck_length"]<15)].shape[0]
    feature["truck_weight_15_20"]=goodss[(goodss["truck_weight"]>=15) & (goodss["truck_length"]<20)].shape[0]
    feature["truck_weight_20_over"]=goodss[(goodss["truck_weight"]>=20)].shape[0]

    for i in range(17):
        feature["truck_type_"+str(i-1)]=goodss[goodss["truck_type"]==i-1].shape[0]
    for i in range(7):
        feature["handling_type_"+str(i)]=goodss[goodss["handling_type"]==i].shape[0]
    re_feature={}
    if ll != 0:
        for  k,v in feature.items():
            re_feature[k]=v/ll
    else:
        for  k,v in feature.items():
            re_feature[k]=0
    re_feature["sendGood_sum"]=ll
    return re_feature
    
wind=[
    "多云",
    "少云",
    "局部多云"]
ying= ["阴","阴天"]
sunny=[ "晴间多云","晴"]

bad_whether=[
     "暴雪",
    "大雪暴雪",
    "大雪",
    "中雪大雪",
    "中雪",
     "小雪中雪",
      "暴雨",
    "大雨暴雨",
    "大雨", 
    "中雨大雨",
    "雪",
    "雨夹雪",
    "中雨",
    "雷雨",
     "雷阵雨",
    "小雪",
    "阵雪",
    "小雨中雨",
    "小雨",
    "阵雨",
    "雨",
    "零散阵雨",
    "零散雷雨",
    "刮风",
    "雾",
    "薄雾"
]

#计算两天之间的相隔天数
def CalDiffDay(day1,day2):
    date_time1 = datetime.strptime(day1,'%Y%m%d')
    date_time2 = datetime.strptime(day2,'%Y%m%d')
    a= abs(date_time1-date_time2)
    return int(a.days)

#提取初步特征，整合csv文件
def getGoodsCSV(driver,code,goods):
    goods_3201=goods[(goods["day"]<=20180201)|(goods["day"]>=20180310)]#剔除春运和春节阶段的数据
    #创建新的dataframe
    data_columns=["city","count_fmin","distance",
                "sendGood_sum",
                "truck_length_other","truck_length_3_9","truck_length_9_15","truck_length_15_over",  
                "truck_weight_other","truck_weight_5_10","truck_weight_10_15","truck_weight_15_20","truck_weight_20_over",

    
                "truck_type_-1","truck_type_0","truck_type_1","truck_type_2","truck_type_3","truck_type_4","truck_type_5","truck_type_6",
                "truck_type_7","truck_type_8","truck_type_9","truck_type_10","truck_type_11","truck_type_12","truck_type_13","truck_type_14",
                "truck_type_15",
                "handling_type_0","handling_type_1","handling_type_2","handling_type_3","handling_type_4","handling_type_5","handling_type_6",  
                  
                  
                "date","weekday",
                'week_sum', 'week_mean', 'week_std', 'week_max','week_min',"week_median",
                "week_diff1","week_diff2","week_diff3","week_diff4","week_diff5","week_diff6",
                "week_diff_diff1", "week_diff_diff2", "week_diff_diff3","week_diff_diff4", "week_diff_diff5",
                 "whether","temperature","wind","label"                
                 ]
    feature_csv3201 = pd.DataFrame(columns=data_columns)#生成空的pandas表
    look_back=7
    index_start=look_back
    for j in range(len(goods_3201)-look_back-7):
        fmin= getLossMin(goods_3201.iloc[index_start-look_back:index_start]["count"].tolist())[0]  
        holiday_check=goods_3201.iloc[index_start-look_back:index_start+7]["day"].tolist()
        index_start+=1
        flag=1
        for h in holiday_check:#不可以有假期
            if getDetailHoliday(h)==1:
                flag=0
                break    
        if CalDiffDay(str(holiday_check[0]),str(holiday_check[len(holiday_check)-1]))>=17:#不能断层
            flag=0 
        if flag==0:
            continue
        _good=np.array(goods_3201.iloc[index_start-look_back:index_start]["count"].tolist())
        week_sum=_good.sum()
        week_mean=_good.mean()
        week_std=_good.std()
        week_max=_good.max()
        week_min=_good.min()
        week_median=np.median(_good)
        week_diff=np.diff(_good)
        week_diff_diff=np.diff(week_diff)
        if flag==1:
            for i in range(7):
                feature={ "city":code,"count_fmin":0,"date":0,"weekday":0,"whether":"none","temperature":"none","wind":"none","label":0 }
                data_now=goods_3201.iloc[index_start+i]["day"]#获得日期
                label=goods_3201.iloc[index_start+i:index_start+1+i]["count"].tolist()[0]#获得日期
                week=getWeekday(str(data_now))
                send_ff=extractSendGoodFe(int(data_now),int(code))
                feature.update(send_ff)
                if getDetailHoliday(data_now)==1:
                    break
                feature["count_fmin"]=fmin
                feature["date"]=data_now
                feature["weekday"]=week
                feature["label"]=label 
                
                feature["distance"]=i
                
                feature["week_sum"]=week_sum
                feature["week_mean"]=week_mean
                feature["week_std"]=week_std
                feature["week_max"]=week_max
                feature["week_min"]=week_min
                feature["week_median"]=week_median  
                for i in range(look_back-1):
                    feature["week_diff"+str(i+1)]=float(week_diff[i])
                for i in range(look_back-2):
                    feature["week_diff_diff"+str(i+1)]=float(week_diff[i])   
                msgWhether=whether[(whether["date"]==getBeforeDay(str(data_now),0))
                                   &((whether["code"])>=code)
                                   &((whether["code"])<=code+99)]      
                feature["whether"]=(str(msgWhether["weather"].tolist()))
                feature["temperature"]=(str(msgWhether["temperature"].tolist()))
                feature["wind"]=(str(msgWhether["wind"].tolist()))
                feature_csv3201=feature_csv3201.append(feature,ignore_index=True)
    return  feature_csv3201

#提取天气特征
def  getFeature(feature=None):
    feature_csv3201=feature
    feature=["wind_","ying","sunny","bad_whether","temp_low_sum","temp_high_sum","temp_low_mean","temp_high_mean","temp_median_mean","temp_median_sum"
            ]
    for  aa in feature:
        feature_csv3201[aa]=0
    for i in range(len(feature_csv3201)):
        ##天气特征提取
        whether_type=[]
        wh=eval(feature_csv3201.iloc[i]["whether"])
        for each in wh:
            whether_type+=each.split("/")            
        if(len(whether_type)>0):           
            feature_csv3201.loc[i,"wind_"]=containSum(whether_type,wind)/len(whether_type)
            feature_csv3201.loc[i,"ying"]=containSum(whether_type,ying)/len(whether_type)
            feature_csv3201.loc[i,"sunny"]=containSum(whether_type,sunny)/len(whether_type)
            feature_csv3201.loc[i,"bad_whether"]=containSum(whether_type,bad_whether)/len(whether_type)
        else:
            feature_csv3201.loc[i,"wind_"]=0
            feature_csv3201.loc[i,"ying"]=0
            feature_csv3201.loc[i,"sunny"]=0
            feature_csv3201.loc[i,"bad_whether"]=0
                     ##温度特征提取
        temp=eval(feature_csv3201.iloc[i]["temperature"])
        temp_low=[]
        temp_high=[]
        temp_median=[]
        for  te in temp:
            if(te.split("/")[0]=="" or te.split("/")[1] is ''):
                break
            else:
                temp_low.append(int(te.split("/")[0]))
                temp_high.append(int(te.split("/")[1]))
                temp_median.append((int(te.split("/")[0])+int(te.split("/")[1]))/2)

        np_low=np.array(temp_low)
        np_high=np.array(temp_high)
        np_median=np.array(temp_median)
        
        feature_csv3201.loc[i,"temp_low_sum"]=np.sum(np_low)
        feature_csv3201.loc[i,"temp_high_sum"]=np.sum(np_high)
        feature_csv3201.loc[i,"temp_low_mean"]=np_low.mean()
        feature_csv3201.loc[i,"temp_high_mean"]=np_high.mean()
        feature_csv3201.loc[i,"temp_median_mean"]=np_median.mean()
        feature_csv3201.loc[i,"temp_median_sum"]=np.sum(np_median)

    del feature_csv3201["temperature"]
    del feature_csv3201["whether"]
    del feature_csv3201["wind"]
    return feature_csv3201
print("finished!")


# In[128]:


import os
path_whether=open("./train/城市天气记录.csv",encoding="gb2312")
whether_1=pd.read_csv(path_whether,engine='python')
path_whether=open("./train/天气数据.csv",encoding="gb2312")
whether_2=pd.read_csv(path_whether,engine='python')
whether_2["date"]=whether_2["date"].apply(lambda x: str(datetime.strptime(x,'%Y/%m/%d').strftime("%Y-%m-%d")))
whether=pd.concat([whether_1,whether_2],axis=0)

a=whether["weather"].tolist()
whether_type=[]
for  i in a:
    whether_type+=i.split("/")
w=np.array(whether_type)
wheath_fea=np.unique(w).tolist()
wheath_fea.remove('')


path_goods_3201=open("./train/货量表/320100.csv",encoding="utf-8")
goods_1=pd.read_csv(path_goods_3201)
path_drivers_3201=open("./train/司机量表/320100.csv",encoding="utf-8")
result_1=getGoodsCSV(pd.read_csv(path_drivers_3201),320100,goods_1)

path_goods_3201=open("./train/货量表/320200.csv",encoding="utf-8")
goods_2=pd.read_csv(path_goods_3201)
path_drivers_3201=open("./train/司机量表/320200.csv",encoding="utf-8")
result_2=getGoodsCSV(pd.read_csv(path_drivers_3201),320200,goods_2)

path_goods_3201=open("./train/货量表/320300.csv",encoding="utf-8")
goods_3=pd.read_csv(path_goods_3201)
path_drivers_3201=open("./train/司机量表/320300.csv",encoding="utf-8")
result_3=getGoodsCSV(pd.read_csv(path_drivers_3201),320300,goods_3)

path_goods_3201=open("./train/货量表/320400.csv",encoding="utf-8")
goods_4=pd.read_csv(path_goods_3201)
path_drivers_3201=open("./train/司机量表/320400.csv",encoding="utf-8")
result_4=getGoodsCSV(pd.read_csv(path_drivers_3201),320400,goods_4)

path_goods_3201=open("./train/货量表/320500.csv",encoding="utf-8")
goods_5=pd.read_csv(path_goods_3201)
path_drivers_3201=open("./train/司机量表/320500.csv",encoding="utf-8")
result_5=getGoodsCSV(pd.read_csv(path_drivers_3201),320500,goods_5)

path_goods_3201=open("./train/货量表/320600.csv",encoding="utf-8")
goods_6=pd.read_csv(path_goods_3201)
path_drivers_3201=open("./train/司机量表/320600.csv",encoding="utf-8")
result_6=getGoodsCSV(pd.read_csv(path_drivers_3201),320600,goods_6)

path_goods_3201=open("./train/货量表/320700.csv",encoding="utf-8")
goods_7=pd.read_csv(path_goods_3201)
path_drivers_3201=open("./train/司机量表/320700.csv",encoding="utf-8")
result_7=getGoodsCSV(pd.read_csv(path_drivers_3201),320700,goods_7)

path_goods_3201=open("./train/货量表/320800.csv",encoding="utf-8")
goods_8=pd.read_csv(path_goods_3201)
path_drivers_3201=open("./train/司机量表/320800.csv",encoding="utf-8")
result_8=getGoodsCSV(pd.read_csv(path_drivers_3201),320800,goods_8)

path_goods_3201=open("./train/货量表/320900.csv",encoding="utf-8")
goods_9=pd.read_csv(path_goods_3201)
path_drivers_3201=open("./train/司机量表/320900.csv",encoding="utf-8")
result_9=getGoodsCSV(pd.read_csv(path_drivers_3201),320900,goods_9)

path_goods_3201=open("./train/货量表/321000.csv",encoding="utf-8")
goods_10=pd.read_csv(path_goods_3201)
path_drivers_3201=open("./train/司机量表/321000.csv",encoding="utf-8")
result_10=getGoodsCSV(pd.read_csv(path_drivers_3201),321000,goods_10)

path_goods_3201=open("./train/货量表/321100.csv",encoding="utf-8")
goods_11=pd.read_csv(path_goods_3201)
path_drivers_3201=open("./train/司机量表/321100.csv",encoding="utf-8")
result_11=getGoodsCSV(pd.read_csv(path_drivers_3201),321100,goods_11)

path_goods_3201=open("./train/货量表/321200.csv",encoding="utf-8")
goods_12=pd.read_csv(path_goods_3201)
path_drivers_3201=open("./train/司机量表/321200.csv",encoding="utf-8")
result_12=getGoodsCSV(pd.read_csv(path_drivers_3201),321200,goods_12)

path_goods_3201=open("./train/货量表/321300.csv",encoding="utf-8")
goods_13=pd.read_csv(path_goods_3201)
path_drivers_3201=open("./train/司机量表/321300.csv",encoding="utf-8")
result_13=getGoodsCSV(pd.read_csv(path_drivers_3201),321300,goods_13)

data_goods=pd.concat([result_1,result_2,result_3,result_4,result_5,result_6,result_7,result_8,
               result_9,result_10,result_11,result_12,result_13],axis=0)
result_goods = data_goods.reset_index(drop=True)#将连接之后的表格索引重置




test_goods_1=getFeature(copy.deepcopy(data_goods))#进一步获取特征

# In[147]:

import copy
import xgboost as xgb
# print(test_goods.shape,test_driver.shape)
from sklearn.preprocessing import LabelEncoder
import copy
median_cahce=copy.deepcopy(test_goods_1)
# median_cahce=median_cahce.dropna(axis=0)
Data_set=median_cahce
Data_set["date"]=Data_set["date"].astype(str)
Data_set["date"]=Data_set["date"].apply(lambda x:x[-2:])
Data_set["date"]=Data_set["date"].apply(lambda x: int(x)%10)
# Data_set=pd.get_dummies(Data_set,columns=["weekday"])
Data_set = Data_set.infer_objects()
del Data_set["city"]
print(Data_set.columns)
label=Data_set["label"]
del Data_set["label"]
del Data_set["distance"]
print(Data_set.columns)


# In[148]:


from sklearn.model_selection import  train_test_split   
from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import GridSearchCV

X_train,X_test,y_train,y_test=train_test_split(Data_set,label,test_size=0.1,random_state=1)
print(X_train.shape,y_train.shape)

model = xgb.XGBRegressor(learning_rate=0.1,
                         n_estimators=6000, 
                         min_child_weight=3, 
                         objective='reg:linear',
                         eval_metric="rmse",
                         n_jobs =20,
                         max_depth=8,
                         reg_lambda =5,
                         random_state=26)

model.fit(Data_set,label)
y_pred=model.predict(X_test)
print(type(y_test))
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
print("rmse:",rmse(y_test,y_pred))    


# In[154]:







#####提取待数据数据特征，并预测结果###########
############################################################################################





from datetime import timedelta

path_whether=open("./train/城市天气记录.csv",encoding="gb2312")
whether_1=pd.read_csv(path_whether,engine='python')
path_whether=open("./train/天气数据.csv",encoding="gb2312")
whether_2=pd.read_csv(path_whether,engine='python')
whether_2["date"]=whether_2["date"].apply(lambda x: str(datetime.strptime(x,'%Y/%m/%d').strftime("%Y-%m-%d")))
whether=pd.concat([whether_1,whether_2],axis=0)


def getWheFeature(parm_whe,para_tem):
    re_feature={}
    feature=["wind_","ying","sunny","bad_whether","temp_low_sum","temp_high_sum","temp_low_mean","temp_high_mean","temp_median_mean","temp_median_sum"
            ]
    for  aa in feature:
        re_feature[aa]=0
        ##天气特征提取
    whether_type=[]
    wh=eval(parm_whe)
    for each in wh:
        whether_type+=each.split("/")
    if(len(whether_type)>0):           
        re_feature["wind_"]=containSum(whether_type,wind)/len(whether_type)
        re_feature["ying"]=containSum(whether_type,ying)/len(whether_type)
        re_feature["sunny"]=containSum(whether_type,sunny)/len(whether_type)
        re_feature["bad_whether"]=containSum(whether_type,bad_whether)/len(whether_type)
    else:
        re_feature["wind_"]=0
        re_feature["ying"]=0
        re_feature["sunny"]=0
        re_feature["bad_whether"]=0
        
    ##温度特征提取
    temp=eval(para_tem)
    temp_low=[]
    temp_high=[]
    temp_median=[]
    for  te in temp:
        if(te.split("/")[0]=="" or te.split("/")[1] is ''):
            break
        else:
            temp_low.append(int(te.split("/")[0]))
            temp_high.append(int(te.split("/")[1]))
            temp_median.append((int(te.split("/")[0])+int(te.split("/")[1]))/2)   
    np_low=np.array(temp_low)
    np_high=np.array(temp_high)
    np_median=np.array(temp_median)
    re_feature["temp_low_sum"]=np.sum(np_low)
    re_feature["temp_high_sum"]=np.sum(np_high)
    re_feature["temp_low_mean"]=np_low.mean()
    re_feature["temp_high_mean"]=np_high.mean()
    re_feature["temp_median_mean"]=np_median.mean()
    re_feature["temp_median_sum"]=np.sum(np_median)
    return re_feature
    
cloums_train_Goods=['count_fmin',#"distance",
                    "sendGood_sum",
    "truck_length_other","truck_length_3_9","truck_length_9_15","truck_length_15_over",  
    "truck_weight_other","truck_weight_5_10","truck_weight_10_15","truck_weight_15_20","truck_weight_20_over",

    "truck_type_-1","truck_type_0","truck_type_1","truck_type_2","truck_type_3","truck_type_4","truck_type_5","truck_type_6",
    "truck_type_7","truck_type_8","truck_type_9","truck_type_10","truck_type_11","truck_type_12","truck_type_13","truck_type_14",
    "truck_type_15",
    "handling_type_0","handling_type_1","handling_type_2","handling_type_3","handling_type_4","handling_type_5","handling_type_6",            
    "date",
    'weekday',
    'week_sum', 'week_mean', 'week_std', 'week_max','week_min',"week_median",
    "week_diff1","week_diff2","week_diff3","week_diff4","week_diff5","week_diff6",
    "week_diff_diff1", "week_diff_diff2", "week_diff_diff3","week_diff_diff4", "week_diff_diff5", 
    "wind_","ying","sunny","bad_whether","temp_low_sum","temp_high_sum","temp_low_mean",
    "temp_high_mean","temp_median_mean","temp_median_sum"]


weekday_num=[0,1,2,3,4,5,6]

def getTestGoodsCSV(drivers,code,date_now):
    goods_3201=drivers
    df = pd.DataFrame(columns =cloums_train_Goods) #创建一个空的dataframe
    feature=[]
    fmin=getLossMin(goods_3201.tail(7)["count"].tolist())[0]#取倒数第7行
    feature.append(fmin)    
    
    print(feature)
    ffff=extractSendGoodFe(20180531,code)#获取城市的特征
    
    feature+=ffff.values()
    feature.append(int(str(date_now)[-2:])%10)
    
    week=int(getWeekday(str(date_now)))
    feature.append(week)
    
    wek_n=goods_3201.tail(7)["count"].tolist()
    week_sum=np.array(wek_n).sum()
    week_mean=np.array(wek_n).mean()
    week_std=np.array(wek_n).std()
    week_max=np.array(wek_n).max()
    week_min=np.array(wek_n).min()
    week_median=np.median(wek_n)
    week_diff=np.diff(np.array(wek_n))
    week_diff_diff=np.diff(week_diff)
    
    feature.append(week_sum)
    feature.append(week_mean)
    feature.append(week_std)
    feature.append(week_max)
    feature.append(week_min)
    feature.append(week_median)
    feature+=week_diff.tolist()
    feature+=week_diff_diff.tolist()
    ###################天气特征######################
    msgWhether=whether[(whether["date"]==toNormalDate(str(date_now)))&((whether["code"])>=code)&((whether["code"])<=code+99)]
    val=getWheFeature(str(msgWhether["weather"].tolist()),str(msgWhether["temperature"].tolist()))
    feature=feature+list(val.values())


    df.loc[0]=feature
    result=int(round(model.predict(df)[0]))  
    
    item_new={}
    item_new["city"]=code
    item_new["day"]=date_now
    item_new["count"]=result 
    return item_new

def getGoodsResult(filename):
    clo=["city","day","count"]
    data_csv=pd.DataFrame(columns=clo)
#     print(filename.split("."))
    code=int(filename.split(".")[0])
    path_goods_3201=open("./train/货量表/"+filename,encoding="utf-8")
    csv_goods=pd.read_csv(path_goods_3201,engine='python')

    goods_1=getTestGoodsCSV(csv_goods,code,20180601)
    goods_2=getTestGoodsCSV(csv_goods,code,20180602)
    goods_3=getTestGoodsCSV(csv_goods,code,20180603)
    goods_4=getTestGoodsCSV(csv_goods,code,20180604)
    goods_5=getTestGoodsCSV(csv_goods,code,20180605)
    goods_6=getTestGoodsCSV(csv_goods,code,20180606)
    goods_7=getTestGoodsCSV(csv_goods,code,20180607)
    
    data_csv=data_csv.append(goods_1,ignore_index=True)
    data_csv=data_csv.append(goods_2,ignore_index=True)
    data_csv=data_csv.append(goods_3,ignore_index=True)
    data_csv=data_csv.append(goods_4,ignore_index=True)
    data_csv=data_csv.append(goods_5,ignore_index=True)
    data_csv=data_csv.append(goods_6,ignore_index=True)
    data_csv=data_csv.append(goods_7,ignore_index=True)
    data_csv.reset_index()
    result =data_csv.sort_values(by='day',ascending = True)
    return result    

# In[155]:

from  xgboost import plot_importance
import matplotlib.pylab as plt
feature_importance = model.feature_importances_
feature_importance = (feature_importance / feature_importance.max())
print('特征：', test.columns)
print('每个特征的重要性：', feature_importance)
fig, ax=plt.subplots(figsize=(25,25))
plot_importance(model,ax=ax,max_num_features =130)
plt.title("Featurertances")
plt.show()

# In[156]:
#得到货物量预测结果
rootDir="./train/货量表"
pd_submitGoods = pd.DataFrame()
result=[]
for filename in os.listdir(rootDir):
    aa=getGoodsResult(filename)
    pd_submitGoods=pd_submitGoods.append(aa)

# In[157]:

pd_submitGoods=pd_submitGoods.sort_values(by=['city',"day"],ascending = True)
pd_submitGoods.head(100)

# In[158]:

pd_submitGoods=pd_submitGoods.reset_index(drop=True)
pd_submitGoods=pd_submitGoods.rename(columns={"count":"cargo_count"})
pd_submitGoods.to_csv("./goods.csv",index=0)




