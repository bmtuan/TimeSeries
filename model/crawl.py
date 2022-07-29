from pymongo import MongoClient
import pandas as pd
import argparse
from datetime import datetime
parser = argparse.ArgumentParser()
parser.add_argument("-id","--device_id",help="ID of sensor fimi",type=int, default=3)
args = parser.parse_args()

client = MongoClient(
        host="202.191.57.62",
        port=27017,
        username="root",
        password="98859B9980A218F6DAD192B74781E15D",
    )
start = datetime(2022,3,1,0,0,0)
end = datetime(2022,6,10,0,0,0)
db_fimi = client["fimi"]
mycol = db_fimi["sensor"]
for i in range(30):
#    if i == 12: 
    cursor = mycol.find({'device_id':f'fimi_{i}', 'time': {'$gte': start, '$lte': end}}).sort([('time',1)])
    print(f'fimi_{i}')
    PM2_5 = []
    time = []
    PM2_5 = []
    CO = []
    SO2 = []
    NO2 = []
    PM1_0 = []
    PM10 = []
    temperature = []
    humidity = []

    for x in cursor:
        time.append(x['time'])
        PM2_5.append(x['PM2_5'])
        CO.append(x['CO'])
        SO2.append(x['SO2'])
        NO2.append(x['NO2'])
        PM1_0.append(x['PM1_0'])
        PM10.append(x['PM10_0'])
        temperature.append(x['temperature'])
        humidity.append(x['humidity'])

    dicts = {
        'time': time, 
        'PM2_5': PM2_5,
        'CO': CO,
        'SO2': SO2,
        'NO2': NO2,
        'PM1_0': PM1_0,
        'PM10': PM10,
        'temperature': temperature,
        'humidity': humidity
        }

    df_result = pd.DataFrame(dicts)
    df_result.to_csv(f'/media/aimenext/disk1/tuanbm/TimeSerires/model/data/2506/sensor_{i}.csv')