from pymongo import MongoClient

def connect_fimi_MRA(): 
    client = MongoClient(
        host="202.191.57.62",
        port=27017,
        username="root",
        password="98859B9980A218F6DAD192B74781E15D",
    )

    db_fimi = client["fimi"]
    mycol = db_fimi["fimi_MRA"]
    return mycol

def connect_signal_MRA(): 
    client = MongoClient(
        host="202.191.57.62",
        port=27017,
        username="root",
        password="98859B9980A218F6DAD192B74781E15D",
    )

    db_fimi = client["fimi"]
    mycol = db_fimi["signal_MRA"]
    return mycol