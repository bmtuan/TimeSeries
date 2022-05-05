from pymongo import MongoClient

def init_db(): 
    client = MongoClient(
        host="202.191.57.62",
        port=27017,
        username="root",
        password="98859B9980A218F6DAD192B74781E15D",
    )

    db_fimi = client["fimi"]
    mycol = db_fimi["MRA_fimi"]
    return mycol