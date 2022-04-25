from fastapi import FastAPI
from init_model import *

app = FastAPI()


@app.get('/bmtuans')
def abc():
    return LSTM