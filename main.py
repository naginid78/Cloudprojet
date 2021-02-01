from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from joblib import load
from sklearn.ensemble import RandomForestRegressor

import os
import re

app = FastAPI()


def check_price(h:int=None, d:datetime=None):

    reg = load('ml/clf.joblib')
    data_input=[[h,d]]
    price=reg.predict(data_input)
    return "The car you choose has an estimated price of ${}".format(price)
    

@app.get("/", response_class=HTMLResponse)
def root(request:Request):
    return templates.TemplateResponse("index.html",{"request": request})

@app.post("/predict/", response_class=HTMLResponse)
async def predict(request:Request, text: str = Form(...)):
    result = check_msg(text)
    return templates.TemplateResponse("predict.html",{"request": request, "text":text, "result":result})
