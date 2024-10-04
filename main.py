from typing import Optional
from fastapi import FastAPI, File, UploadFile, Request, Form
import utils_main
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

# app object 
app = FastAPI()

# mounting images and templates 
app.mount("/static", StaticFiles(directory= "static"), name="static")  
template = Jinja2Templates(directory= "templates")   


# @app.get("/")
# def read_root():
#     return {"Quote": "What Allah does, it is a great decision and it has a postive impact"}

@app.get("/")
def home(request: Request):
    return template.TemplateResponse("index.html", {"request":request})

@app.post("/")
def demo_home(request: Request, file:UploadFile= File(...)):
    # return template.TemplateResponse("prediction.html", {"request":request})
    result = None
    error = None

    try:
        result = utils_main.get_prediction(input_img=file)
    except Exception as e:
        error = e
    return template.TemplateResponse("index.html", {"request":request, "result":result, "error":error})


# dataset page 
@app.get("/about")
def home(request: Request):
    return template.TemplateResponse("about.html", {"request":request})

# dataset page 
@app.get("/dataset")
def home(request: Request):
    return template.TemplateResponse("dataset.html", {"request":request})

# dataset page 
@app.get("/researcher")
def home(request: Request):
    return template.TemplateResponse("researcher.html", {"request":request})


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    return utils_main.get_result(input_img=file, is_api=True)

@app.post("/pred")
async def demo_predict(request:Request, file:UploadFile= File(...)):
    # return utils.get_result(img_file= file)
    # result= utils.test_image(input_img=file)
    # result = utils.get_features_n_gradcam(input_img=file)

    # result = utils.get_prediction(input_img=file)
    # return template.TemplateResponse("prediction.html", {"request":request, "result":result})
    result = None
    error = None

    try:
        result = utils_main.get_prediction(input_img=file)
    except Exception as e:
        error = e
    return template.TemplateResponse("prediction.html", {"request":request, "result":result, "error":error})