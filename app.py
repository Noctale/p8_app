import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
import json
from json import JSONEncoder
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
from tensorflow.keras import models

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

app = FastAPI()
model = models.load_model('best_model.h5')
   
    
@app.post('/mask_predict')
async def predict_mask(file: UploadFile = File(...)):
    #Bloc preprocessing
    #y_img = load_image_into_numpy_array(await file.read())
    
    y_img = Image.open(BytesIO(await file.read()))
    y_img = np.array(y_img)
    
    #y_img = np.asarray(y_img, dtype="float32")
    y_img = cv2.resize(y_img, (128, 128))
    #y_img = np.asarray(y_img, dtype="int32")
    
    #Bloc prédiction
    batch_img = np.empty((32, 128, 128, 3))
    batch_img[0] = y_img
    y_pred = model.predict(batch_img)
    
    #Bloc serialisation
    numpyData = {"array": y_pred[0]}
    encodedNumpyData = json.dumps(numpyData, cls=NumpyArrayEncoder)
    
    response = Response(encodedNumpyData)
    return response


    
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
