import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from application import get_lime_exp


app = FastAPI()

#file: UploadFile = File(...)
@app.post('/enhanced/interpretability/explain')
async def upload_file(file: UploadFile = File(...)):
    
    if not file.filename:
        raise HTTPException(status_code=400, detail='No selected file')

    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        raise HTTPException(status_code=400, detail='Image must be jpg or png format!')


    try:
        file_path = 'temp.png'
        with open(file_path, 'wb') as f:
            f.write(file.file.read())


        # Call the processing function from application.py
        # path_for_input_image = 'test_images\MI_2619_874.png'
        results = get_lime_exp(file_path)
        
        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)