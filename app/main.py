import io
import PIL
import requests
from io import BytesIO
from diffusers import PaintByExamplePipeline
from fastapi import FastAPI, UploadFile, Form, Response
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from transformers import AutoImageProcessor
import torch
from transformers import AutoModelForImageClassification
from transformers import pipeline
from typing import Union


# This model is a `zero-shot-classification` model.
# It will classify text, except you are free to choose any label you might imagine
classifier = pipeline(model="facebook/bart-large-mnli")
result = classifier(
    "I have a problem with my iphone that needs to be resolved asap!!",
    candidate_labels=["urgent", "not urgent", "phone", "tablet", "computer"],
)
print(result)


vision_classifier = pipeline(model="google/vit-base-patch16-224")
preds = vision_classifier(
    images="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
)
preds = [{"score": round(pred["score"], 4), "label": pred["label"]}
         for pred in preds]
print(preds)


image = 'pipeline-cat-chonk.jpeg'
image = Image.open(image)

image_processor = AutoImageProcessor.from_pretrained(
    "google/vit-base-patch16-224")
inputs = image_processor(image, return_tensors="pt")

model = AutoModelForImageClassification.from_pretrained(
    "google/vit-base-patch16-224")
with torch.no_grad():
    logits = model(**inputs).logits

predicted_label = logits.argmax(-1).item()
result = model.config.id2label[predicted_label]

print(result)


model = AutoModelForImageClassification.from_pretrained(
    "google/vit-base-patch16-224")
vision_classifier = pipeline(model="google/vit-base-patch16-224")
classifier = pipeline("sentiment-analysis", model="stevhliu/my_awesome_model")
pbe_pipe = PaintByExamplePipeline.from_pretrained(
    "Fantasy-Studio/Paint-by-Example",
    torch_dtype=torch.float16,
)

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.post("/login/")  # python 3.6+ non_annotated
async def login(username: str = Form(), password: str = Form()):
    print(password)
    return {"username": username}


@app.post("/predict/")
async def predict(text: str = Form()):
    return {"result": classifier(text)}


@app.post("/files/")
async def create_file(file: bytes = File()):
    return {"file_size": len(file)}


@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    return {"filename": file.filename}


@app.post("/predict_image/")
async def predict_image(file: UploadFile):
    print(file.content_type)

    contents = await file.read()
    img = Image.open(io.BytesIO(contents))
    preds = vision_classifier(
        images=img
    )
    preds = [{"score": round(pred["score"], 4), "label": pred["label"]}
             for pred in preds]

    return {"preds": preds}


@app.post("/predict_image2/")
async def predict_image(file: UploadFile):

    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    image_processor = AutoImageProcessor.from_pretrained(
        "google/vit-base-patch16-224")
    inputs = image_processor(image, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_label = logits.argmax(-1).item()
    result = model.config.id2label[predicted_label]

    return {"result": result}


@app.post("/paintbyexample/")
async def paintbyexample(file: UploadFile):

    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    def download_image(url):
        response = requests.get(url)
        return PIL.Image.open(BytesIO(response.content)).convert("RGB")

    # img_url = (
    #     "https://raw.githubusercontent.com/Fantasy-Studio/Paint-by-Example/main/examples/image/example_1.png"
    # )
    # mask_url = (
    #     "https://raw.githubusercontent.com/Fantasy-Studio/Paint-by-Example/main/examples/mask/example_1.png"
    # )
    # example_url = "https://raw.githubusercontent.com/Fantasy-Studio/Paint-by-Example/main/examples/reference/example_1.jpg"

    # mask_image = download_image(mask_url).resize((512, 512))
    # example_image = download_image(example_url).resize((512, 512))

    init_image = image.resize((512, 512))
    mask_image = Image.open('merged_sample_gd5.png').resize((512, 512))
    example_image = Image.open('suit4.jpg').resize((512, 512))

    pipe = pbe_pipe.to("cuda")

    image = pipe(image=init_image, mask_image=mask_image,
                 example_image=example_image).images[0]

    image.show()
    image.save(io.BytesIO(), format='png')
    return Response(content=io.BytesIO().getvalue(), media_type='image/png')
