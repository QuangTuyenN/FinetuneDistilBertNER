# Import library for build API
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fastapi.middleware.cors import CORSMiddleware

from transformers import pipeline

checkpoint = "quangtuyennguyen/Vi-DistilBert-NER"
pipe = pipeline('token-classification', model=checkpoint, aggregation_strategy='simple')


def name_entity(text):
    ner = pipe(text)
    for entity in sorted(ner, key=lambda x: x['start'], reverse=True):
        label = entity['entity_group']
        start = entity['start']
        end = entity['end']
        word = text[start:end]
        text = text[:start] + str({word: label}) + text[end:]
    return text


# initial FastAPI app
app_ner = FastAPI()


# initial model input data api
class InputData(BaseModel):
    text: str


# CORS setting
app_ner.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả các nguồn (nếu muốn giới hạn, thay "*" bằng danh sách các URL cụ thể)
    allow_credentials=True,
    allow_methods=["*"],  # Cho phép tất cả các phương thức (GET, POST, PUT, DELETE, v.v.)
    allow_headers=["*"],  # Cho phép tất cả các headers
)


@app_ner.post("/process")
def process_data(input_data: InputData):
    input_to_ner = input_data.text
    text = name_entity(input_to_ner)
    return {
        "text": text
    }

