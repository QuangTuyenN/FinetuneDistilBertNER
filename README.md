This project finetune DistilBERT for NER in Vietnamese language

It will classify name entity in sentence in 4 class: DATETIME, LOCATION, ORGANIZATION, PERSON

Pretrain model: distilbert/distilbert-base-multilingual-cased

Dataset: Minggz/Vi-Ner

Train model, eval, push to Huggingface hub by train.py

Deploy model using fastapi by server_fastapi.py

run: uvicorn server_fastapi:app_ner --host 0.0.0.0 --port 1234 --reload