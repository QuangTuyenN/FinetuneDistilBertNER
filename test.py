from transformers import pipeline

checkpoint = "quangtuyennguyen/Vi-DistilBert-NER"
pipe = pipeline('token-classification', model=checkpoint, aggregation_strategy='simple')

text = "tớ sẽ qua đón bạn Quang Tuyền đi chơi trung thu ở Tam Kỳ vào tối mai lúc 9 giờ nhé"
ner = pipe(text)
for entity in sorted(ner, key=lambda x: x['start'], reverse=True):
    label = entity['entity_group']
    start = entity['start']
    end = entity['end']
    word = text[start:end]
    text = text[:start] + str({word: label}) + text[end:]

print(text)

# tớ sẽ qua đón bạn {'Quang Tuyền': 'PERSON'} đi chơi trung thu ở {'Tam Kỳ': 'LOCATION'} vào {'tối mai': 'DATETIME'} lúc 9 {'giờ': 'DATETIME'} nhé