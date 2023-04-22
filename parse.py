import json


def prepare_train(file):
    with open(file, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    formatted_data = []

    for row in train_data:
        text = row['text']
        entities = []
        for i in range(len(row['extracted_part']['text'])):
            start = row['extracted_part']['answer_start'][0]
            end = row['extracted_part']['answer_end'][0]
            label = row['label']
            entities.append((start, end, label))
        formatted_data.append((text, {"entities": entities}))

    return formatted_data


def to_dict(id, text, label):
    return {"id": id, "text": text, "label": label}
