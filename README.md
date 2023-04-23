# Named entity recognition (NER) with spaCy
A Python script that allows you to extract the desired piece of text from a document by label.

Test task
## Usage
Install all packages:
```
pip install -r requirements.txt
```
```
usage: model.py [-h] [-p] [-f] [-it ITER] [-d DROP] [-e] [-i path] [-o path]
                [-r] [-v]

NER

optional arguments:
  -h, --help            show this help message and exit
  -p, --predict         use it for prediction from a text input
  -f, --fit             use it for training model
  -it ITER, --iter ITER
                        Number of iterations during training
  -d DROP, --drop DROP  Dropout rate
  -e, --evaluate        use it to evaluate model performance
  -i path, --input path
                        path to file in json with input data for fit / predict
                        / evaluate
  -o path, --out path   path to output in predictions.json
  -r, --rollback        return the model to its original state
  -v, --verbose         verbose output

Example: !python model.py -f -i ./example_data/train.json -it 2 -drop 0.2
```
## Format
**For prediction:**

```
"id": 311837655,
"text": "��������� � ���������� ������������ �������� ��� ������� �0124200000622005291 ����� ���������� ����� ��������� 0124200000622005291 ������������ ������� ������� �������� ����������� ������ ��������� ��������� �������� �������� ������, ������������ ��� ����������� ���������������� � �������, �� �� ���� ����������� ������� ����������� �� ���������� ����������� ������� �����: ������ �� 2022 ��� ������ �� 2023 ��� ������ �� 2024 ��� ����� �� ����������� ���� 541976.34 541976.34 0.00 0.00 0.00 ����� ���������� ��������� �������� �� �������� �� ����� ���������� ��������� �������������� �� ���� ����������� ������� ����� ����� ��������� (� ������ ���������) �� 2022 ��� �� 2023 ��� �� 2024 ��� �� 2025 ��� 1 2 3 4 541976.34 0 0 0 541976.34 0.00 0.00 0.00 ��� ����� �������� ����� ��������� (� ������ ���������) �� 2022 ��� �� 2023 ��� �� 2024 ��� �� 2025 ��� 1 2 3 4 5 244 541976.34 0 0 0 ����� 541976.34 0.00 0.00 0.00 ����� �������� ������, ���������� ������ ��� �������� ������ 163002, �. �����������, ��. �������� �����, �.7, ��������� ��������� ������������� ����������� �������������� ������ �� ���������� ��������� � ������������ �� ��. 95 ������ � 44-�� �� ����������� ������ ����������� ������ �� ��������� ����������� ���������� ��������� ��������� ����������� ���������� ��������� ������ ����������� ���������� ��������� 10.00% ������� ����������� ���������� ���������, ���������� � ����������� � ������������ � ����������� � ��������� �� ������������� ������� ��������� ��������� \"����� ���������� �����\" 03224643110000002400 \"����� �������� �����\" 20246�24120 \"���\" 011117401 \"������������ ��������� �����������\" ��������� ����������� ����� ������//��� �� ������������� ������� � ��������� ����������� ������ �. ����������� \"����� ������������������ �����\" 40102810045370000016 ���������� � �������� �������� ������, ������, ������ ��������� �������� �������� ������, ������, ������ �� ���������� � ����������� � ������������ ������������ ������ � ������������ � �������� ��������� ���������� � �������� ������������� ������ � ������������ � �������� ��������� ����, �� ������� ��������������� �������� � ������������ � �������� ��������� ����������� ����������� ������������ ����������� ����������� ������������ �� ��������� ���������� � ���������� � (���) ������������ ������������� ��������� ���������� ��� ������������ �� ��������� � ���������� ������������ �������� ��� ������� �0124200000622005291 ����� ���������� ����� ��������� 0124200000622005291 ������������ ������� ������� �������� ����������� ������ ��������� ��������� �������� �������� ������, ������������ ��� ����������� ���������������� � �������,",
"label": "����������� ���������� ���������"
```

**For fit:**
```
"id": 311837655,
"text": "����������� ������ ����������� ������ �� ��������� ����������� ���������� ��������� ��������� ����������� ���������� ��������� ������ ����������� ���������� ��������� 10.00%",
"label": "����������� ���������� ���������"
"extracted_part": {
    "text": ["����������� ���������� ���������� ��������� ����������� � ������� 5%."
    ],
    "answer_start": [
        1251
    ],
    "answer_end":
        1320
    ]
}
```

**Similar format for evaluation as in fit.**
## Training and examples
To see how the model was trained, see `research.ipynb` jupiter notebook.<br />
To see how to use the model, see `usage.ipynb` jupiter notebook.<br />
See also the sample data in [examples_data](example_data).

## Sources
- https://spacy.io/