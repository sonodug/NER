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

`{`<br />
&emsp;     `"id": 311837655,`<br />
&emsp;    `"text": "ќбеспечение за€вки ќбеспечение за€вок не требуетс€ ќбеспечение исполнени€ контракта “ребуетс€ обеспечение исполнени€ контракта –азмер обеспечени€ исполнени€ контракта 10.00%",`<br />
&emsp;    `"label": "обеспечение исполнени€ контракта"`<br />
`}`

**For fit:**

`{`<br />
&emsp;     `"id": 311837655,`<br />
&emsp;    `"text": "ќбеспечение за€вки ќбеспечение за€вок не требуетс€ ќбеспечение исполнени€ контракта “ребуетс€ обеспечение исполнени€ контракта –азмер обеспечени€ исполнени€ контракта 10.00%",`<br />
&emsp;    `"label": "обеспечение исполнени€ контракта"`<br />
&emsp;  `"extracted_part": {`<br />
&emsp;&emsp;       `"text": [`<br />
&emsp;&emsp;&emsp;      `"ќбеспечение исполнени€ насто€щего  онтракта установлено в размере 5%."`<br />
&emsp;&emsp;      `],`<br />
&emsp;&emsp;&emsp;      `"answer_start": [`<br />
&emsp;&emsp;&emsp;&emsp;        `1251`<br />
&emsp;&emsp;&emsp;       `],`<br />
&emsp;&emsp;&emsp;        `"answer_end": [`<br />
&emsp;&emsp;&emsp;       `1320`<br />
&emsp;&emsp;      `]`<br />
&emsp;       `}`<br />
`}`

**Similar format for evaluation as in fit.**
## Training and examples
To see how the model was trained, see `research.ipynb` jupiter notebook.<br />
To see how to use the model, see `usage.ipynb` jupiter notebook.<br />
See also the sample data in [examples_data](example_data).

## Sources
- https://spacy.io/