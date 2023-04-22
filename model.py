import spacy
import random
import warnings
import argparse
import parse
from spacy.util import minibatch, compounding
from pathlib import Path
from spacy.training.example import Example
from tqdm import tqdm
from spacy.scorer import Scorer

TRAIN_DATA_PATH = "./data/train.json"
pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]


class NER:
    def __init__(self, pipe_exceptions=pipe_exceptions):
        self.nlp = spacy.load('ner_model')
        self.ner = self.nlp.get_pipe("ner")

        self.prepared_data = parse.prepare_train(TRAIN_DATA_PATH)

        for _, annotations in self.prepared_data:
            for ent in annotations.get("entities"):
                self.ner.add_label(ent[2])

        self.unaffected_pipes = [pipe for pipe in self.nlp.pipe_names if pipe not in pipe_exceptions]

    def fit(self, batch_size=compounding(4.0, 32.0, 1.001), iterations=10, dropout_rate=0.3):
        with self.nlp.disable_pipes(*self.unaffected_pipes):
            warnings.filterwarnings("ignore", category=UserWarning, module='spacy')
            for iteration in range(iterations):
                random.shuffle(self.prepared_data)
                losses = {}
                batches = minibatch(self.prepared_data, size=batch_size)
                progress_bar = tqdm(total=len(self.prepared_data), desc=f"Iteration {iteration}", leave=False)
                for batch in batches:
                    for text, annotations in batch:
                        doc = self.nlp.make_doc(text)
                        example = Example.from_dict(doc, annotations)
                        self.nlp.update([example], losses=losses, drop=dropout_rate)
                        tqdm._instances.clear()
                        progress_bar.update(1)

                print('  Losses:', losses['ner'])
                print()

    def predict(self, file):
        print(file)

    def eval_metrics(self, test_data):
        print()

    def eval_loss(self, test_data):
        print()


def run(args, ner):
    if args.predict:
        if (args.input == "./example/input.json"):
            print("Id:")
            id = int(input())
            print("Text:")
            text = input()
            print("Label:")
            label = input()
            dct = parse.text_to_dict(id, text, label)
            ner.predict(dct)
        else:
            ner.predict(args.input)


def get_args():
    parser = argparse.ArgumentParser(
        description="NER",
        epilog="Hi")

    parser.add_argument("-p", "--predict", help="use it for prediction from a text input:\n"
                                                "(id, text, label: (обеспечение исполнения контракта | обеспечение гарантийных обязательств)\n"
                                                "or from a json file",
                        action="store_true")
    parser.add_argument("-i", "--input", help="path to file in json with input data for predict",
                        metavar="path", type=str, default="./example/input.json")
    parser.add_argument("-sf", "--samples", help="path to file in json with train samples for fit",
                        metavar="path", type=str, default="./example/train_data.json")
    parser.add_argument("-o", "--out", help="path for output in predictions.json",
                        metavar="path", type=str, default="./output/")

    return parser.parse_args()


def main():
    global args
    args = get_args()
    ner = NER()

    run(args, ner)


if __name__ == '__main__':
    main()
