# Importing libraries
import json
import spacy
import random
import warnings
import argparse
import parse
import shutil
import pathlib
import numpy as np
from spacy.util import minibatch, compounding
from spacy.training.example import Example
from tqdm import tqdm
from spacy.scorer import Scorer
from os import makedirs

# Setting default paths
DEFAULT_TRAIN_DATA_PATH = "./example_data/train.json"
DEFAULT_EVAL_DATA_PATH = "./example_data/eval_cropped.json"
SRC_MODEL_DIR = 'ner_model'

# Defining exceptions and labels
pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
labels = ["обеспечение исполнения контракта", "обеспечение гарантийных обязательств"]


# Creating a class for NER
class NER:
    # Initializing the class and loading the model
    def __init__(self, pipe_exceptions=pipe_exceptions):
        self.nlp = spacy.load('ner_model')
        self.ner = self.nlp.get_pipe("ner")
        self.unaffected_pipes = [pipe for pipe in self.nlp.pipe_names if pipe not in pipe_exceptions]

        # Adding the labels to the model
        for label in labels:
            self.ner.add_label(label)

    # A function to train model
    def fit(self, data_path=DEFAULT_TRAIN_DATA_PATH, batch_size=compounding(4.0, 32.0, 1.001), iterations=1,
            dropout_rate=0.3):
        prepared_data = parse.prepare_data(data_path)

        # Adding the labels to the model
        for _, annotations in prepared_data:
            for ent in annotations.get("entities"):
                self.ner.add_label(ent[2])

        # Disabling pipes that are not affected by NER training
        with self.nlp.disable_pipes(*self.unaffected_pipes):
            warnings.filterwarnings("ignore", category=UserWarning, module='spacy')
            for iteration in range(1, iterations + 1):
                random.shuffle(prepared_data)
                losses = {}
                batches = minibatch(prepared_data, size=batch_size)
                progress_bar = tqdm(total=len(prepared_data), bar_format='{l_bar}{bar:20}{r_bar}{bar:-10b}',
                                    desc=f"Iteration {iteration}", leave=False)
                for batch in batches:
                    for text, annotations in batch:
                        doc = self.nlp.make_doc(text)
                        example = Example.from_dict(doc, annotations)
                        self.nlp.update([example], losses=losses, drop=dropout_rate)
                        tqdm._instances.clear()
                        progress_bar.update(1)

                print('  Losses:', losses['ner'])
                print()

    # A function to predict the label of a given text or json file
    def predict(self, file_path, out_dir, verbose, is_json=True):
        try:
            makedirs(out_dir)
        except FileExistsError:
            pass

        # If the input is a json file
        if is_json:
            doc = self.nlp(file_path['text'])
            if len(doc.ents) != 0:
                for ent in doc.ents:
                    if ent.label_ == file_path['label']:
                        entity = ent
                    else:
                        print('Nothing')
                        return
                answer_start = file_path['text'].find(entity.text)
                answer_end = answer_start + len(entity.text)
                file_path['extracted_part'] = {'text': [entity.text],
                                               'answer_start': [answer_start],
                                               'answer_end': [answer_end]}

                with open(out_dir + 'predictions.json', 'w') as f:
                    json.dump(file_path, f, ensure_ascii=False, indent=2)

                if verbose:
                    print(f"\nResult:\n{file_path['extracted_part']}")
        # If the input is a text file
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                test_data = json.load(f)

            test_extracted_part = []

            progress_bar = tqdm(total=len(test_data), bar_format='{l_bar}{bar:20}{r_bar}{bar:-10b}',
                                desc="Processing", leave=False)

            progress_bar.update(1)

            for data in test_data:
                doc = self.nlp(data["text"])

                if len(doc.ents) != 0:
                    for ent in doc.ents:
                        if ent.label_ == data['label']:
                            entity = ent
                    answer_start = data['text'].find(entity.text)
                    answer_end = answer_start + len(entity.text)
                    test_extracted_part.append({'id': data['id'], 'text': data['text'], 'label': data['label'],
                                                'extracted_part': {'text': [entity.text],
                                                                   'answer_start': [answer_start],
                                                                   'answer_end': [answer_end]}})
                else:
                    test_extracted_part.append({'id': data['id'], 'text': data['text'], 'label': data['label'],
                                                'extracted_part': {'text': [""], 'answer_start': [0],
                                                                   'answer_end': [0]}})
                progress_bar.update(1)

            with open(out_dir + 'predictions.json', 'w', encoding="utf-8") as f:
                json.dump(test_extracted_part, f, ensure_ascii=False, indent=2, cls=NpEncoder)

            if verbose:
                verbose_count = 6 if len(test_extracted_part) > 5 else len(test_extracted_part)

                for i in range(verbose_count):
                    print(test_extracted_part[i]['extracted_part'])

                print(f".....{len(test_extracted_part) - verbose_count} more elements")

            formatted_path = out_dir[1:].replace("/", "\\")
            print(f"    \nPredictions at {pathlib.Path().resolve()}{formatted_path}predictions.json")

    # A function for evaluating the model by the main metrics
    def evaluate(self, data_path=DEFAULT_TRAIN_DATA_PATH):
        warnings.filterwarnings("ignore", category=UserWarning, module='spacy')

        prep_data = parse.prepare_data(data_path)
        scorer = Scorer()
        examples = []

        progress_bar = tqdm(total=len(prep_data), bar_format='{l_bar}{bar:20}{r_bar}{bar:-10b}',
                            desc="Processing", leave=False)

        for text, annotations in prep_data:
            predict = self.nlp(text)
            example = Example.from_dict(predict, annotations)
            example.predicted = self.nlp(str(example.predicted))
            examples.append(example)
            progress_bar.update(1)

        scores = scorer.score_spans(examples, "ents")
        print("    \nPrecision: {} \nRecall: {} \nF1-score: {}\n".format(scores['ents_p'],
                                                                     scores['ents_r'],
                                                                     scores['ents_f']))

    # A function to rollback changes made to the model
    @staticmethod
    def rollback_changes():
        dst_dir = 'copy_ner_model'

        # copy the directory and its contents to the destination directory
        shutil.copytree(SRC_MODEL_DIR, dst_dir)

        # delete the previous directory and its contents
        shutil.rmtree(SRC_MODEL_DIR)
        shutil.move(dst_dir, 'ner_model')


# Creating a class to encode numpy types into json
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


# Creating a function to run the model based on the user's input
def run(args, ner_model):
    # Processing incorrect input
    if (args.predict and args.evaluate) or (args.predict
                     and args.fit) or (args.fit and args.evaluate) or (args.rollback
                     and args.predict) or (args.rollback and args.fit):
        print("Incorrect input format. Use python model.py -h for help.")
        return
    # Predicting the label of a text or json file
    if args.predict:
        if args.input == "default":
            print("\nId:")
            id = int(input())
            print("\nText:")
            text = input()
            print("\nLabel:")
            label = input()
            dct = parse.to_dict(id, text, label)
            ner_model.predict(dct, args.out, args.verbose, is_json=True)
        else:
            ner_model.predict(args.input, args.out, args.verbose, is_json=False)

    # Fitting the model
    if args.fit:
        if args.input == 'default':
            path = DEFAULT_TRAIN_DATA_PATH
            if args.iter is None and args.drop is None:
                ner_model.fit(path)
            elif args.iter is None and args.drop is not None:
                ner_model.fit(path, dropout_rate=args.drop)
            elif args.iter is not None and args.drop is None:
                ner_model.fit(path, iterations=args.iter)
            else:
                ner_model.fit(path, iterations=args.iter, dropout_rate=args.drop)
        else:
            path = args.input
            if args.iter is None and args.drop is None:
                ner_model.fit(path)
            elif args.iter is None and args.drop is not None:
                ner_model.fit(path, dropout_rate=args.drop)
            elif args.iter is not None and args.drop is None:
                ner_model.fit(path, iterations=args.iter)
            else:
                ner_model.fit(path, iterations=args.iter, dropout_rate=args.drop)

    # Evaluating the model
    if args.evaluate:
        if args.input == 'default':
            path = DEFAULT_EVAL_DATA_PATH
            ner_model.evaluate(path)
        else:
            path = args.input
            ner_model.evaluate(path)

    # Rolling back changes made to the model
    if args.rollback:
        NER.rollback_changes()


# A function to get the arguments from the user's input
def get_args():
    parser = argparse.ArgumentParser(
        description="NER",
        epilog="Hi")

    parser.add_argument("-p", "--predict", help="""use it for prediction from a text input:
                                                (id, text, label: (обеспечение исполнения контракта | обеспечение гарантийных обязательств)
                                                or from a json file""",
                        action="store_true")
    parser.add_argument("-f", "--fit", help="use it for training model",
                        action="store_true")
    parser.add_argument("-it", "--iter", help="Number of iterations during training", type=int)
    parser.add_argument("-d", "--drop", help="Dropout rate", type=int)
    parser.add_argument("-e", "--evaluate", help="use it to evaluate model performance",
                        action="store_true")
    parser.add_argument("-i", "--input", help="path to file in json with input data for fit / predict / evaluate",
                        metavar="path", type=str, default="default")
    parser.add_argument("-o", "--out", help="path to output in predictions.json",
                        metavar="path", type=str, default="./output_data/")
    parser.add_argument("-r", "--rollback", help="return the model to its original state",
                        action="store_true")
    parser.add_argument("-v", "--verbose", help="verbose output",
                        action="store_true")

    return parser.parse_args()


def main():
    global args
    args = get_args()
    ner = NER()

    run(args, ner)


if __name__ == '__main__':
    main()
