import argparse
import spacy
import enchant
import re
import os
import jsonlines
import torch
import numpy as np
from tqdm import tqdm
from simcse import SimCSE

uncertain_list = [
    "The answer is unknown.",
    "The answer is uncertain.",
    "The answer is unclear.",
    "There is no definitive answer.",
    "There is no known case.",
    "There is no concrete answer to this question.",
    "There is no public information available.",
    "It is impossible to know.",
    "It is impossible to answer.",
    "It is impossible to provide a definitive answer.",
    "It is impossible to determine."
    "It is difficult to predict.",
    "It is not known.",
    "We need to know the value.",
    "We do not know.",
    "We can't determine.",
    "We can't calculate.",
    "We are not given enough information.",
    "We need additional information.",
    "We cannot provide an answer.",
    "I'm not sure.",
    "I'm unable to determine."
    "This problem cannot be answered.",
    "Please provide that information.",
]

parser = argparse.ArgumentParser()
parser.add_argument("--filename", type=str, help="Input Filename", required=True)
args = parser.parse_args()

model = SimCSE("princeton-nlp/sup-simcse-roberta-large")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
threshold = 0.75
nlp = spacy.load("en_core_web_sm")
dictionary = enchant.Dict("en_US")
names = [
    "Bianca", "Nico", "Calum", "Ravi", "Pardee", "Zack", "Ivar", "Iain", "Jaden",
    "Kaleb", "Colton", "Celestia", "Jaylen", "Rebecca", "Jordana", "Elsa", "Aren",
    "Yasna", "Matias", "Cally", "Andrena", "Debelyn", "Juwella", "Carli", "Nori",
    "Mae", "Lea", "Caleb", "Kaiden", "Holly", "Dane", "Francie", "Markus", "Mara",
    "Amaro", "Luca", "Barbi", "Nunzio", "Keanu", "Rayden", "Rihanna", "Jerusha",
    "Abie", "Davonte", "Maiya", "Albaszu", "Bucky", "Zhang", "Jung", "Popton",
    "Ragnar", "Anika", "Reeta", "Grayson", "Kiki", "Jayden", "Aitana", "Jayda",
    "Julio", "Mateo", "Celine", "Sansa", "Remi", "Juanico", "Maddy", "Geric", "Theo",
    "Roxy", "Shyne", "Catriona", "Jessy", "Callen", "Hajar", "Farah", "Terez", "Esme",
    "Gervais", "Milena", "Kaylee", "Dani", "Dabbie", "Juvy", "Nataly", "Samanta",
    "Iesha", "Vidya", "Amalie", "dane", "Oli", "Alden", "Vihaan", "Trish", "Orlan", "Uki", "Bea", "Robie", "Daxton",
    "Riku", "Kylie", "Nelly", "Jonny", "Ivanka", "Woody", "Daniela", "Ricciana", "Margarita", "Sandro", "Emiliano",
    "Daphney", "Shara", "Cortney", "I", "Bucyus", "Talia", "Kyla", "Jessa", "Adriel", "Wynter", "Wolfgang", "Ludo",
    "Lowella", "Zahra", "Mancino", "Tilly", "Comcast", "Faber", "Meena", "Julio", "Michaela", "Aleena", "Paulo",
    "Kimiko", "Connor", "Johan", "Tobee", "Kimmie", "Etsy", "Jaydee", "Keesha", "Elodie", "Hasan", "Alina", "Oshea",
    "Jonsey", "Jine", "Danai", "Kira", "Jairus", "Arniel", "Jesy", "Hermione", "Payton", "Amare", "Emmalyn", "Dewei",
    "Dikembe", "Benjy", "Seeya", "Haruto"
]

locations = [
    "Wayupnorth", "Northton", "Southton", "Silverlake", "Ravenswood", "Diaz", "Seahawks", "Paddington", "africa",
    "asia", "europe",
    "Hoopits", "Zeoland",
    "Neglarts", "Bucyrus"
]

others = [
    "siamese", "parmesan", "st", "nd", "rd", "th", "Walmart", "Oreos", "popsicle",
    "Bobbit", "kgs", "MLB", "gameplay", "deadlift", "Slurpees", "Coronavirus", "MSRP", "coronavirus", "reais", "legos",
    "COVID", "housesits", "gummies", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "HVAC", "dodgeballs", "cheezits"
]

word2dict = names + locations + others

for word in word2dict:
    dictionary.add(word)


def readResult(data_dir) -> []:
    data = []
    with jsonlines.open(data_dir, "r") as f:
        for item in f:
            data.append(item)
    return data


def cut_sentence(text):
    sentences = re.split(r"(\.|\!|\?|。|！|？|\.{6}|\n)", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def extract_last_few_sentences(text, cnt=2):
    sentences = cut_sentence(text)
    ret = ""
    if sentences:
        for index in range(len(sentences)):
            if index > cnt:
                break
            else:
                ret += sentences[-(index + 1)]
        return ret
    else:
        return ""


def extract_number(text: str):
    pattern = r"\s*(-?\d+(\.\d+)?)"
    matches = re.findall(pattern, text)
    return matches


def list_files_in_directory(directory):
    file_list = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_list.append(file_path)

    return file_list


def cut_sub_string(input_string, window_size=6, punctuation=".,?!"):
    input_string = input_string.strip().lower()
    if len(input_string) < 2:
        return [""]
    if input_string[-1] in punctuation:
        input_string = input_string[:-1]
    string_list = input_string.split()
    length = len(string_list)
    if length <= window_size:
        return [input_string]
    else:
        res = []
        for i in range(length - window_size + 1):
            sub_string = " ".join(string_list[i: i + window_size])
            if sub_string != "" or sub_string != " ":
                res.append(sub_string)
        return res


def is_english_word(word):
    pattern = re.compile(r'^[a-zA-Z]+$')
    if bool(pattern.match(word)) == True:
        if (len(word) < 2 and word != "a" and word != 'A' and word != "i"):
            return False
        if dictionary.check(word) == True:
            return True
        else:
            return False
    else:
        return False


def check_contains_math_expression(s):
    doc = nlp(s)

    expressions = []
    item = ""
    for token in doc:
        if token.text == "," or token.text == "." or token.text == "'s" or token.is_quote == True:
            continue
        # print(token)
        if is_english_word(token.text) == False:
            item += token.text
        else:
            if item != "":
                expressions.append(item)
                item = ""

    if item != "":
        expressions.append(item)

    variable_pattern = r'[a-zA-Z]'

    max_length_variable_expression = None
    max_length = 0

    for expression in expressions:
        variables = re.findall(variable_pattern, expression)
        if variables:
            allowed_characters = re.compile(r'^[0-9a-zA-Z()+\-*/\[\]]*$')
            if bool(allowed_characters.match(expression)) == False:
                continue
            expression_length = len(expression)
            if expression_length > max_length:
                max_length = expression_length
                max_length_variable_expression = expression

    if (max_length_variable_expression != None):
        return True
    else:
        return False


def judge_generated_text_unanswerable(text):
    sentences = cut_sentence(text)
    sub_str_list = []
    for sub_sen in sentences:
        if len(sub_sen) >= 2:
            sub_str_list.extend(cut_sub_string(sub_sen))
    if len(sub_str_list) != 0:
        similarities = model.similarity(sub_str_list, uncertain_list, device=device)
    else:
        similarities = [0]
    max_uncertainty = np.max(similarities)
    if max_uncertainty > threshold:
        return True

    for index in range(len(sentences)):
        if index > 2:
            break
        if check_contains_math_expression(sentences[-(index + 1)]):
            return True
    return False


if __name__ == '__main__':

    data_dir = args.filename

    result = readResult(data_dir)

    answerable_cnt = 0
    unanswerable_cnt = 0

    TP = 0
    FP = 0
    FN = 0
    Acc = 0

    for item in tqdm(result):
        id: int = item['id']
        answerable: bool = item['answerable']
        generated_text: str = item['generated_text']
        answer = None
        extracted_number = None
        pred_unanswerable = judge_generated_text_unanswerable(generated_text)
        # print(id, pred_unanswerable)
        generated_text = generated_text.replace('\n', ' ')

        if answerable == True:
            answerable_cnt += 1
        else:
            unanswerable_cnt += 1

        if pred_unanswerable == False:
            last_sentence = extract_last_few_sentences(generated_text)
            extracted_number = [float(item[0]) for item in extract_number(last_sentence)]
            # print(id, last_sentence, extracted_number, answer, answerable_correct)
        else:
            pass

        if answerable == False:
            if pred_unanswerable == True:
                TP += 1
                # print('TP', id)
            else:
                FN += 1

        elif answerable == True and pred_unanswerable == True:
            FP += 1

        if answerable == True:
            if pred_unanswerable == False:
                answer = item['answer'][0]
                if answer in extracted_number:
                    Acc += 1
        # print(TP, FP, FN, Acc)

    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    F1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    print("Filename:", data_dir)
    print("Threshold", threshold)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", F1)
    print("Acc", Acc / answerable_cnt)
    print("TP: ", TP)
    print("FP: ", FP)
    print("FN: ", FN)
