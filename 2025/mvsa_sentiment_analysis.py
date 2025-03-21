import torch
import numpy as np
from torchvision.models import resnet50
from transformers import BertTokenizer, BertModel
import os

train_ratio = 0.8
root = "./data/MVSA_Single"

def read_data():
    image_files = []
    text_files = []
    for file in os.listdir(os.path.join(root, "data")):
        main_file_name = os.path.splitext(file)[0]
        ext_name = os.path.splitext(file)[1]
        if ext_name == ".jpg":
            image_files.append(file)
            text_files.append(main_file_name + ".txt")
    return image_files, text_files

def read_labels():
    results = {}
    with open(os.path.join(root, "labelResultAll.txt"), "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines[1:]:
            line = line.strip("\n")
            ID, labels = line.split("\t")
            label_text, label_image = labels.split(",")
            if label_text == label_image:
                results[ID] = label_text

    return results

# examples = [{"text": "1.text", "image": "1.jpg", "label": "positive"}, ...]
def generate_exmaples(image_files, text_files, label_dict):
    label_vocab = {}
    label_index = 0
    examples = []
    for key, value in label_dict.items():
        image_file = str(key) + ".jpg"
        text_file = str(key) + ".txt"
        label = value

        examples.append({
            "text": text_file,
            "image": image_file,
            "label": label
        })

        if not label in label_vocab:
            label_vocab[label] = label_index
            label_index += 1

    return examples, label_vocab

def split_dataset(examples, train_ratio):
    indices = np.arange(len(examples))
    np.random.shuffle(indices)
    examples = np.array(examples)[indices]

    num_train = int(len(examples) * train_ratio)

    return examples[:num_train], examples[num_train:]

if __name__ == '__main__':
    image_files, text_files = read_data()
    label_dict = read_labels()
    examples, label_vocab = generate_exmaples(image_files, text_files, label_dict)
    train_exmaples, val_exmaples = split_dataset(examples, train_ratio)
    print("complete")
