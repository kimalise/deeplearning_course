import torch
import numpy as np
from torchvision.models import resnet50
from transformers import BertTokenizer, BertModel
import os
from torch.utils.data import Dataset, DataLoader
import cv2
import torchvision.transforms as transforms

train_ratio = 0.8
root = "./data/MVSA_Single"
epochs = 10
learning_rate = 1e-3
batch_size = 4

transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", cache_dir="./cache")

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

class MVSADataset(Dataset):
    def __init__(self, examples, transforms, label_vocab):
        super(MVSADataset, self).__init__()
        self.examples = examples
        self.transforms = transforms
        self.label_vocab = label_vocab

    def __getitem__(self, index):
        example = self.examples[index]
        img = cv2.imread(os.path.join(root, "data", example['image']), cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = self.transforms(img)

        with open(os.path.join(root, "data", example['text']), "r", encoding="utf-8") as f:
            text = f.readlines()[0]

        label = self.label_vocab[example["label"]]

        return img_tensor, text, label

    def __len__(self):
        return len(self.examples)

def collate_fn(batch):
    batch_text = []
    batch_img = []
    batch_label = []
    for item in batch:
        batch_img.append(item[0])
        batch_text.append(item[1])
        batch_label.append(item[2])

    img_tensor = torch.stack(batch_img, dim=0)

    text_tensor = tokenizer.batch_encode_plus(
        batch_text_or_text_pairs=batch_text,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_length=True,
        return_tensors="pt",
    )
    label_tensor = torch.tensor(batch_label, dtype=torch.long)

    return img_tensor, text_tensor, label_tensor

def test_dataset():
    image_files, text_files = read_data()
    label_dict = read_labels()
    examples, label_vocab = generate_exmaples(image_files, text_files, label_dict)
    train_exmaples, val_exmaples = split_dataset(examples, train_ratio)

    dataset = MVSADataset(train_exmaples, transforms, label_vocab)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
    for step, batch in enumerate(dataloader):
        print(batch)
        break

class MVSAModel(torch.nn.Module):
    def __init__(self):
        super(MVSAModel, self).__init__()
        resnet = resnet50(pretrained=True)
        self.image_encoder = torch.nn.Sequential(*resnet.children())[:-1]
        # model = torch.nn.Sequential(A, B, C)
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased", cache_dir="./cache")
        self.fc = torch.nn.Linear(2048 + 768, 3)

    def forward(self, img_tensor, text_tensor):
        img_feature = self.image_encoder(img_tensor) # [B, 3, h, w] -> [B, 2048, 1, 1]
        img_feature = torch.squeeze(img_feature) # [B, 2048]

        text_feature = self.text_encoder(
            text_tensor['input_ids'],
            text_tensor['attention_mask'],
            text_tensor['token_type_ids']
        )
        text_feature = text_feature['pooler_output'] # cls [B, 768]
        feature = torch.cat([img_feature, text_feature], dim=1) # [B, 2048 + 768]
        output = self.fc(feature) # [B, 3]
        return output

def train(model, dataloader):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()

    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            img_tensor = batch[0]
            text_tensor = batch[1]
            label_tensor = batch[3]

            pred_logits = model(img_tensor, text_tensor)
            loss = criterion(pred_logits, label_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                print("epoch: {}, step: {}, loss: {:.4f}".format(epoch, step, loss.item()))


if __name__ == '__main__':
    # image_files, text_files = read_data()
    # label_dict = read_labels()
    # examples, label_vocab = generate_exmaples(image_files, text_files, label_dict)
    # train_exmaples, val_exmaples = split_dataset(examples, train_ratio)
    test_dataset()
    print("complete")




