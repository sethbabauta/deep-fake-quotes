import pandas as pd

filler = "UNK"

def get_data():
    return pd.read_csv("data.csv")

def extract_vocab(data):

    strip_chars = []
    remove_chars = [':', ';', ',', '“', '”', '"', '.', '…', '’', '‘']

    # extracting the vocab
    samples = [x[0].lower() for x in data.values]

    for i in range(len(samples)):
        for char in remove_chars:
            samples[i] = samples[i].replace(char, "")

    text = "\n".join(samples)
    vocab = []
    for word in text.split():
        if word not in vocab:
            vocab.append(word)

    vocab.append(filler)

    return vocab

data = get_data()

vocab = extract_vocab(data)

with open("vocab.txt", "w") as f:
    for w in vocab:
        f.write(w + "\n")