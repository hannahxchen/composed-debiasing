import json
import re
import torch
from tqdm import tqdm
import flair
from flair.data import Sentence
from flair.models import SequenceTagger
from datasets import load_dataset

# load tagger
tagger = SequenceTagger.load('pos-fast')
flair.device = torch.device('cuda:0')

d = {
    "He": "She", "he": "she",
    "She": "He", "she": "he",
    "Himself": "Herself", "himself": "herself",
    "Herself": "Himself", "herself": "himself",
    "His": "Her", "his": "her",
    "Him": "Her", "him": "her",
    "Hers": "His", "hers": "his",
    "Mr": "Ms", "mr": "ms",
    "Ms": "Mr", "ms": "mr",
    "Mrs": "Mr", "mrs": "mr"
}

pattern = re.compile(r'\b(' + '|'.join(d.keys()) + r')\b')

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def main(dataset, datasplit):
    
    processed = []
    unprocessed = []
    for data in tqdm(dataset[datasplit]):
        temp = data
        title_scrubbed_gender_swapped = pattern.sub(lambda x: d[x.group()], data["title_scrubbed"])
        temp["title_scrubbed_gender_swapped"] = title_scrubbed_gender_swapped
        if re.search(fr'\bher\b', data["title_scrubbed"], flags=re.I):
            unprocessed.append(temp)
        else:
            processed.append(temp)

    all_sentences = [Sentence(x["title_scrubbed_gender_swapped"]) for x in unprocessed]
    all_sentences = list(chunks(all_sentences, 412))
    sentences = []
    for ss in tqdm(all_sentences):
        tagger.predict(ss)
        sentences.extend(ss)

    for i, data in enumerate(tqdm(unprocessed)):
        temp = data
        title_scrubbed_gender_swapped = data["title_scrubbed_gender_swapped"]
        for token in sentences[i]:
            if token.text.lower() != "her":
                continue
            new_token = None
            tag = token.get_tag("pos").value
            if tag == "PRP":
                new_token = "him"
            elif tag == "PRP$":
                new_token = "his"
            
            if new_token:
                if token.text[0].isupper():
                    new_token = new_token[0].upper() + new_token[1:]
                token.text = new_token

        title_scrubbed_gender_swapped = sentences[i].to_plain_string()
        temp["title_scrubbed_gender_swapped"] = title_scrubbed_gender_swapped
        processed.append(temp)
        
    with open(f"datasets/Biasbios/v2/{datasplit}.json", "w") as f:
        for x in processed:
            json.dump(x, f)
            f.write("\n")

if __name__ == "__main__":
    data_files = {
        "train": "datasets/Biasbios/train.json",
        "valid": "datasets/Biasbios/valid.json",
        "test": "datasets/Biasbios/test.json"
    }
    dataset = load_dataset("json", data_files=data_files)

    for datasplit in data_files.keys():
        print(f"Running {datasplit} set")
        main(dataset, datasplit)