import time
import datetime
import os
import requests
from bs4 import BeautifulSoup
from datasets import load_dataset, Dataset


API_URL = "https://translaturia.fhgr.ch/"


def load_data(translation_dataset, config_name):
    ds = load_dataset(translation_dataset, config_name)
    return ds["train"]


def translate_sentence(sentence, wait_time):
    resp = requests.post(API_URL, data={"input": sentence})
    response_timestamp = datetime.datetime.now()
    readable_time = response_timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")
    resp.raise_for_status()
    time.sleep(wait_time)
    soup = BeautifulSoup(resp.text, "html.parser")
    out = soup.find("textarea", {"id": "output"})
    if not out:
        raise RuntimeError("Could not find translated output in response.")
    return out.get_text(strip=True), readable_time


def store_dataset(dataset_list, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    new_dataset = Dataset.from_list(dataset_list)
    new_dataset.to_json(filepath, lines=True)


def update_elem(old_elem, german_sentence, translation, timestamp):
    new_elem = old_elem.copy()
    new_elem.pop("original_target", None)
    new_elem["source"] = german_sentence
    new_elem["target"] = translation
    new_elem["lp"] = "de_DE-rm-rumgr"
    new_elem["translation_time"] = timestamp
    return new_elem


def main():
    dataset = load_data("google/wmt24pp", "en-de_DE")
    new_dataset_list = []
    wait_time = 2 # seconds
    counter = 0
    total = len(dataset)
    print("Starting translation...")
    for elem in dataset:
        german_sentence = elem["target"]
        translation, timestamp = translate_sentence(german_sentence, wait_time)
        new_elem = update_elem(elem, german_sentence, translation, timestamp)
        new_dataset_list.append(new_elem)
        counter += 1
        print(f"Translated {counter}/{total} sentences", end="\r", flush=True)

    output_path = "./translations/de_DE-rm-rumgr.jsonl"
    store_dataset(new_dataset_list, output_path)
    print("Dataset saved successfully")


if __name__ == "__main__":
    main()
