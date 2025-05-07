from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, RobertaTokenizer
from multiprocessing import Pool
import multiprocessing
import json
import datasets

def get_dataset(file_path):
    dataset = CodeReviewDataset(file_path)
    dataloader = None
    return dataset, dataloader

class CodeReviewDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        print("Reading dataset from {}".format(file_path))
        self.data = [json.loads(line) for line in open(file_path)]
        print(f"data size: {len(self.data)}")
        self.data = [{  'old_code': d['old'],
                        'comment': d['comment'],
                        'new_code': d['new'],
                        'lang': d['lang'],
                        'patch': d['old_hunk']} for d in self.data]

    def get(self):
        return self.data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def save_dataset_to_disk(path, output):
    dataset = CodeReviewDataset(path)
    dataset = datasets.Dataset.from_list(dataset)
    dataset.save_to_disk(output)

def load_dataset(file_path):
    print("Reading dataset from {}".format(file_path))
    dataset = datasets.load_dataset('json', data_files=file_path)['train']\
                # .rename_column("patch", "code_diff")\
                # .rename_column("msg", "comment")
    print(dataset)
    print(f"data size: {len(dataset)}")
    return dataset


if __name__ == "__main__":
    save_dataset_to_disk('../../data/code_refinement/ref-train.jsonl', '../hf-datasets/train')
    save_dataset_to_disk('../../data/code_refinement/ref-test.jsonl', '../hf-datasets/test')
