import tensorflow as tf
from random import random, randint, choices, choice, shuffle
from os import path
import json

class MnistDataTracker:
    def __init__(self, valid_ratio = 0.05):
        train_data, valid_data, test_data = self._load_data(valid_ratio)
        self._build_lists(train_data, valid_data, test_data)
        self._index_tracker = {
            "train": {
                i: 0
                for i in range(10)
            },
            "valid": {
                i: 0
                for i in range(10)
            },
            "test": {
                i: 0
                for i in range(10)
            }
        }

    def _load_data(self, valid_ratio):
        (_, train_labels), (_, test_labels)  = tf.keras.datasets.mnist.load_data()
        train_valid_data = [(sample_idx, label) for sample_idx, label in enumerate(train_labels)]
        test_data = [(sample_idx, label) for sample_idx, label in enumerate(test_labels)]
        shuffle(train_valid_data)
        split_idx = int(len(train_valid_data) * valid_ratio)
        valid_data, train_data = train_valid_data[:split_idx], train_valid_data[split_idx:]
        return train_data, valid_data, test_data

    def _build_lists(self, train_data, valid_data, test_data):
        self.sample_lists = {
            "train": {
                i: list()
                for i in range(10)
            },
            "valid": {
                i: list()
                for i in range(10)
            },
            "test": {
                i: list()
                for i in range(10)
            }
        }
        for (sample_idx, label) in train_data:
            self.sample_lists["train"][label].append(sample_idx)
        for (sample_idx, label) in valid_data:
            self.sample_lists["valid"][label].append(sample_idx)
        for (sample_idx, label) in test_data:
            self.sample_lists["test"][label].append(sample_idx)

        for train_valid_test in ["train", "valid", "test"]:
            for i in range(10):
                shuffle(self.sample_lists[train_valid_test][i])

    def __call__(self, train_valid_test, digit):
        list_index = self._index_tracker[train_valid_test][digit]
        sample_index = self.sample_lists[train_valid_test][digit][list_index]
        self._index_tracker[train_valid_test][digit] += 1
        if self._index_tracker[train_valid_test][digit] == len(self.sample_lists[train_valid_test][digit]):
            self._index_tracker[train_valid_test][digit] = 0
            shuffle(self.sample_lists[train_valid_test][digit])
        return sample_index



class MemberDataGenerator:
    def __init__(self, n, member_ratio = 0.5, valid_ratio = 0.05):
        self.n = n
        self.member_ratio = member_ratio
        self.mnist_data_tracker = MnistDataTracker(valid_ratio)

    def generate_example(self, train_valid_test):
        if self.n < 5:
            digits_picked = [randint(0, 9) for _ in range(self.n)]
            other_digits = [i for i in range(10) if i not in digits_picked]
        else:
            digit_pool = choices(range(10), k=5)
            other_digits = [i for i in range(10) if i not in digit_pool]
            digits_picked = [choice(digit_pool) for _ in range(self.n)]

        generated_set = [self.mnist_data_tracker(train_valid_test, digit_picked) for digit_picked in digits_picked]
        is_member = True if random() < self.member_ratio else False
        if is_member:
            target_digit = choice(digits_picked)
        else:
            target_digit = choice(other_digits)
        return {
            "set": generated_set,
            "target_digit": target_digit,
            "is_member": is_member
        }

    def generate_examples(self, num_examples, train_valid_test):
        return [self.generate_example(train_valid_test) for _ in range(num_examples)]


    def __call__(self, num_train_examples, num_valid_example, num_test_examples):
        data = {
            "train": self.generate_examples(num_train_examples, "train"),
            "valid": self.generate_examples(num_train_examples, "valid"),
            "test": self.generate_examples(num_train_examples, "test")
        }
        return data

def example_to_dict_format(n, example_id, example):
    return {
        "id": example_id,
        "raw_inputs_data": {
            "element_input({})".format(i): example["set"][i]
            for i in range(n)
        },
        "symbolic_inputs_data": {
            "target_input": {
                "value": str(example["target_digit"]),
                "value_idx": example["target_digit"]
            }
        },
        "target_labels": {
            "member_output": ["true" if example["is_member"] else "false"]
        },
        "target_labels_indices": {
            "member_output": [1 if example["is_member"] else 0]
        }
    }


def generate_data(n, member_ratio, filename, num_train_examples, num_valid_examples, num_test_examples, valid_ratio):
    generator = MemberDataGenerator(n, member_ratio, valid_ratio)
    data = generator(num_train_examples, num_valid_examples, num_test_examples)
    data = {
        "train": [
            example_to_dict_format(n, example_id, example)
            for example_id, example in enumerate(data["train"])
        ],
        "valid": [
            example_to_dict_format(n, example_id, example)
            for example_id, example in enumerate(data["valid"])
        ],
        "test": [
            example_to_dict_format(n, example_id, example)
            for example_id, example in enumerate(data["test"])
        ]
    }
    with open(path.join("embedding_reasoning", "experiments", "member", "{}".format(n), "{}.json".format(filename)), 'w') as json_fp:
        json.dump(data, json_fp)