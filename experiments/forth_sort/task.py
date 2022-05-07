from os import path

import tensorflow as tf

from framework.dataset import EmbeddingReasoningDatasetCreator
from framework.task import Task
from experiments.data.raw_datasets import DigitComparisonDataset
from experiments.networks.perception_networks import ComparisonNet
from experiments.networks.reasoning_networks import sort_reasoning

class ForthSortDatasetCreator(EmbeddingReasoningDatasetCreator):
    def __init__(self, task_metadata, config, full_sample_data):
        self.comparison_dataset = DigitComparisonDataset()
        super(ForthSortDatasetCreator, self).__init__(task_metadata, config, full_sample_data)

    def build_full_latent_dataset(self):
        self.latent_concept_datasets = {
            "comparison": self.comparison_dataset.build_latent_concept_dataset("comparison")
        }

    def retrieve_raw_input(self, latent_concept_name, latent_concept_sample_id, train_valid_test):
        return self.comparison_dataset.retrieve_raw_input(latent_concept_sample_id, train_valid_test)


class ForthSortTask(Task):
    BASE_PATH = path.join("experiments", "forth_sort")

    def __init__(self, n):
        self.__length = n
        super(ForthSortTask, self).__init__("forth_sort", ForthSortDatasetCreator,
                                            filepaths={
                                                "metadata": path.join(ForthSortTask.BASE_PATH, "{}".format(n),
                                                                      "metadata.json"),
                                                "data": path.join(ForthSortTask.BASE_PATH, "{}".format(n),
                                                                  "data.json"),
                                                "rules": path.join(ForthSortTask.BASE_PATH, "{}".format(n),
                                                                   "rules.json"),
                                                "config": path.join(ForthSortTask.BASE_PATH, "{}".format(n),
                                                                   "config.json"),
                                            })

    def get_perception_networks(self):
        return [("comparison_net", ComparisonNet(self.config["embedding_size"]), "comparison")]

    def get_reasoning_network(self):
        num_pairs = self.__length * (self.__length - 1) // 2
        input_list = [tf.keras.Input(shape=(self.config["embedding_size"],)) for _ in range(num_pairs)] + \
                     [tf.keras.Input(shape=(10,)) for _ in range(self.__length)]
        return sort_reasoning(input_list, self.__length)


    def get_dataset_creator(self):
        return ForthSortDatasetCreator

    def dataset_test(self):
        ds = self.dataset_creator.get_full_dataset("train", shuffle=False)
        ds.symbolic_labels_mode = False
        num_pairs = self.__length * (self.__length - 1) // 2
        for batch_inputs, batch_labels in ds:
            batch_pair_inputs, batch_list_inputs = batch_inputs[:num_pairs], batch_inputs[num_pairs:]
            for i in range(len(batch_pair_inputs[0])):
                pair_inputs = [p[i] for p in batch_pair_inputs]
                list_inputs = [p[i] for p in batch_list_inputs]
                list_outputs = [p[i] for p in batch_labels]
                print(pair_inputs, list_inputs, list_outputs)


    def get_labels_for_pca(self, latent_concept_name):
        return ["not_swap", "swap"]

    def pca_label_extractor(self, label):
        return label
