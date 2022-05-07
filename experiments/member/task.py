from os import path

import numpy as np
import tensorflow as tf

from framework.dataset import EmbeddingReasoningDatasetCreator
from framework.task import Task

from experiments.data.raw_datasets import MNISTRawDataset
from experiments.networks.perception_networks import LeNet
from experiments.networks.reasoning_networks import member_attention_reasoning


class MemberDatasetCreator(EmbeddingReasoningDatasetCreator):
    def __init__(self, task_metadata, config, full_sample_data):
        self.mnist_dataset = MNISTRawDataset(valid_imgs_from_training_set=True)
        super(MemberDatasetCreator, self).__init__(task_metadata, config, full_sample_data)

    def build_full_latent_dataset(self):
        self.latent_concept_datasets = {
            "element": self.mnist_dataset.build_latent_concept_dataset("element"),
        }

    def retrieve_raw_input(self, latent_concept_name, latent_concept_sample_id, train_valid_test):
        return self.mnist_dataset.retrieve_raw_input(latent_concept_sample_id, train_valid_test)


class MemberTask(Task):
    BASE_PATH = path.join("experiments", "member")

    def __init__(self, n=3):
        self.n = n
        super(MemberTask, self).__init__("member", MemberDatasetCreator, filepaths={
            "metadata": path.join(MemberTask.BASE_PATH, "{}".format(n), "metadata.json"),
            "data": path.join(MemberTask.BASE_PATH, "{}".format(n), "data.json"),
            "rules": path.join(MemberTask.BASE_PATH, "{}".format(n), "rules.json"),
            "config": path.join(MemberTask.BASE_PATH, "{}".format(n), "config.json")
        })

    def get_perception_networks(self):
        return [("element_recognizer", LeNet(self.config["embedding_size"]), "element")]

    def get_reasoning_network(self):
        inputs = [
            tf.keras.Input(shape=(self.config["embedding_size"],))
            for _ in range(self.n)
        ] + [tf.keras.Input(shape=(10,))]
        return member_attention_reasoning(inputs)


    def dataset_test(self):
        import matplotlib.pyplot as plt
        ds = self.dataset_creator.get_full_dataset("train", shuffle=False)
        ds.symbolic_labels_mode = True
        for inputs, (labels,) in ds:
            for i in range(len(labels)):
                fig, axs = plt.subplots(1, self.n)
                sample_imgs = [imgs[i] for imgs in inputs[:-1]]
                for ax, img in zip(axs, sample_imgs):
                    ax.imshow(img)
                    ax.axis("off")
                target_input = np.argmax(inputs[-1][i])
                plt.figtext(0.5, 0.1, "{}, {}".format(target_input, labels[i]), ha="center", fontsize=18, )
                plt.show()

    def get_labels_for_pca(self, latent_concept_name):
        return ["{}".format(i) for i in range(10)]

    def pca_label_extractor(self, label):
        return label
