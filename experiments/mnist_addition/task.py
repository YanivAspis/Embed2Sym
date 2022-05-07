from os import path

import tensorflow as tf

from framework.dataset import EmbeddingReasoningDatasetCreator
from framework.task import Task
from experiments.data.raw_datasets import MNISTRawDataset, CIFAR10RawDataset
from experiments.networks.perception_networks import LeNet, ResNet56Cifar10
from experiments.networks.reasoning_networks import mlp_reasoning, mlp_reasoning_cifar10


def cifar10_learning_rate_schedule(epoch, lr):
    if epoch == 100 or epoch == 150:
        return lr * 0.1
    else:
        return lr


class AdditionDatasetCreator(EmbeddingReasoningDatasetCreator):
    def __init__(self, task_metadata, config, full_sample_data):
        self.mnist_dataset = MNISTRawDataset(valid_imgs_from_training_set=True)
        super(AdditionDatasetCreator, self).__init__(task_metadata, config, full_sample_data)


    def build_full_latent_dataset(self):
        self.latent_concept_datasets = {
            "digit": self.mnist_dataset.build_latent_concept_dataset("digit")
        }

    def retrieve_raw_input(self, latent_concept_name, latent_concept_sample_id, train_valid_test):
        return self.mnist_dataset.retrieve_raw_input(latent_concept_sample_id, train_valid_test)

class CIFAR10AdditionDatasetCreator(EmbeddingReasoningDatasetCreator):
    def __init__(self, task_metadata, config, full_sample_data):
        self.cifar10_dataset = CIFAR10RawDataset(valid_imgs_from_training_set=True, class_names=["{}".format(i) for i in range(10)])
        super(CIFAR10AdditionDatasetCreator, self).__init__(task_metadata, config, full_sample_data)


    def build_full_latent_dataset(self):
        self.latent_concept_datasets = {
            "digit": self.cifar10_dataset.build_latent_concept_dataset("digit")
        }

    def retrieve_raw_input(self, latent_concept_name, latent_concept_sample_id, train_valid_test):
        return self.cifar10_dataset.retrieve_raw_input(latent_concept_sample_id, train_valid_test)


class MNISTAdditionTask(Task):
    def __init__(self, n = 1, cifar10 = False):
        self.n = n
        self.cifar10 = cifar10
        if cifar10:
            super(MNISTAdditionTask, self).__init__("mnist_addition", CIFAR10AdditionDatasetCreator,
                                                    filepaths={
                                                        "metadata": path.join("experiments", "mnist_addition", "{}".format(n), "metadata_cifar10.json"),
                                                        "data": path.join("experiments", "mnist_addition", "{}".format(n), "data_cifar10.json"),
                                                        "rules": path.join("experiments", "mnist_addition", "{}".format(n), "rules_cifar10.json"),
                                                        "config": path.join("experiments", "mnist_addition", "{}".format(n), "config_cifar10.json"),
                                                    })
            self.config["learning_rate_schedule"] = cifar10_learning_rate_schedule
        else:
            super(MNISTAdditionTask, self).__init__("mnist_addition", AdditionDatasetCreator, filepaths={
                "metadata": path.join("experiments", "mnist_addition", "{}".format(n), "metadata.json"),
                "data": path.join("experiments", "mnist_addition", "{}".format(n), "data.json"),
                "rules": path.join("experiments", "mnist_addition", "{}".format(n), "rules.json"),
                "config": path.join("experiments", "mnist_addition", "{}".format(n), "config.json"),
            })

    def get_perception_networks(self):
        if self.cifar10:
            return [("digit_recognizer", ResNet56Cifar10(output_size=self.config["embedding_size"], final_activation="relu"), "digit")]
        else:
            return [("digit_recognizer", LeNet(self.config["embedding_size"]), "digit")]

    def get_reasoning_network(self):
        inputs = [
            tf.keras.Input(shape=(self.config["embedding_size"]))
            for _ in range(2*self.n)
        ]
        inputs_merged = tf.keras.layers.Concatenate()(inputs)
        if self.cifar10:
            return mlp_reasoning_cifar10([inputs_merged], model_inputs=inputs)
        else:
            return mlp_reasoning([inputs_merged], model_inputs=inputs)

    def dataset_test(self):
        import matplotlib.pyplot as plt
        import numpy as np
        ds = self.dataset_creator.get_full_dataset("test", shuffle=True)
        ds.symbolic_labels_mode = True
        for (imgs1, imgs2), labels in ds:
            labels = np.stack(labels, axis=1)
            for img1, img2, label in zip(imgs1, imgs2, labels):
                fig, ((ax1, ax2)) = plt.subplots(1, 2)
                ax1.imshow((img1 + 1) / 2)
                ax1.axis("off")
                ax2.imshow((img2 + 1) / 2)
                ax2.axis("off")
                plt.figtext(0.5, 0.1, "{}".format(label), ha="center", fontsize=18, )
                plt.show()

    def get_labels_for_pca(self, latent_concept_name):
        return ["{}".format(i) for i in range(10)]

    def pca_label_extractor(self, label):
        return label
