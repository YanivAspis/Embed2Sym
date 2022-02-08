import numpy as np
import tensorflow as tf
from framework.dataset import LatentConceptSample

class MNISTRawDataset:
    def __init__(self, valid_imgs_from_training_set = True, digits_to_load = range(10), digit_names = ["{}".format(i) for i in range(10)]):
        self.valid_imgs_from_training_set = valid_imgs_from_training_set
        self.digits_to_load = digits_to_load
        self.digit_names = digit_names
        (self.train_imgs, self.train_labels), (self.test_imgs, self.test_labels) = tf.keras.datasets.mnist.load_data()
        self.valid_imgs, self.valid_labels = (self.train_imgs, self.train_labels) if self.valid_imgs_from_training_set else (self.test_imgs, self.test_labels)

    def build_latent_concept_dataset(self, latent_concept_name):
        return {
                "train": {
                    i: LatentConceptSample(i, latent_concept_name, self.digit_names[self.digits_to_load.index(self.train_labels[i])], self.digits_to_load.index(self.train_labels[i]))
                    for i in range(len(self.train_imgs))
                    if self.train_labels[i] in self.digits_to_load
                },
                "valid": {
                    i: LatentConceptSample(i, latent_concept_name, self.digit_names[self.digits_to_load.index(self.valid_labels[i])], self.digits_to_load.index(self.valid_labels[i]))
                    for i in range(len(self.valid_imgs))
                    if self.valid_labels[i] in self.digits_to_load
                },
                "test": {
                    i: LatentConceptSample(i, latent_concept_name, self.digit_names[self.digits_to_load.index(self.test_labels[i])], self.digits_to_load.index(self.test_labels[i]))
                    for i in range(len(self.test_imgs))
                    if self.test_labels[i] in self.digits_to_load
                }
        }

    def retrieve_raw_input(self, latent_concept_sample_id, train_valid_test):
        img = {
            "train": self.train_imgs,
            "valid": self.valid_imgs,
            "test": self.test_imgs
        }[train_valid_test][latent_concept_sample_id]
        return (np.array(img / 255, dtype=np.float32) - 0.5) / 0.5



class CIFAR10RawDataset:
    DEFAULT_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    def __init__(self, valid_imgs_from_training_set = True, classes_to_load = DEFAULT_CLASSES, class_names = DEFAULT_CLASSES):
        self.valid_imgs_from_training_set = valid_imgs_from_training_set
        self.classes_to_load = classes_to_load
        self.class_names = class_names
        (self.train_imgs, self.train_labels), (self.test_imgs, self.test_labels) = tf.keras.datasets.cifar10.load_data()
        self.valid_imgs, self.valid_labels = (self.train_imgs, self.train_labels) if self.valid_imgs_from_training_set else (self.test_imgs, self.test_labels)
        self.samples = {
            "train": self.train_imgs,
            "valid": self.valid_imgs,
            "test": self.test_imgs
        }
        self.datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True)
        self.pixel_mean = np.expand_dims(np.array([0.485, 0.456, 0.406]), axis=(0,1))
        self.pixel_std = np.expand_dims(np.array([0.229, 0.224, 0.225]), axis=(0,1))


    def build_latent_concept_dataset(self, latent_concept_name):
        return {
                "train": {
                    i: LatentConceptSample(i, latent_concept_name,
                                           self.class_names[self.classes_to_load.index(CIFAR10RawDataset.DEFAULT_CLASSES[self.train_labels[i][0]])],
                                           self.classes_to_load.index(CIFAR10RawDataset.DEFAULT_CLASSES[self.train_labels[i][0]]))
                    for i in range(len(self.train_imgs))
                    if CIFAR10RawDataset.DEFAULT_CLASSES[self.train_labels[i][0]] in self.classes_to_load
                },
                "valid": {
                    i: LatentConceptSample(i, latent_concept_name,
                                           self.class_names[self.classes_to_load.index(CIFAR10RawDataset.DEFAULT_CLASSES[self.valid_labels[i][0]])],
                                           self.classes_to_load.index(CIFAR10RawDataset.DEFAULT_CLASSES[self.valid_labels[i][0]]))
                    for i in range(len(self.valid_imgs))
                    if CIFAR10RawDataset.DEFAULT_CLASSES[self.valid_labels[i][0]] in self.classes_to_load
                },
                "test": {
                    i: LatentConceptSample(i, latent_concept_name,
                                           self.class_names[self.classes_to_load.index(CIFAR10RawDataset.DEFAULT_CLASSES[self.test_labels[i][0]])],
                                           self.classes_to_load.index(CIFAR10RawDataset.DEFAULT_CLASSES[self.test_labels[i][0]]))
                    for i in range(len(self.test_imgs))
                    if CIFAR10RawDataset.DEFAULT_CLASSES[self.test_labels[i][0]] in self.classes_to_load
                }
        }

    def retrieve_raw_input(self, latent_concept_sample_id, train_valid_test):
        img = self.samples[train_valid_test][latent_concept_sample_id]
        img = (np.array(img).astype("float32") / 255 - self.pixel_mean) / self.pixel_std
        if train_valid_test == "train":
            return self.datagen.random_transform(img)
        else:
            return img