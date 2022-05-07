import random
import json

import numpy as np
import tensorflow as tf



class LatentConceptSample:
    def __init__(self, sample_id, latent_concept_name, latent_concept_value, value_index):
        self.sample_id = sample_id
        self.latent_concept_name = latent_concept_name
        self.latent_concept_value = latent_concept_value
        self.latent_concept_value_index = value_index

    def to_dict(self):
        return {
            "id": self.sample_id,
            "latent_concept_name": self.latent_concept_name,
            "latent_concept_value": self.latent_concept_value,
            "latent_concept_value_index": self.latent_concept_value_index
        }

    @staticmethod
    def from_dict(latent_concept_sample_dict):
        return LatentConceptSample(latent_concept_sample_dict["id"],
                                   latent_concept_sample_dict["latent_concept_name"],
                                   latent_concept_sample_dict["latent_concept_value"],
                                   latent_concept_sample_dict["latent_concept_value_index"])


class Sample:
    def __init__(self, sample_id, raw_inputs_data, symbolic_inputs_data, target_labels, target_labels_indices):
        self.sample_id = sample_id
        self.raw_inputs_data = raw_inputs_data
        self.symbolic_inputs_data = symbolic_inputs_data
        self.target_labels = target_labels
        self.target_labels_indices = target_labels_indices

    def to_dict(self):
        return {
            "id": self.sample_id,
            "raw_inputs_data": self.raw_inputs_data,
            "symbolic_inputs_data": self.symbolic_inputs_data,
            "target_labels": self.target_labels,
            "target_labels_indices": self.target_labels_indices
        }

    @staticmethod
    def from_dict(sample_dict):
        return Sample(sample_dict["id"], sample_dict["raw_inputs_data"], sample_dict["symbolic_inputs_data"],
                      sample_dict["target_labels"], sample_dict["target_labels_indices"])


class EmbeddingReasoningDatasetCreator:
    def __init__(self, task_metadata, config, full_sample_data):
        self._task_metadata = task_metadata
        self.config = config

        self.build_full_latent_dataset()
        self.build_full_task_datasets(full_sample_data)
        self.select_samples()


    def build_full_latent_dataset(self):
        return NotImplementedError()

    def build_full_task_datasets(self, full_sample_data):
        self.full_task_datasets = {
            train_valid_test: {
                sample.sample_id: sample for sample in full_sample_data[train_valid_test]
            }
            for train_valid_test in ["train", "valid", "test"]
        }

    def select_samples(self):
        train_ids = [sample_id for sample_id in self.full_task_datasets["train"].keys()]
        if "num_train_samples" in self.config:
            random.shuffle(train_ids)
            train_ids = train_ids[:self.config["num_train_samples"]]
        self.selected_sample_ids = {
            "train": train_ids,
            "valid": [sample_id for sample_id in self.full_task_datasets["valid"].keys()],
            "test": [sample_id for sample_id in self.full_task_datasets["test"].keys()]
        }

        self.selected_latent_sample_ids = {
            latent_concept_name: {
                "train": list(),
                "valid": list(),
                "test": list()
            }
            for latent_concept_name in self._task_metadata.latent_concepts.keys()
        }

        latent_concept_to_raw_inputs = {
            latent_concept_name: [raw_input.name for raw_input in self._task_metadata.raw_inputs.values() if raw_input.latent_concept_name == latent_concept_name]
            for latent_concept_name in self._task_metadata.latent_concepts.keys()
        }

        for train_valid_test in ["train", "valid", "test"]:
            for sample_id in self.selected_sample_ids[train_valid_test]:
                for latent_concept_name in self._task_metadata.latent_concepts.keys():
                    self.selected_latent_sample_ids[latent_concept_name][train_valid_test] += \
                        [self.full_task_datasets[train_valid_test][sample_id].raw_inputs_data[raw_input_name]
                         for raw_input_name in latent_concept_to_raw_inputs[latent_concept_name]]

        for train_valid_test in ["train", "valid", "test"]:
            for latent_concept_name in self._task_metadata.latent_concepts.keys():
                self.selected_latent_sample_ids[latent_concept_name][train_valid_test] = \
                    list(set(self.selected_latent_sample_ids[latent_concept_name][train_valid_test]))

    def retrieve_raw_input(self, latent_concept_name, latent_concept_sample_id, train_valid_test):
        raise NotImplementedError()

    def get_full_dataset(self, train_valid_test, shuffle):
        full_ds_obj = EmbeddingReasoningFullDataset(self._task_metadata,
                                                    self.full_task_datasets[train_valid_test],
                                                    self.selected_sample_ids[train_valid_test],
                                                    train_valid_test,
                                                    self.config, shuffle)
        full_ds_obj.retrieve_raw_input = self.retrieve_raw_input
        return full_ds_obj

    def get_latent_concept_dataset(self, latent_concept_name, train_valid_test, shuffle):
        raw_input_ds_obj = LatentConceptDataset(self._task_metadata.latent_concepts[latent_concept_name],
                                                self.latent_concept_datasets[latent_concept_name][train_valid_test],
                                                self.selected_latent_sample_ids[latent_concept_name][train_valid_test],
                                                train_valid_test,
                                                self.config, shuffle)
        raw_input_ds_obj.retrieve_raw_input = self.retrieve_raw_input
        return raw_input_ds_obj

    def to_dict(self):
        return {
            train_valid_test: [sample.to_dict() for sample in self.full_task_datasets[train_valid_test].values()]
            for train_valid_test in ["train", "valid", "test"]
        }

    @staticmethod
    def from_dict(dataset_creator_dict, dataset_creator_class, task_metadata, config):
        full_sample_data = {
            train_valid_test: [Sample.from_dict(sample) for sample in dataset_creator_dict[train_valid_test]]
            for train_valid_test in ["train", "valid", "test"]
        }
        return dataset_creator_class(task_metadata, config, full_sample_data)

    def to_json(self, json_filepath):
        with open(json_filepath, 'w') as json_fp:
            json.dump(self.to_dict(), json_fp)

    @staticmethod
    def from_json(json_filepath, dataset_creator_class, task_metadata, config):
        with open(json_filepath, 'r') as json_fp:
            return EmbeddingReasoningDatasetCreator.from_dict(json.load(json_fp), dataset_creator_class, task_metadata, config)






class EmbeddingReasoningFullDataset(tf.keras.utils.Sequence):
    def __init__(self, task_metadata, full_task_dataset, sample_ids, train_valid_test, config, shuffle):
        self._task_metadata = task_metadata
        self._full_task_dataset = full_task_dataset
        self._sample_ids = sample_ids
        self._train_valid_test = train_valid_test
        self.config = config
        self.batch_size = config["batch_size"] if train_valid_test == "train" else config["test_batch_size"]
        self.shuffle = shuffle
        self.symbolic_labels_mode = False
        self.on_epoch_end()

    def retrieve_target_concept_label(self, sample, target_name):
        if self.symbolic_labels_mode:
            return sample.target_labels[target_name]
        else:
            label_array = np.zeros(shape=(len(self._task_metadata.target_concepts[self._task_metadata.targets[target_name].target_concept_name].values),))
            label_array[sample.target_labels_indices[target_name]] = 1.0
        return label_array

    def get_samples(self, sample_ids):
        samples = [self._full_task_dataset[sample_id] for sample_id in sample_ids]
        inputs = [
                     np.array([self.retrieve_raw_input(raw_input.latent_concept_name,
                                                       sample.raw_inputs_data[raw_input.name],
                                                       self._train_valid_test) for
                               sample in samples])
                     for raw_input in self._task_metadata.raw_inputs.values()
                 ] + [
                     tf.keras.utils.to_categorical(
                         [sample.symbolic_inputs_data[symbolic_input_name]["value_idx"] for sample in samples],
                         num_classes=self._task_metadata.symbolic_inputs[symbolic_input_name].num_values)
                     for symbolic_input_name in self._task_metadata.symbolic_inputs.keys()
                 ]
        labels = [np.array([self.retrieve_target_concept_label(sample, target_name) for sample in samples])
                  for target_name in self._task_metadata.targets.keys()]
        return tuple(inputs), tuple(labels)

    def sample_items(self, k):
        if k > len(self._sample_ids):
            k = len(self._sample_ids)
        ids = random.sample(self._sample_ids, k)
        return self.get_samples(ids)

    def __getitem__(self, batch_index):
        if (batch_index + 1) * self.batch_size > len(self._sample_ids):
            ids = self._sample_ids[batch_index * self.batch_size:]
        else:
            ids = self._sample_ids[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]
        return self.get_samples(ids)

    def __len__(self):
        if len(self._sample_ids) % self.batch_size == 0:
            return int(np.floor(len(self._sample_ids) / self.batch_size))
        else:
            return int(np.floor(len(self._sample_ids) / self.batch_size)) + 1

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self._sample_ids)



class LatentConceptDataset(tf.keras.utils.Sequence):
    def __init__(self, latent_concept_metadata, latent_concept_dataset, latent_sample_ids, train_valid_test, config, shuffle):
        self._latent_concept_metadata = latent_concept_metadata
        self._latent_concept_dataset = latent_concept_dataset
        self._latent_sample_ids = latent_sample_ids
        self._train_valid_test = train_valid_test
        self.config = config
        self.batch_size = config["batch_size"] if train_valid_test == "train" else config["test_batch_size"]
        self.shuffle = shuffle
        self.symbolic_labels_mode = False
        self.on_epoch_end()

    def retrieve_latent_concept_label(self, sample):
        if self.symbolic_labels_mode:
            return sample.latent_concept_value
        else:
            label_index = sample.latent_concept_value_index
            return tf.keras.utils.to_categorical(label_index, num_classes=len(self._latent_concept_metadata.values))

    def get_samples(self, sample_ids):
        samples = [self._latent_concept_dataset[sample_id] for sample_id in sample_ids]
        raw_inputs = np.array(
            [self.retrieve_raw_input(self._latent_concept_metadata.name, sample.sample_id, self._train_valid_test) for sample in samples])
        labels = np.array([self.retrieve_latent_concept_label(sample) for sample in samples])
        return raw_inputs, labels


    def sample_items(self, k):
        ids = random.sample(self._latent_sample_ids, k)
        return self.get_samples(ids)


    def __getitem__(self, batch_index):
        if (batch_index + 1) * self.batch_size > len(self._latent_sample_ids):
            ids = self._latent_sample_ids[batch_index * self.batch_size:]
        else:
            ids = self._latent_sample_ids[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]
        return self.get_samples(ids)

    def __len__(self):
        if len(self._latent_sample_ids) % self.batch_size == 0:
            return int(np.floor(len(self._latent_sample_ids) / self.batch_size))
        else:
            return int(np.floor(len(self._latent_sample_ids) / self.batch_size)) + 1

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self._latent_sample_ids)