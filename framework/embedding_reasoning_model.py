import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
from framework.logic_programs import ReasoningOptimisationTask
from framework.losses import get_loss_function
from framework.metrics import get_metric_function
from utils.clingo_wrapper import Clingo
from sklearn.metrics.cluster import rand_score, adjusted_rand_score, mutual_info_score, normalized_mutual_info_score, adjusted_mutual_info_score, homogeneity_completeness_v_measure, pair_confusion_matrix
from sklearn.metrics import confusion_matrix
from os import path, makedirs
import pickle
import json
import time

CLUSTERING_METRICS = {
    "Rand Score": rand_score,
    "Adjusted Rand Score": adjusted_rand_score,
    "Mutual Info Score": mutual_info_score,
    "Normalized Mutual Info Score": normalized_mutual_info_score,
    "Adjusted Mutual Info Score": adjusted_mutual_info_score,
    "Homogeneity_Completeness_V_Measure": homogeneity_completeness_v_measure,
    "Pair Confusion Matrix": pair_confusion_matrix
}



class EmbeddingReasoningModel:
    def __init__(self, task_metadata, config):
        self._task_metadata = task_metadata
        self._config = config
        self._perception_networks = dict()
        self._latent_concept_to_perception_network = dict()
        self._reasoning_networks = dict()
        self._target_concept_to_reasoning_network = dict()
        self._rules = []
        self._training_only_rules = []

    def add_perception_network(self, network_name, network, latent_concept_name):
        self._perception_networks[network_name] = network
        self._latent_concept_to_perception_network[latent_concept_name] = network_name

    def set_reasoning_network(self, network):
        self._reasoning_network = network

    def add_rule(self, rule):
        self._rules.append(rule)

    def add_training_only_rule(self, rule):
        self._training_only_rules.append(rule)

    def build_neural_models(self):
        self.raw_model_inputs = {
            raw_input.name: raw_input.to_keras_input() for raw_input in self._task_metadata.raw_inputs.values()
        }
        self.symbolic_model_inputs = {
            symbolic_input.name: symbolic_input.to_keras_input() for symbolic_input in self._task_metadata.symbolic_inputs.values()
        }
        self.model_inputs = list(self.raw_model_inputs.values()) + list(self.symbolic_model_inputs.values())
        self.model_embeddings = {
            raw_input.name: self._perception_networks[self._latent_concept_to_perception_network[raw_input.latent_concept_name]](input)
            for input, raw_input in zip(self.raw_model_inputs.values(), self._task_metadata.raw_inputs.values())
        }
        self.reasoning_input = list(self.model_embeddings.values()) + list(self.symbolic_model_inputs.values())
        self.reasoning_embedding = self._reasoning_network(self.reasoning_input)
        self.model_outputs = [
            tf.keras.layers.Dense(units=len(self._task_metadata.target_concepts[target.target_concept_name].values),
                                  activation=tf.keras.activations.sigmoid,
                                  name=target.name.replace('(', "_").replace(')', ""))(self.reasoning_embedding)
            for target in self._task_metadata.targets.values()
        ]
        self.neural_only_model = tf.keras.Model(inputs=self.model_inputs, outputs=self.model_outputs)
        self.latent_embeddings_models = {
            latent_concept_name: tf.keras.Model(
                inputs=self._perception_networks[self._latent_concept_to_perception_network[latent_concept_name]].inputs,
                outputs=self._perception_networks[self._latent_concept_to_perception_network[latent_concept_name]].outputs)
            for latent_concept_name in self._task_metadata.latent_concepts.keys()
        }

    def _set_up_clusters(self):
        self._clusters = {
            latent_concept_name: {
                "inputs": [raw_input.name for raw_input in self._task_metadata.raw_inputs.values() if raw_input.latent_concept_name == latent_concept_name],
                "cluster": KMeans(n_clusters=len(self._task_metadata.latent_concepts[latent_concept_name].values))
            }
            for latent_concept_name in self._task_metadata.latent_concepts.keys()
        }

    def _train_clusters(self, train_samples):
        for cluster_info in self._clusters.values():
            embeddings = []
            for sample in train_samples:
                embeddings += [sample["embeddings"][raw_input_name] for raw_input_name in cluster_info["inputs"]]
            cluster_info["cluster"].fit(embeddings)

    def _sample_for_cluster_training(self, train_dataset_creator, num_train_samples):
        train_ds = train_dataset_creator.get_full_dataset("train", shuffle=False)
        train_ds.symbolic_labels_mode = True
        input_vals, output_labels = train_ds.sample_items(k=num_train_samples)
        raw_input_vals = input_vals[:len(self.raw_model_inputs)]
        sym_input_vals = input_vals[len(self.raw_model_inputs):]
        embeddings = {
            raw_input_name: self.latent_embeddings_models[self._task_metadata.raw_inputs[raw_input_name].latent_concept_name].predict(x=raw_inputs)
            for raw_input_name, raw_inputs in zip(self._task_metadata.raw_inputs.keys(), raw_input_vals)
        }
        symbolic_inputs = {
            symbolic_input_name: np.take(self._task_metadata.symbolic_concepts[self._task_metadata.symbolic_inputs_to_symbolic_concepts[symbolic_input_name]].values, np.argmax(symbolic_input_value, axis=-1))
            for symbolic_input_name, symbolic_input_value in zip(self._task_metadata.symbolic_inputs.keys(), sym_input_vals)
        }
        labels = [
            {
                target_name: output_labels[i][j]
                for i, target_name in enumerate(self._task_metadata.targets.keys())
            }
            for j in range(len(output_labels[0]))
        ]

        return [
            {
                "embeddings": {
                    raw_input_name: embeddings[raw_input_name][idx]
                    for raw_input_name in embeddings.keys()
                },
                "symbolic_inputs": symbolic_inputs,
                "cluster_assignments": {
                    raw_input.name: {
                        "raw_input_name": raw_input.name,
                        "latent_concept_name": raw_input.latent_concept_name,
                    }
                    for raw_input in self._task_metadata.raw_inputs.values()
                },
                "target_labels": label,
            }
            for idx, label in enumerate(labels)
        ]

    def _assign_clusters(self, samples):
        for cluster_info in self._clusters.values():
            for raw_input_name in cluster_info["inputs"]:
                embeddings = [sample["embeddings"][raw_input_name] for sample in samples]
                predictions = cluster_info["cluster"].predict(embeddings)
                for sample, prediction in zip(samples, predictions):
                    sample["cluster_assignments"][raw_input_name]["cluster_index"] = prediction

    def _setup_optimizer(self):
        optimizer_class = {
            "adam": tf.keras.optimizers.Adam,
            "sgd": tf.keras.optimizers.SGD
        }[self._config["optimizer"]]
        optimizer_args = {
            "learning_rate": self._config["learning_rate"]
        }
        if "momentum" in self._config:
            optimizer_args.update({
                "momentum": self._config["momentum"]
            })
        return optimizer_class(**optimizer_args)

    def train_neural_only(self, dataset_creator, checkpoint_folder):
        loss_funcs = {
            output_layer.name.split('/')[0]: get_loss_function(self._config["loss"][target.target_concept_name])()
            for output_layer, target in zip(self.model_outputs, self._task_metadata.targets.values())
        }
        metric_funcs = {
            output_layer.name.split('/')[0]: get_metric_function(self._config["metrics"][target.target_concept_name])()
            for output_layer, target in zip(self.model_outputs, self._task_metadata.targets.values())
        }
        self.neural_only_model.compile(optimizer=self._setup_optimizer(),
                                       loss=loss_funcs,
                                       metrics=metric_funcs)
        #self.neural_only_model.summary()
        fit_args = {
            "x": dataset_creator.get_full_dataset("train", shuffle=True),
            "batch_size": self._config["batch_size"],
            "epochs": self._config["train_epochs"],
            "verbose": 1,
            "validation_data": dataset_creator.get_full_dataset("valid", shuffle=False),
            "validation_batch_size": self._config["test_batch_size"],
        }

        fit_args.update({
            "callbacks": []
        })

        if "learning_rate_schedule" in self._config:
            fit_args["callbacks"].append(tf.keras.callbacks.LearningRateScheduler(self._config["learning_rate_schedule"]))

        if checkpoint_folder is not None:
            makedirs(checkpoint_folder, exist_ok=True)
            target_name, metric_name = next(iter(self._config["metrics"].items()))
            if len(metric_funcs) == 1:
                metric_to_monitor = "val_{}".format(metric_name)
            else:
                metric_to_monitor = "val_{}_{}".format(self.model_outputs[0].name.split('/')[0], metric_name)
            model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=path.join(checkpoint_folder, "weights.h5"),
                                                                  save_weights_only=True,
                                                                  save_best_only=True,
                                                                  monitor=metric_to_monitor,
                                                                  mode="max")
            fit_args["callbacks"].append(model_checkpoint)



        self.neural_only_model.fit(**fit_args)

        # load weights of best model according to validation data
        if checkpoint_folder is not None:
            self.neural_only_model.load_weights(path.join(checkpoint_folder, "weights.h5"))



    def train(self, dataset_creator, checkpoint_folder = None):
        self.build_neural_models()
        self._set_up_clusters()

        train_begin_time = time.time()
        self.train_neural_only(dataset_creator, checkpoint_folder)
        print("Neural training time = {}".format(time.time() - train_begin_time))

        # Print Neural Only model's training and test accuracy
        self.neural_only_model.evaluate(x=dataset_creator.get_full_dataset("train", shuffle=False), batch_size=self._config["test_batch_size"])
        self.neural_only_model.evaluate(x=dataset_creator.get_full_dataset("test", shuffle=False), batch_size=self._config["test_batch_size"])

        cluster_train_samples = self._sample_for_cluster_training(dataset_creator, self._config["num_samples_for_cluster_training"])
        cluster_train_begin_time = time.time()
        self._train_clusters(cluster_train_samples)
        print("Cluster training time = {}".format(time.time() - cluster_train_begin_time))
        self._assign_clusters(cluster_train_samples)
        task = ReasoningOptimisationTask(self._task_metadata, self._rules + self._training_only_rules, cluster_train_samples, self._task_metadata.concepts_to_learn)

        # Uncomment this to debug Clingo Optimisation task
        #with open("test_k_means.p", 'w') as test_fp:
        #    test_fp.write("\n".join(task._optimisation_program))

        logic_optimisation_begin_time = time.time()
        self._cluster_mapping, self.learned_rules = task.solve()
        train_ending_time = time.time()
        print("Logic optimisation time = {}".format(train_ending_time - logic_optimisation_begin_time))
        print("Total training time = {}".format(train_ending_time - train_begin_time))
        #print(self._cluster_mapping)

    def predict_neural_only(self, data):
        return self.neural_only_model.predict(x=data, batch_size=self._config["batch_size"])

    def predict_latent_concept_model(self, latent_embeddings_model, latent_concept_name, data, return_without_mapping = False):
        embeddings = latent_embeddings_model.predict(x=data, batch_size=self._config["test_batch_size"])
        cluster_assignments = self._clusters[latent_concept_name]["cluster"].predict(embeddings.astype(np.double))
        if return_without_mapping:
            return cluster_assignments
        return np.take(self._cluster_mapping[latent_concept_name], cluster_assignments)

    def predict_neural_symbolic(self, data, symbolic_mode = False):
        concept_assignments = {
            raw_input.name: self.predict_latent_concept_model(self.latent_embeddings_models[raw_input.latent_concept_name], raw_input.latent_concept_name, data[raw_input.name])
            for raw_input in self._task_metadata.raw_inputs.values()
        }
        symbolic_inputs = {
            symbolic_input.name: data[symbolic_input.name]
            for symbolic_input in self._task_metadata.symbolic_inputs.values()
        }
        num_samples = len(data[next(iter(self._task_metadata.raw_inputs.keys()))])
        program = [
            "sample({}).".format(i)
            for i in range(num_samples)
        ]
        program += [
            "holds({}, {}, {}, {}).".format(sample_id,
                                            self._task_metadata.raw_inputs[raw_input_name].latent_concept_name,
                                            raw_input_name,
                                            self._task_metadata.latent_concepts[
                                                self._task_metadata.raw_inputs[raw_input_name].latent_concept_name].values[
                                                assignment])
            for raw_input_name, assignments in concept_assignments.items()
            for sample_id, assignment in enumerate(assignments)
        ] + [
            "holds({}, {}, {}, {}).".format(sample_id,
                                            self._task_metadata.symbolic_inputs[symbolic_input_name].symbolic_concept_name,
                                            symbolic_input_name,
                                            symbolic_input_val)
            for symbolic_input_name, symbolic_input_vals in symbolic_inputs.items()
            for sample_id, symbolic_input_val in enumerate(symbolic_input_vals)
        ]
        program.append("")
        if len(self._rules) > 0:
            program += self._rules
            program.append("")
        if len(self.learned_rules) > 0:
            program += self.learned_rules
            program.append("")

        program.append("#show holds/3.")

        # Uncomment to debug inference answer set program
        #with open("test_classification.p", 'w') as test_fp:
        #    test_fp.write("\n".join(program))

        clingo_output = Clingo().classify("\n".join(program), class_name="holds", num_class_args=3)
        clingo_output = [output for output in clingo_output if output[1] in self._task_metadata.targets.keys()]
        if symbolic_mode:
            predictions = {
                int(output[0]): {
                    target_name: []
                    for target_name in self._task_metadata.targets.keys()
                }
                for output in clingo_output
            }
            for output in clingo_output:
                predictions[int(output[0])][output[1]].append(output[2])
        else:
            predictions = {
                int(output[0]): {
                    target.name: np.zeros(shape=(len(self._task_metadata.target_concepts[target.target_concept_name].values),), dtype=np.float32)
                    for target in self._task_metadata.targets.values()
                }
                for output in clingo_output
            }
            for output in clingo_output:
                value_index = self._task_metadata.target_concepts[self._task_metadata.targets[output[1]].target_concept_name].values.index(output[2])
                predictions[int(output[0])][output[1]][value_index] = 1.0
        sorted_predictions = [np.empty(shape=(len(predictions), len(self._task_metadata.target_concepts[target.target_concept_name].values)))
                              for target in self._task_metadata.targets.values()]
        for i in range(len(predictions)):
            for j, target_name in enumerate(self._task_metadata.targets.keys()):
                sorted_predictions[j][i] = predictions[i][target_name]

        return tuple(sorted_predictions)

    def evaluate_fully_neural_model_on_multiple_targets(self, data_creator, valid_test):
        batch_size = self._config["test_batch_size"]
        ds = data_creator.get_full_dataset(valid_test, shuffle=False)
        target_predictions = self.neural_only_model.predict(ds, batch_size=batch_size)
        target_predictions = np.argmax(np.stack(target_predictions, axis=1), axis=-1)
        avg_accuracy = 0.0
        for batch_num, (_, labels) in enumerate(ds):
            labels = np.argmax(np.stack(labels, axis=1), axis=-1)
            batch_predictions = target_predictions[batch_num*batch_size:(batch_num*batch_size + len(labels))]
            batch_avg = np.average(np.all(batch_predictions == labels, axis=1))
            avg_accuracy = batch_num * avg_accuracy / (batch_num+1) + batch_avg / (batch_num+1)
        print("Neural only acc on multiple targets = {}".format(avg_accuracy))

    def evaluate_neural_only_model(self, data_creator, valid_test):
        assert valid_test in ["valid", "test"]
        self.neural_only_model.evaluate(x=data_creator.get_full_dataset(valid_test, shuffle=False), batch_size=self._config["batch_size"])

    def evaluate_clustering(self, latent_concept_name, data_creator, valid_test):
        assert valid_test in ["valid", "test"]
        raw_input_ds = data_creator.get_latent_concept_dataset(latent_concept_name, valid_test, shuffle=False)
        latent_embeddings_model = self.latent_embeddings_models[latent_concept_name]
        all_predictions = []
        all_labels = []
        for inputs, one_zero_labels in raw_input_ds:
            all_labels += np.argmax(one_zero_labels, axis=1).tolist()
            all_predictions += self.predict_latent_concept_model(latent_embeddings_model, latent_concept_name, inputs, return_without_mapping=True).tolist()
        for metric_name, metric_func in CLUSTERING_METRICS.items():
            print("{} = {}".format(metric_name, metric_func(all_labels, all_predictions)))


    def evaluate_latent_concept_model(self, latent_concept_name, data_creator, valid_test):
        assert valid_test in ["valid", "test"]
        average_accuracy = 0.0
        N = 0
        raw_input_ds = data_creator.get_latent_concept_dataset(latent_concept_name, valid_test, shuffle=False)
        latent_embeddings_model = self.latent_embeddings_models[latent_concept_name]
        all_labels = []
        all_predictions = []
        for inputs, one_zero_labels in raw_input_ds:
            labels = np.argmax(one_zero_labels, axis=1)
            all_labels += labels.tolist()
            predictions = self.predict_latent_concept_model(latent_embeddings_model, latent_concept_name, inputs)
            all_predictions += predictions.tolist()
            average_accuracy = (N / (N + len(inputs))) * average_accuracy + (len(inputs) / (N + len(inputs))) * np.average(predictions == labels)
            N += len(inputs)
        print("{} average accuracy = {}".format(latent_concept_name, average_accuracy))
        #print("{} confusion matrix = {}".format(latent_concept_name, confusion_matrix(all_labels, all_predictions)))


    def evaluate_neural_symbolic_model(self, data_creator, valid_test):
        assert valid_test in ["valid", "test"]
        ds = data_creator.get_full_dataset(valid_test, shuffle=False)
        average_accuracy = 0.0
        num_raw_inputs = len(self._task_metadata.raw_inputs)
        N = 0
        for inputs, one_zero_labels in ds:
            data = {
                raw_input_name: inputs[i]
                for i, raw_input_name in enumerate(self._task_metadata.raw_inputs.keys())
            }
            data.update({
                symbolic_input.name: [self._task_metadata.symbolic_concepts[symbolic_input.symbolic_concept_name].values[idx] for idx in np.argmax(inputs[i+num_raw_inputs], axis=-1)]
                for i, symbolic_input in enumerate(self._task_metadata.symbolic_inputs.values())
            })

            predictions = self.predict_neural_symbolic(data)


            batch_accuracy = np.average(np.all(np.array([np.all(target_predictions == labels, axis=-1)
                                       for target_predictions, labels in zip(predictions, one_zero_labels)]), axis=0))
            average_accuracy = (N / (N + len(inputs[0]))) * average_accuracy + (len(inputs[0]) / (N + len(inputs[0]))) * batch_accuracy
            N += len(inputs[0])
        print("Neural-Symbolic average accuracy = {}".format(average_accuracy))

    def save(self, save_dir, filename):
        makedirs(save_dir, exist_ok=True)
        for network_name, network in self._perception_networks.items():
            network.save_weights(path.join(save_dir, "{}_{}.h5".format(filename, network_name)))
        self._reasoning_network.save_weights(path.join(save_dir, "{}_neural_reasoning.h5".format(filename)))
        with open(path.join(save_dir, "{}_clusters.p".format(filename)), 'wb') as cluster_fp:
            pickle.dump(self._clusters, cluster_fp)
        logic_info = {
            "cluster_mapping": {
                latent_concept_name: mapping.tolist()
                for latent_concept_name, mapping in self._cluster_mapping.items()
            },
            "learned_rules": self.learned_rules
        }
        with open(path.join(save_dir, "{}_logic_info.json".format(filename)), 'w') as json_fp:
            json.dump(logic_info, json_fp)

    def load_network(self, load_dir, filename, network_name):
        load_path = path.join(load_dir, "{}_{}.h5".format(filename, network_name))
        self._perception_networks[network_name].load_weights(load_path)

    def load_clusters(self, load_dir, filename, latent_concepts = None):
        with open(path.join(load_dir, "{}_clusters.p".format(filename)), 'rb') as cluster_fp:
            loaded_cluster_info = pickle.load(cluster_fp)

        if latent_concepts is None:
            latent_concepts = self._task_metadata.latent_concepts.keys()
        for latent_concept in latent_concepts:
            self._clusters[latent_concept] = loaded_cluster_info[latent_concept]

    def load_logic_info(self, load_dir, filename, latent_concepts = None):
        with open(path.join(load_dir, "{}_logic_info.json".format(filename)), 'r') as json_fp:
            loaded_logic_info = json.load(json_fp)
        self.learned_rules = loaded_logic_info["learned_rules"]
        if latent_concepts is None:
            latent_concepts = self._task_metadata.latent_concepts.keys()
            self._cluster_mapping = dict()
        for latent_concept in latent_concepts:
            self._cluster_mapping.update({
                latent_concept: np.array(loaded_logic_info["cluster_mapping"][latent_concept])
            })


    @staticmethod
    def load(load_dir, filename, task):
        model = EmbeddingReasoningModel(task.metadata, task.config)
        for rule in task.rules:
            model.add_rule(rule)
        for rule in task.training_only_rules:
            model.add_training_only_rule(rule)
        for net_name, network, latent_concept_name in task.get_perception_networks():
            model.add_perception_network(net_name, network, latent_concept_name)
        model.set_reasoning_network(task.get_reasoning_network())
        model.build_neural_models()
        model._set_up_clusters()
        for network_name in model._perception_networks.keys():
            model.load_network(load_dir, filename, network_name)
        model.load_clusters(load_dir, filename)
        model.load_logic_info(load_dir, filename)
        return model













