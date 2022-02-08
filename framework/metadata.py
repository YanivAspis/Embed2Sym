import tensorflow as tf
import json


class Concept:
    def __init__(self, name, values):
        self.name = name
        self.values = values

    def to_dict(self):
        return {
            "name": self.name,
            "values": self.values
        }

    @staticmethod
    def from_dict(sample_dict):
        return Concept(sample_dict["name"], sample_dict["values"])


class RawInput:
    def __init__(self, name, input_shape, latent_concept_name):
        self.name = name
        self.input_shape = input_shape
        self.latent_concept_name = latent_concept_name

    def to_keras_input(self):
        return tf.keras.Input(shape=self.input_shape)

    def to_dict(self):
        return {
            "name": self.name,
            "input_shape": self.input_shape,
            "latent_concept_name": self.latent_concept_name
        }

    @staticmethod
    def from_dict(raw_input_dict):
        return RawInput(raw_input_dict["name"], raw_input_dict["input_shape"], raw_input_dict["latent_concept_name"])


class SymbolicInput:
    def __init__(self, name, symbolic_concept_name, num_values):
        self.name = name
        self.symbolic_concept_name = symbolic_concept_name
        self.num_values = num_values

    def to_keras_input(self):
        return tf.keras.Input(shape=(self.num_values,))

    def to_dict(self):
        return {
            "name": self.name,
            "symbolic_concept_name": self.symbolic_concept_name,
            "num_values": self.num_values
        }

    @staticmethod
    def from_dict(symbolic_input_dict):
        return SymbolicInput(symbolic_input_dict["name"], symbolic_input_dict["symbolic_concept_name"], symbolic_input_dict["num_values"])


class Target:
    def __init__(self, name, target_concept_name, multi_valued):
        self.name = name
        self.target_concept_name = target_concept_name
        self.multi_valued = multi_valued

    def to_dict(self):
        return {
            "name": self.name,
            "target_concept_name": self.target_concept_name,
            "multi_valued": self.multi_valued
        }

    @staticmethod
    def from_dict(target_dict):
        return Target(target_dict["name"], target_dict["target_concept_name"], target_dict["multi_valued"])



class TaskMetadata:
    def __init__(self):
        self.raw_inputs = dict()
        self.symbolic_inputs = dict()
        self.targets = dict()
        self.latent_concepts = dict()
        self.symbolic_concepts = dict()
        self.target_concepts = dict()
        self.raw_inputs_to_latent_concepts = dict()
        self.symbolic_inputs_to_symbolic_concepts = dict()
        self.target_to_target_concepts = dict()
        self.concepts_to_learn = []

    def add_latent_concept(self, latent_concept):
        self.latent_concepts[latent_concept.name] = latent_concept

    def add_symbolic_concept(self, symbolic_concept):
        self.symbolic_concepts[symbolic_concept.name] = symbolic_concept

    def add_target_concept(self, target_concept):
        self.target_concepts[target_concept.name] = target_concept

    def add_raw_input(self, raw_input):
        self.raw_inputs[raw_input.name] = raw_input
        self.raw_inputs_to_latent_concepts[raw_input.name] = raw_input.latent_concept_name

    def add_symbolic_input(self, symbolic_input):
        self.symbolic_inputs[symbolic_input.name] = symbolic_input
        self.symbolic_inputs_to_symbolic_concepts[symbolic_input.name] = symbolic_input.symbolic_concept_name

    def add_target(self, target):
        self.targets[target.name] = target
        self.target_to_target_concepts[target.name] = target.target_concept_name

    def add_concept_to_learn(self, concept_name, num_args):
        self.concepts_to_learn.append((concept_name, num_args))

    def to_dict(self):
        return {
            "latent_concepts": [latent_concept.to_dict() for latent_concept in self.latent_concepts.values()],
            "symbolic_concepts": [symbolic_concept.to_dict() for symbolic_concept in self.symbolic_concepts.values()],
            "target_concepts": [target_concept.to_dict() for target_concept in self.target_concepts.values()],
            "raw_inputs": [raw_input.to_dict() for raw_input in self.raw_inputs.values()],
            "symbolic_inputs": [symbolic_input.to_dict() for symbolic_input in self.symbolic_inputs.values()],
            "targets": [target.to_dict() for target in self.targets.values()],
            "concepts_to_learn": [{
                "name": concept_name,
                "num_args": num_args
            } for (concept_name, num_args) in self.concepts_to_learn]
        }

    @staticmethod
    def from_dict(metadata_dict):
        metadata = TaskMetadata()
        for latent_concept in metadata_dict["latent_concepts"]:
            metadata.add_latent_concept(Concept.from_dict(latent_concept))
        for symbolic_concept in metadata_dict["symbolic_concepts"]:
            metadata.add_symbolic_concept(Concept.from_dict(symbolic_concept))
        for target_concept in metadata_dict["target_concepts"]:
            metadata.add_target_concept(Concept.from_dict(target_concept))
        for raw_input in metadata_dict["raw_inputs"]:
            metadata.add_raw_input(RawInput.from_dict(raw_input))
        for symbolic_input in metadata_dict["symbolic_inputs"]:
            metadata.add_symbolic_input(SymbolicInput.from_dict(symbolic_input))
        for target in metadata_dict["targets"]:
            metadata.add_target(Target.from_dict(target))
        for concept_to_learn in metadata_dict["concepts_to_learn"]:
            metadata.add_concept_to_learn(concept_to_learn["name"], concept_to_learn["num_args"])
        return metadata

    def to_json(self, json_filepath):
        with open(json_filepath, 'w') as json_fp:
            json.dump(self.to_dict(), json_fp)

    @staticmethod
    def from_json(json_filepath):
        with open(json_filepath, 'r') as json_fp:
            return TaskMetadata.from_dict(json.load(json_fp))