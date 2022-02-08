import numpy as np
from utils.clingo_wrapper import Clingo

class ReasoningOptimisationTask:
    def __init__(self, task_metadata, rules, samples, concepts_to_learn):
        self._latent_concepts = task_metadata.latent_concepts
        self._raw_inputs = task_metadata.raw_inputs
        self._symbolic_inputs = task_metadata.symbolic_inputs
        self._target_concepts = task_metadata.target_concepts
        self._rules = rules
        self._optimisation_program = []
        self._add_latent_concept_facts()
        self._add_background_rules()
        self._add_sample_facts(samples)
        self._add_sample_cluster_assignments(samples)
        self._add_symbolic_inputs(samples)
        self._add_target_label_weak_constraints(samples)
        self._optimisation_program.append("#show cluster_mapping/3.")
        self._clingo_obj = Clingo()
        self._concepts_to_learn = concepts_to_learn

    def _add_sample_facts(self, samples):
        self._optimisation_program += [
            "sample({}).".format(i)
            for i in range(len(samples))
        ]
        self._optimisation_program.append("")

    def _add_latent_concept_facts(self):
        self._optimisation_program += [
            "latent_concept_cluster({}, {}, {}).".format(latent_concept.name, value, idx)
            for latent_concept in self._latent_concepts.values()
            for idx, value in enumerate(latent_concept.values)
        ]
        self._optimisation_program.append("")

    def _add_background_rules(self):
        self._optimisation_program += [
            "{{cluster_mapping({}, X, Y) : Y=0..{}}} = 1 :- X=0..{}.".format(latent_concept.name, len(latent_concept.values) - 1, len(latent_concept.values) - 1)
            for latent_concept in self._latent_concepts.values()
        ]
        self._optimisation_program.append("")
        self._optimisation_program += [
            ":- cluster_mapping({}, X, Z), cluster_mapping({}, Y, Z), X != Y.".format(latent_concept.name, latent_concept.name)
            for latent_concept in self._latent_concepts.values()
        ]
        self._optimisation_program.append("")
        self._optimisation_program.append("holds(SampleID, LatentConceptName, RawInputName, LatentConceptValue) :- latent_concept_cluster(LatentConceptName, LatentConceptValue, Cluster1), sample_cluster_assignment(SampleID, LatentConceptName, RawInputName, Cluster2), cluster_mapping(LatentConceptName, Cluster2, Cluster1).")
        self._optimisation_program.append("")
        if len(self._rules) > 0:
            self._optimisation_program += self._rules
            self._optimisation_program.append("")

    def _add_sample_cluster_assignments(self, samples):
        self._optimisation_program += [
            "sample_cluster_assignment({}, {}, {}, {}).".format(idx, cluster_assignment["latent_concept_name"], cluster_assignment["raw_input_name"], cluster_assignment["cluster_index"])
            for idx, sample in enumerate(samples)
            for cluster_assignment in sample["cluster_assignments"].values()
        ]
        self._optimisation_program.append("")

    def _add_symbolic_inputs(self, samples):
        self._optimisation_program += [
            "holds({}, {}, {}, {}).".format(idx, self._symbolic_inputs[symbolic_input_name].symbolic_concept_name, symbolic_input_name, symbolic_input_value[idx])
            for idx, sample in enumerate(samples)
            for symbolic_input_name, symbolic_input_value in sample["symbolic_inputs"].items()
        ]

    def _add_target_label_weak_constraints(self, samples):
        self._optimisation_program += [
            ":~ {}. [-1, {}]".format(", ".join([
                "holds({}, {}, {})".format(idx, target_name, label)
                for target_name, label_list in sample["target_labels"].items()
                for label in label_list
            ]), idx)
            for idx, sample in enumerate(samples)
        ]
        self._optimisation_program.append("")

    def solve(self):
        class_names = ["cluster_mapping"] + [concept_to_learn[0] for concept_to_learn in self._concepts_to_learn]
        num_class_args = [3] + [concept_to_learn[1] for concept_to_learn in self._concepts_to_learn]
        results = self._clingo_obj.optimise("\n".join(self._optimisation_program), class_names=class_names, num_class_args=num_class_args)
        solution = {
            latent_concept.name: np.empty(shape=(len(latent_concept.values,)), dtype=np.uint8)
            for latent_concept in self._latent_concepts.values()
        }
        learned_rules = []
        for mapping in results["cluster_mapping"]:
            solution[mapping[0]][int(mapping[1])] = int(mapping[2])
        for concept_to_learn in self._concepts_to_learn:
            for learned_fact in results[concept_to_learn[0]]:
                learned_rules.append("{}({}).".format(concept_to_learn[0], ",".join(learned_fact)))
        return solution, learned_rules