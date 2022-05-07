from os import path
import json

from framework.metadata import TaskMetadata



class Task:
    TASKS_BASE_DIR = path.join("embedding_reasoning", "experiments")

    def __init__(self, task_name, dataset_creator_class, filepaths = dict()):
        self.task_name = task_name
        self._load_metadata(filepaths)
        self._load_config(filepaths)
        self._load_dataset_creator(dataset_creator_class, filepaths)
        self._load_rules(filepaths)

    def _load_metadata(self, filepaths):
        metadata_filepath = filepaths["metadata"] if "metadata" in filepaths else path.join(Task.TASKS_BASE_DIR, self.task_name, "metadata.json")
        self.metadata = TaskMetadata.from_json(metadata_filepath)

    def _load_config(self, filepaths):
        config_filepath = filepaths["config"] if "config" in filepaths else path.join(Task.TASKS_BASE_DIR, self.task_name, "config.json")
        with open(config_filepath, 'r') as config_fp:
            self.config = json.load(config_fp)

    def _load_dataset_creator(self, dataset_creator_class, filepaths):
        data_filepath = filepaths["data"] if "data" in filepaths else path.join(Task.TASKS_BASE_DIR, self.task_name, "data.json")
        self.dataset_creator = dataset_creator_class.from_json(data_filepath, dataset_creator_class, self.metadata, self.config)

    def _load_rules(self, filepaths):
        rules_filepath = filepaths["rules"] if "rules" in filepaths else path.join(Task.TASKS_BASE_DIR, self.task_name, "rules.json")
        with open(rules_filepath, 'r') as rules_fp:
            rules_data = json.load(rules_fp)
        self.rules = rules_data["rules"]
        self.training_only_rules = rules_data["training_only_rules"]


