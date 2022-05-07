from os import path

import tensorflow as tf
import tqdm

from framework.embed2sym_model import Embed2SymModel
from experiments.mnist_addition.task import MNISTAdditionTask
from experiments.member.task import MemberTask
from experiments.forth_sort.task import ForthSortTask

TASKS = {
    "mnist_addition": MNISTAdditionTask,
    "member": MemberTask,
    "forth_sort": ForthSortTask
}

def test_dataset(task_name, task_args):
    task = TASKS[task_name](**task_args)
    task.dataset_test()

def display_pca(task, latent_concept_name, model, train_valid_test = "valid", n_dims = 2, save_dir = None):
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    latent_model = model.latent_embeddings_models[latent_concept_name]
    ds = task.dataset_creator.get_latent_concept_dataset(latent_concept_name, train_valid_test, shuffle=False)
    ds.symbolic_labels_mode = True
    embeddings = []
    all_labels = []
    for raw_inputs, labels in tqdm.tqdm(ds):
        embeddings += latent_model.predict(raw_inputs).tolist()
        all_labels += labels.tolist()

    pca_obj = PCA(n_components=n_dims)
    embeddings_reduced = pca_obj.fit_transform(embeddings)
    datapoints = {
        label: ([], []) if n_dims == 2 else ([], [], [])
        for label in task.get_labels_for_pca(latent_concept_name)
    }
    for embedding, label in zip(embeddings_reduced, all_labels):
        digit = task.pca_label_extractor(label)
        datapoints[digit][0].append(embedding[0])
        datapoints[digit][1].append(embedding[1])
        if n_dims == 3:
            datapoints[digit][2].append(embedding[2])

    NUM_COLORS = 20
    cm = plt.get_cmap('tab20')
    fig = plt.figure()
    if n_dims == 2:
        ax = fig.add_subplot()
    else:
        ax = fig.add_subplot(projection="3d")
    colors = [cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)]
    for (digit, points), color in zip(datapoints.items(), colors[:10]):
        if n_dims == 2:
            ax.scatter(x=points[0], y=points[1], label=digit, color=color)
        else:
            ax.scatter(xs=points[0], ys=points[1], zs=points[2], label=digit, color=color)
    plt.title("{} PCA".format(latent_concept_name))
    plt.legend()
    if save_dir is not None:
        plt.savefig(path.join(save_dir, "pca_{}_{}_{}d.png".format(train_valid_test, latent_concept_name, n_dims)))
    plt.show()

def display_tsne(task, latent_concept_name, model, train_valid_test = "valid", n_dims = 2, save_dir = None):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import numpy as np

    latent_model = model.latent_embeddings_models[latent_concept_name]
    ds = task.dataset_creator.get_latent_concept_dataset(latent_concept_name, train_valid_test, shuffle=False)
    ds.symbolic_labels_mode = True
    embeddings = []
    all_labels = []
    for raw_inputs, labels in tqdm.tqdm(ds):
        embeddings += latent_model.predict(raw_inputs).tolist()
        all_labels += labels.tolist()

    if len(embeddings) > 5000:
        indices = np.arange(len(embeddings))
        np.random.shuffle(indices)
        indices = indices[:5000]
        embeddings = np.array(embeddings)[indices]
        all_labels = np.array(all_labels)[indices]

    if len(embeddings[0]) > 50:
        pca_obj = PCA(n_components=50)
        embeddings = pca_obj.fit_transform(embeddings)

    tsne_obj = TSNE(n_components=n_dims, perplexity=50, learning_rate=200, init='pca')
    embeddings_reduced = tsne_obj.fit_transform(embeddings)
    datapoints = {
        label: ([], []) if n_dims == 2 else ([], [], [])
        for label in task.get_labels_for_pca(latent_concept_name)
    }
    for embedding, label in zip(embeddings_reduced, all_labels):
        digit = task.pca_label_extractor(label)
        datapoints[digit][0].append(embedding[0])
        datapoints[digit][1].append(embedding[1])
        if n_dims == 3:
            datapoints[digit][2].append(embedding[2])

    NUM_COLORS = 20
    cm = plt.get_cmap('tab20')
    fig = plt.figure()
    if n_dims == 2:
        ax = fig.add_subplot()
    else:
        ax = fig.add_subplot(projection="3d")
    colors = [cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)]
    for (digit, points), color in zip(datapoints.items(), colors[:10]):
        if n_dims == 2:
            ax.scatter(x=points[0], y=points[1], label=digit, color=color)
        else:
            ax.scatter(xs=points[0], ys=points[1], zs=points[2], label=digit, color=color)
    plt.title("{} t-SNE".format(latent_concept_name))
    plt.legend(loc="upper left", bbox_to_anchor=(1.0, 1.01))
    if save_dir is not None:
        plt.savefig(path.join(save_dir, "tsne_{}_{}_{}d.png".format(train_valid_test, latent_concept_name, n_dims)))
    plt.show()


def run_experiment(task_name, latent_concepts_to_evaluate, task_args, results_base = None):
    results_dir = path.join(results_base, task_name) if results_base is not None else None
    checkpoint_dir = path.join(results_dir, "checkpoints") if results_dir is not None else None

    task = TASKS[task_name](**task_args)
    model = Embed2SymModel(task.metadata, task.config)
    for rule in task.rules:
        model.add_rule(rule)
    for rule in task.training_only_rules:
        model.add_training_only_rule(rule)
    for net_name, network, latent_concept_name in task.get_perception_networks():
        model.add_perception_network(net_name, network, latent_concept_name)
    model.set_reasoning_network(task.get_reasoning_network())
    model.train(task.dataset_creator, checkpoint_folder=checkpoint_dir)
    for latent_concept in latent_concepts_to_evaluate:
        #model.evaluate_clustering(latent_concept, task.dataset_creator, "valid")
        model.evaluate_latent_concept_model(latent_concept, task.dataset_creator, "test")

        # Uncomment these if you want to produce PCA/t-SNE figures
        #display_pca(task, latent_concept, model, train_valid_test="train", n_dims=2, save_dir=results_dir)
        #display_pca(task, latent_concept, model, train_valid_test="valid", n_dims=2, save_dir=results_dir)
        #display_tsne(task, latent_concept, model, train_valid_test="train", n_dims=2, save_dir=results_dir)
        #display_tsne(task, latent_concept, model, train_valid_test="valid", n_dims=2, save_dir=results_dir)
    if len(task.metadata.targets) > 1:
        model.evaluate_fully_neural_model_on_multiple_targets(task.dataset_creator, "test")
    model.evaluate_neural_symbolic_model(task.dataset_creator, "test")
    tf.keras.backend.clear_session()

