from experiments.experiment_runner import run_experiment
import argparse

task_to_latent_concepts = {
    "mnist_addition": ["digit"],
    "member": ["element"]
}

task_to_valid_N = {
    "mnist_addition": [1,2,3,4,15],
    "cifar10_addition": [1],
    "member": [3,4,5,20]
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Embed2Sym Experiment Runner", description="Run experiments presented in the paper \"Embed2Sym - Scalable Neuro-Symbolic Reasoing via Clustered Embeddings\"")
    parser.add_argument("task", type=str, choices=["mnist_addition", "cifar10_addition", "member"], help="Experiment to run")
    parser.add_argument("N", type=int, help="For MNIST Addition - number of digits per summand. (1, 2, 3, 4 or 15). For Member - number of elements in set. (3,4,5 or 20)")
    args = parser.parse_args()

    if args.N not in task_to_valid_N[args.task]:
        parser.error("For {} task, N must be one of {}".format(args.task, task_to_valid_N[args.task]))

    if args.task == "cifar10_addition":
        task_name = "mnist_addition"
        cifar10 = True
    else:
        task_name = args.task
        cifar10 = False

    task_args = {
        "n": args.N
    }
    if task_name == "mnist_addition":
        task_args["cifar10"] = cifar10

    run_experiment(task_name=task_name,
                   latent_concepts_to_evaluate=task_to_latent_concepts[task_name],
                   task_args=task_args,
                   results_base="results")