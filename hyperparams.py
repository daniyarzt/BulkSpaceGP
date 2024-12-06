from avalanche.benchmarks import SplitMNIST
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import forgetting_metrics, accuracy_metrics, loss_metrics, StreamConfusionMatrix
from avalanche.logging import InteractiveLogger
from avalanche.models import SimpleMLP
from avalanche.training import EWC
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from ray import tune, train
import ray
from ray.tune.schedulers import ASHAScheduler
import torch


# Define evaluation components
benchmark = SplitMNIST(n_experiences=2)
text_logger = InteractiveLogger()
eval_plugin = EvaluationPlugin(
    accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    forgetting_metrics(experience=True, stream=True),
    StreamConfusionMatrix(num_classes=benchmark.n_classes, save_image=False),
    loggers=[text_logger]
)

# Define the benchmark outside of train_ewc and use ray.put
benchmark_id = ray.put(SplitMNIST(n_experiences=2))

# Define training function for Ray Tune
def train_ewc(config, benchmark_id):

    # Retrieve the benchmark from Ray object store
    benchmark = benchmark_id

    # Create the model and EWC strategy with config-defined parameters
    model = SimpleMLP(num_classes=benchmark.n_classes, hidden_layers=2)
    optimizer = SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"])
    strategy = EWC(
        model=model,
        optimizer=optimizer,
        criterion=CrossEntropyLoss(),
        ewc_lambda=config["ewc_lambda"],
        train_mb_size=config["train_mb_size"],
        train_epochs=5,
        eval_mb_size=100,
        evaluator=eval_plugin
    )

    results_ewc = []
    # Train on each experience in the train stream
    for exp_id, experience in enumerate(benchmark.train_stream):
        strategy.train(experience)
        results_ewc.append(strategy.eval(benchmark.test_stream[exp_id]))
    
    results = strategy.eval(benchmark.test_stream)

    # get the trace of the tensor results[x]
    confusion_trace = torch.trace(results["ConfusionMatrix_Stream/eval_phase/test_stream"]).item()

    # Cannot store tensor objects in Ray Tune
    #results["ConfusionMatrix_Stream/eval_phase/test_stream"] = results["ConfusionMatrix_Stream/eval_phase/test_stream"].tolist()
    selected_results = {"accuracy" : results["Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp001"], "confusion_trace" : confusion_trace}
    # Report accuracy for tuning purposes
    train.report(selected_results)

# Define the hyperparameter search space
search_space = {
    "lr": tune.loguniform(1e-5, 1e-3),
    "momentum": tune.uniform(0, 0.9),
    "train_mb_size": tune.choice([35]),
    "ewc_lambda": tune.uniform(10, 5000)
}

max_num_epochs = 5
num_samples = 10

scheduler = ASHAScheduler(
    max_t=max_num_epochs,
    grace_period=1,
    reduction_factor=2)

tuner = tune.Tuner(
    tune.with_resources(
        tune.with_parameters(train_ewc, benchmark_id=benchmark_id),
        resources={"cpu": 8, "gpu": 0}
    ),
    tune_config=tune.TuneConfig(
        metric="confusion_trace",
        mode="min",
        scheduler=scheduler,
        num_samples=num_samples,
    ),
    param_space=search_space,
)

results = tuner.fit()