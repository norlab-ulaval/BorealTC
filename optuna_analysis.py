import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_intermediate_values,
    plot_param_importances,
    plot_slice,
)

if __name__ == "__main__":
    model = "SVM"
    dataset = "vulpi"
    name = f"{model}_{dataset}"
    storage_name = f"sqlite:///results/{dataset}/optuna/{name}.db"
    print(f"Loading study {name} from {storage_name}")

    sampler = optuna.samplers.RandomSampler(seed=420)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name=name,
        storage=storage_name,
        load_if_exists=True,
    )
    print(f"There was {len(study.trials)} trials completed")
    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    plot_optimization_history(study).show()
    plot_intermediate_values(study).show()

    fig = plot_param_importances(study)
    fig.layout.title = name
    fig.show()

    fig = plot_slice(study)
    fig.layout.title = name
    fig.show()

    study_df = study.trials_dataframe()
    study_df = study_df[(study_df["value"] > 0.99)]
    print(study_df.to_dict())
