import optuna
from optuna.visualization import plot_optimization_history, plot_intermediate_values

if __name__ == "__main__":
    name = "SVM_vulpi"
    storage_name = f"sqlite:///results/optuna/{name}.db"

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

    fig = optuna.visualization.plot_param_importances(study)
    fig.layout.title = name
    fig.show()
