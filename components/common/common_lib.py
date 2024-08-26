from pathlib import Path

from clearml import InputModel

from components.clear_ml_registrator.clear_ml_registrator import ClearMlRegistrator


def get_last_model_id(clear_ml_registrator: ClearMlRegistrator) -> str:
    """Returns the last model id of the task.

    Args:
        clear_ml_registrator: An instance of ClearMlRegistrator with the initiated task.

    Returns:
        String id of the last model of the provided task.
    """
    return clear_ml_registrator.task_models.output[-1].id


def get_model_path(model_id: str) -> Path:
    """Gets the path to the model using the provided id.

    Args:
        model_id: String id of the model.

    Returns:
        Path to the model.
    """
    input_model = InputModel(model_id=model_id)
    model_path = input_model.url
    model_path = Path(model_path.replace("file://", ""))
    return model_path
