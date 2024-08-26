from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from clearml import Logger
from clearml import Task
import matplotlib.pyplot as plt

from clear_ml_secret import CLEAR_ML_KEY
from clear_ml_secret import CLEAR_ML_SECRET


class ClearMlRegistrator:
    def __init__(self,
                 task_name: Optional[str] = None,
                 tags: Optional[List[str]] = None,
                 task_id: Optional[str] = None,
                 config_path: Optional[Union[str, Path]] = None):
        Task.set_credentials(
            api_host="https://api.clear.ml",
            web_host="https://app.clear.ml",
            files_host="https://files.clear.ml",
            key=CLEAR_ML_KEY,
            secret=CLEAR_ML_SECRET
        )
        if task_name is not None and tags is not None:
            self._task = Task.init(project_name='PointClouds',
                                   task_name=task_name,
                                   auto_connect_frameworks={'pytorch': '*.pth'},
                                   tags=tags)
        elif task_id is not None:
            self._task = Task.init(auto_connect_frameworks={'pytorch': '*.pth'},
                                   continue_last_task=task_id)
        else:
            raise ValueError("Either task_name or task_id must be provided.")
        if config_path is not None:
            self._task.connect_configuration(config_path)
        self._log = Logger.current_logger()

    def report_single_value(self, name: str, value: float):
        self._log.report_single_value(name, value)

    def report_scalar(self,
                      title: str,
                      series: str,
                      value: float,
                      iteration: int):
        self._log.report_scalar(title, series, value, iteration)

    def set_up_current_logger(self):
        self._log = Logger.current_logger()

    def upload_artifact(self, name: str, artifact: Any):
        self._task.upload_artifact(name, artifact)

    def report_matplotlib_figure(self,
                                 title: str,
                                 series: str,
                                 figure: plt.Figure,
                                 iteration: Optional[str] = None,
                                 report_image: bool = False,
                                 report_interactive: bool = True):
        self._log.report_matplotlib_figure(title, series, figure, iteration, report_image, report_interactive)

    def upload_evaluation_artifacts(self,
                                    metrics_dict: Dict[str, Any],
                                    train_val_prefix: str):
        for metric_name, metric_value in metrics_dict.items():
            if isinstance(metric_value, dict):
                self.upload_artifact(
                    name=f"{train_val_prefix}_{metric_name}",
                    artifact=metric_value,
                )

    @property
    def task_models(self):
        return self._task.get_models()

    @property
    def task_name(self):
        return self._task.name

    @property
    def config_dict(self):
        return self._task.get_configuration_object_as_dict("General")
