import numpy as np
import pandas as pd
from zenml import pipeline, step
from zenml.config import DockerSettings
# from materializer.custom_materializer import cs_materializer
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.steps import BaseParameters, Output
from steps.clean_data import clean_data
from steps.evaluation import evaluate_model
from steps.ingest_data import ingest_df
from steps.model_train import train_model

# the settings inside docker that should be followed
docker_settings = DockerSettings(required_integrations=[MLFLOW])

class DeployementTriggerConfig(BaseParameters):
    """Deployment trigger config"""
    min_accuracy: float = 0

@step
def deployment_trigger(
    accuracy: float,
    config: DeployementTriggerConfig,
):
    """Implements a simple model deployement trigger that looks at the input model accuracy and decided if it is good enough to deploy or not"""
    return accuracy >= config.min_accuracy

class MLFlowDeploymentLoaderStepParameters(BaseParameters):
    """MLflow deployment geter parameters
    Attribute:
        pipeline_name: name of the pipeline that deployed the MLflow prediction
            server
        step_name: the name of the step that deployed the MLflow prediction
            server
        running: when this flag is set, the step only returns a running service
        model_name: the name of the model that is deployed
    """
    pipeline_name: str
    step_name: str
    running: bool = True

@step(enable_cache=False)
def prediction_service_loader(
    pipeline_name: str,
    pipeline_step_name: str,
    running: bool = True,
    model_name: str = "model",
) -> MLFlowDeploymentService:
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()
    existing_services = mlflow_model_deployer_component.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name=model_name,
        running=running,
    )

    if not existing_services:
        raise RuntimeError(
            f"No running services found for pipeline {pipeline_name} and step {pipeline_step_name}"
            f"step {pipeline_step_name} and model {model_name}"
            f"pipeline for the '{model_name}' model is currently"
            f"running."
        )
    return existing_services[0]


@step
def predictor(
    service: MLFlowDeploymentService,
    data: np.ndarray,
    
)


@pipeline(enable_cache=False, settings={"docker":docker_settings})
def continous_deployment_pipeline(
    data_path: str,
    min_accuracy: float = 0,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
):
    df = ingest_df(data_path=data_path)
    X_train, X_test, y_train, y_test = clean_data(df)
    model = train_model(X_train, X_test, y_train, y_test)
    r2_score, rmse = evaluate_model(model, X_test, y_test)
    deploy_decision = deployment_trigger(r2_score)
    mlflow_model_deployer_step(
        model=model,
        deploy_decision=deploy_decision,
        workers=workers,
        timeout=timeout,
    )