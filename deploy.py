"""
SageMaker JumpStart model deployment and management script.

This module provides functions to deploy, list, and delete SageMaker models.
"""

import logging
import boto3
import sagemaker
from sagemaker.jumpstart.model import JumpStartModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_ROLE = "arn:aws:iam::548692878254:role/service-role/AmazonSageMaker-ExecutionRole-20250211T162036"
DEFAULT_MODEL_ID = "huggingface-llm-mistral-7b-v3"
DEFAULT_MODEL_VERSION = "*"  # You can use "1.1.6" for more stable results


def deploy_model(
    model_id=DEFAULT_MODEL_ID,
    model_version=DEFAULT_MODEL_VERSION,
    role=DEFAULT_ROLE,
    instance_type="ml.g5.2xlarge",
    accept_eula=True,
):
    """
    Deploy a JumpStart model to a SageMaker endpoint.

    Args:
        model_id: The ID of the model to deploy
        model_version: The version of the model to deploy
        role: The IAM role ARN to use for the model
        instance_type: The instance type to deploy on
        accept_eula: Whether to accept the end-user license agreement

    Returns:
        A predictor object for the deployed model
    """
    logger.info(f"Deploying model {model_id} (version {model_version}) on {instance_type}")
    
    # Initialize the model with the role
    model = JumpStartModel(
        model_id=model_id,
        model_version=model_version,
        role=role,
    )
    
    # Deploy the model
    predictor = model.deploy(
        accept_eula=accept_eula,
        instance_type=instance_type,
    )
    
    logger.info(f"Successfully deployed model to endpoint: {predictor.endpoint_name}")
    return predictor


def list_deployed_models():
    """
    List all SageMaker models and endpoints.
    
    Returns:
        Raw response objects for models and endpoints
    """
    sagemaker_client = boto3.client("sagemaker")
    
    # Get all models
    logger.info("Retrieving all SageMaker models")
    models_response = sagemaker_client.list_models()
    
    # Handle pagination for models
    all_models = models_response.copy()
    while "NextToken" in models_response:
        models_response = sagemaker_client.list_models(NextToken=models_response["NextToken"])
        all_models["Models"].extend(models_response["Models"])
        
    # Get all endpoints
    logger.info("Retrieving all SageMaker endpoints")
    endpoints_response = sagemaker_client.list_endpoints()
    
    # Handle pagination for endpoints
    all_endpoints = endpoints_response.copy()
    while "NextToken" in endpoints_response:
        endpoints_response = sagemaker_client.list_endpoints(
            NextToken=endpoints_response["NextToken"]
        )
        all_endpoints["Endpoints"].extend(endpoints_response["Endpoints"])
    
    return all_models, all_endpoints


def delete_deployed_model(endpoint_name, delete_model=True):
    """
    Delete a SageMaker endpoint and optionally its model.
    
    Args:
        endpoint_name: Name of the endpoint to delete
        delete_model: Whether to also delete the associated model(s)
    
    Returns:
        True if deletion was successful, False otherwise
    """
    sagemaker_client = boto3.client("sagemaker")
    
    # Get the endpoint configuration before deleting the endpoint
    try:
        logger.info(f"Retrieving details for endpoint: {endpoint_name}")
        endpoint_details = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
        endpoint_config_name = endpoint_details["EndpointConfigName"]
        
        # Get model names from the endpoint config
        endpoint_config = sagemaker_client.describe_endpoint_config(
            EndpointConfigName=endpoint_config_name
        )
        model_names = [
            variant["ModelName"] for variant in endpoint_config["ProductionVariants"]
        ]
        
        # Delete endpoint
        logger.info(f"Deleting endpoint: {endpoint_name}")
        sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
        
        # Delete endpoint configuration
        logger.info(f"Deleting endpoint configuration: {endpoint_config_name}")
        sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
        
        # Delete models if requested
        if delete_model:
            for model_name in model_names:
                logger.info(f"Deleting model: {model_name}")
                sagemaker_client.delete_model(ModelName=model_name)
        
        return True
    except Exception as e:
        logger.error(f"Error deleting endpoint and resources: {str(e)}")
        return False


def main():
    """
    Main function to demonstrate SageMaker model deployment and management.
    """
    # Deploy a model
    predictor = deploy_model()
    logger.info(f"Deployed model to endpoint: {predictor.endpoint_name}")
    
    # List all deployed models
    list_deployed_models()
    
    # Example of how to delete a model (commented out to prevent accidental deletion)
    # endpoint_name = predictor.endpoint_name
    # delete_deployed_model(endpoint_name)


if __name__ == "__main__":
    all_models, all_endpoints = list_deployed_models()
    print(all_endpoints)
    # main()

