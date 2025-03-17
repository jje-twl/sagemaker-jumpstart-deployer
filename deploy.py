#!/usr/bin/env python3
"""
SageMaker JumpStart CLI

A command-line tool for deploying, listing, and deleting SageMaker JumpStart models.
"""

import argparse
import logging
import sys
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
DEFAULT_MODEL_VERSION = "*"
DEFAULT_INSTANCE_TYPE = "ml.g5.2xlarge"


def deploy_model(
    model_id=DEFAULT_MODEL_ID,
    model_version=DEFAULT_MODEL_VERSION,
    role=DEFAULT_ROLE,
    instance_type=DEFAULT_INSTANCE_TYPE,
    accept_eula=True,
    wait=False,
):
    """Deploy a JumpStart model to a SageMaker endpoint."""
    logger.info(f"Deploying model {model_id} (version {model_version}) on {instance_type}")
    
    model = JumpStartModel(
        model_id=model_id,
        model_version=model_version,
        role=role,
    )
    
    predictor = model.deploy(
        accept_eula=accept_eula,
        instance_type=instance_type,
        wait=wait,
    )
    
    if wait:
        logger.info(f"Successfully deployed model to endpoint: {predictor.endpoint_name}")
    else:
        logger.info(f"Started deployment of model to endpoint: {predictor.endpoint_name}")
        logger.info("Deployment will continue in the background.")
    
    return predictor


def list_deployed_models():
    """List all SageMaker models and endpoints."""
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
    """Delete a SageMaker endpoint and optionally its model."""
    sagemaker_client = boto3.client("sagemaker")
    
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


def print_endpoint_list(endpoints):
    """Print formatted endpoint information."""
    if not endpoints.get("Endpoints"):
        print("No endpoints found.")
        return
        
    print("\nSageMaker Endpoints:")
    print("=" * 80)
    print(f"{'Endpoint Name':<50} {'Status':<15} {'Created':<20}")
    print("-" * 80)
    
    for endpoint in endpoints["Endpoints"]:
        print(f"{endpoint['EndpointName']:<50} {endpoint['EndpointStatus']:<15} {endpoint['CreationTime'].strftime('%Y-%m-%d %H:%M:%S')}")


def print_model_list(models):
    """Print formatted model information."""
    if not models.get("Models"):
        print("No models found.")
        return
        
    print("\nSageMaker Models:")
    print("=" * 80)
    print(f"{'Model Name':<50} {'Created':<20}")
    print("-" * 80)
    
    for model in models["Models"]:
        print(f"{model['ModelName']:<50} {model['CreationTime'].strftime('%Y-%m-%d %H:%M:%S')}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="SageMaker JumpStart model deployment and management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
---------
  # Deploy a model (async by default):
  python deploy.py deploy --model-id huggingface-llm-mistral-7b-v3 --model-version "*"
  
  # Deploy a model and wait for completion:
  python deploy.py deploy --model-id huggingface-llm-mistral-7b-v3 --wait
  
  # List all endpoints and models:
  python deploy.py list
  
  # List only endpoints:
  python deploy.py list --endpoints-only
  
  # List only models:
  python deploy.py list --models-only
  
  # Delete an endpoint:
  python deploy.py delete --endpoint-name my-endpoint-name
  
  # Delete an endpoint but keep the model:
  python deploy.py delete --endpoint-name my-endpoint-name --keep-model
"""
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Deploy command
    deploy_parser = subparsers.add_parser("deploy", help="Deploy a SageMaker model to an endpoint")
    deploy_parser.add_argument(
        "--model-id", 
        default=DEFAULT_MODEL_ID,
        help=f"JumpStart model ID (default: {DEFAULT_MODEL_ID})"
    )
    deploy_parser.add_argument(
        "--model-version", 
        default=DEFAULT_MODEL_VERSION,
        help=f"Model version (default: {DEFAULT_MODEL_VERSION})"
    )
    deploy_parser.add_argument(
        "--instance-type", 
        default=DEFAULT_INSTANCE_TYPE,
        help=f"Instance type for deployment (default: {DEFAULT_INSTANCE_TYPE})"
    )
    deploy_parser.add_argument(
        "--role", 
        default=DEFAULT_ROLE,
        help="SageMaker execution role ARN"
    )
    deploy_parser.add_argument(
        "--no-eula", 
        action="store_false",
        dest="accept_eula",
        help="Do not accept the end-user license agreement"
    )
    deploy_parser.add_argument(
        "--wait", 
        action="store_true",
        help="Wait for deployment to complete before returning (default: async deployment)"
    )
    
    # List command
    list_parser = subparsers.add_parser("list", help="List SageMaker models and endpoints")
    list_parser.add_argument(
        "--endpoints-only", 
        action="store_true",
        help="List only endpoints, not models"
    )
    list_parser.add_argument(
        "--models-only", 
        action="store_true",
        help="List only models, not endpoints"
    )
    list_parser.add_argument(
        "--json", 
        action="store_true",
        help="Output raw JSON instead of formatted tables"
    )
    
    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a SageMaker endpoint")
    delete_parser.add_argument(
        "--endpoint-name", 
        required=True,
        help="Name of the endpoint to delete"
    )
    delete_parser.add_argument(
        "--keep-model", 
        action="store_false",
        dest="delete_model",
        help="Keep the model when deleting the endpoint"
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the CLI."""
    args = parse_args()
    
    if not args.command:
        print("Error: No command specified")
        print("Use --help for usage information")
        return 1
    
    if args.command == "deploy":
        predictor = deploy_model(
            model_id=args.model_id,
            model_version=args.model_version,
            role=args.role,
            instance_type=args.instance_type,
            accept_eula=args.accept_eula,
            wait=args.wait,
        )
        print(f"Successfully initiated deployment of model: {args.model_id}")
        print(f"Endpoint name: {predictor.endpoint_name}")
        
        if not args.wait:
            print("\nDeployment is continuing in the background. You can check status with:")
            print(f"  python deploy.py list --endpoints-only | grep {predictor.endpoint_name}")
        else:
            print("\nModel deployment completed successfully.")
        
    elif args.command == "list":
        models, endpoints = list_deployed_models()
        
        if args.json:
            import json
            if not args.models_only:
                print(json.dumps(endpoints, default=str))
            if not args.endpoints_only:
                print(json.dumps(models, default=str))
        else:
            if not args.models_only:
                print_endpoint_list(endpoints)
            if not args.endpoints_only:
                print_model_list(models)
                
    elif args.command == "delete":
        result = delete_deployed_model(
            endpoint_name=args.endpoint_name,
            delete_model=args.delete_model
        )
        if result:
            print(f"Successfully deleted endpoint: {args.endpoint_name}")
            if args.delete_model:
                print("Associated models were also deleted")
            else:
                print("Associated models were kept")
        else:
            print(f"Failed to delete endpoint: {args.endpoint_name}")
            return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

