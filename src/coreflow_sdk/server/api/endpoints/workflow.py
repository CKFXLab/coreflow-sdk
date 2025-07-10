"""
Workflow endpoint for workflow status and management.
"""

from fastapi import APIRouter, HTTPException
import time

from ....workflow import BaseWorkflow
from ....utils.env import ENV

router = APIRouter()


@router.get("/status")
async def get_workflow_status():
    """
    Get workflow component status and configuration.

    Returns:
        Workflow status information
    """
    try:
        workflow = BaseWorkflow()
        status = workflow.get_component_status()
        env = ENV()
        credentials = env.get_available_credentials()
        disabled_features = env.get_disabled_features()

        return {
            "credentials": credentials,
            "disabled_features": disabled_features,
            "component_status": status,
            "timestamp": time.time(),
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get workflow status: {str(e)}"
        )


@router.get("/info")
async def get_workflow_info():
    """
    Get workflow configuration information.

    Returns:
        Workflow configuration details
    """
    try:
        workflow = BaseWorkflow()
        info = workflow.get_workflow_info()

        return {"workflow_info": info, "timestamp": time.time()}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get workflow info: {str(e)}"
        )


@router.get("/validate")
async def validate_workflow():
    """
    Validate workflow configuration and components.

    Returns:
        Validation results
    """
    try:
        workflow = BaseWorkflow()
        validation = workflow.validate_workflow()

        return {"validation": validation, "timestamp": time.time()}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to validate workflow: {str(e)}"
        )
