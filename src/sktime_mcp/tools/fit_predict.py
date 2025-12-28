"""
fit_predict tool for sktime MCP.

Executes complete forecasting workflows.
"""

from typing import Any, Dict, List, Optional, Union

from sktime_mcp.runtime.executor import get_executor


def fit_predict_tool(
    estimator_handle: str,
    dataset: str,
    horizon: int = 12,
) -> Dict[str, Any]:
    """
    Execute a complete fit-predict workflow.
    
    Args:
        estimator_handle: Handle from instantiate_estimator
        dataset: Name of demo dataset (e.g., "airline", "sunspots")
        horizon: Forecast horizon (default: 12)
    
    Returns:
        Dictionary with:
        - success: bool
        - predictions: Forecast values
        - horizon: Number of steps predicted
    
    Example:
        >>> fit_predict_tool("est_abc123", "airline", horizon=12)
        {
            "success": True,
            "predictions": {1: 450.2, 2: 460.5, ...},
            "horizon": 12
        }
    """
    executor = get_executor()
    return executor.fit_predict(estimator_handle, dataset, horizon)


def fit_tool(
    estimator_handle: str,
    dataset: str,
) -> Dict[str, Any]:
    """
    Fit an estimator on a dataset.
    
    Args:
        estimator_handle: Handle from instantiate_estimator
        dataset: Name of demo dataset
    
    Returns:
        Dictionary with success status
    """
    executor = get_executor()
    data_result = executor.load_dataset(dataset)
    if not data_result["success"]:
        return data_result
    
    return executor.fit(
        estimator_handle,
        y=data_result["data"],
        X=data_result.get("exog"),
    )


def predict_tool(
    estimator_handle: str,
    horizon: int = 12,
) -> Dict[str, Any]:
    """
    Generate predictions from a fitted estimator.
    
    Args:
        estimator_handle: Handle of a fitted estimator
        horizon: Forecast horizon
    
    Returns:
        Dictionary with predictions
    """
    executor = get_executor()
    fh = list(range(1, horizon + 1))
    return executor.predict(estimator_handle, fh=fh)


def list_datasets_tool() -> Dict[str, Any]:
    """
    List available demo datasets.
    
    Returns:
        Dictionary with list of dataset names
    """
    executor = get_executor()
    return {
        "success": True,
        "datasets": executor.list_datasets(),
    }
