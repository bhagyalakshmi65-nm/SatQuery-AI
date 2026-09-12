"""
Result formatting utilities for structured JSON-compatible outputs across the GeoEngine.
"""

from typing import Any, Dict, Optional


def format_response(
    operation: str,
    success: bool,
    data: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None
) -> Dict[str, Any]:
    """
    Format every Geo Engine function output as a standardized Python dictionary.
    
    Args:
        operation: Name of the operation executed.
        success: True if successful, False otherwise.
        data: Payload dictionary containing operation results.
        error: Error message string if success is False.
        
    Returns:
        Structured response dictionary.
    """
    res = {
        "operation": operation,
        "success": success
    }
    if error is not None:
        res["error"] = str(error)
    if data is not None:
        res.update(data)
    return res


def format_error(operation: str, error_message: str) -> Dict[str, Any]:
    """
    Shortcut helper for generating structured error responses.
    
    Args:
        operation: Name of the operation where error occurred.
        error_message: Detailed error string.
        
    Returns:
        Structured error response dictionary.
    """
    return format_response(operation=operation, success=False, error=error_message)
