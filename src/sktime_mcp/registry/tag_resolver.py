"""
Tag Resolver for sktime MCP.

Tags encode estimator capabilities and constraints. This module provides
utilities for working with tags and understanding their meanings.
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import logging

from sktime_mcp.registry.interface import EstimatorNode, get_registry

logger = logging.getLogger(__name__)


@dataclass
class TagInfo:
    """Information about a specific tag."""
    name: str
    description: str
    value_type: str  # "bool", "str", "list", etc.
    possible_values: Optional[List[Any]] = None
    category: str = "general"


class TagResolver:
    """
    Resolver for sktime estimator tags.
    
    Tags encode important semantic information about estimators:
    - Supported data types
    - Probabilistic vs deterministic predictions
    - Composability rules
    - Missing value handling
    - And many more...
    
    This class provides utilities for understanding and querying tags.
    """
    
    # Common tags and their descriptions (based on sktime documentation)
    TAG_DEFINITIONS: Dict[str, TagInfo] = {
        # Forecasting capability tags
        "capability:pred_int": TagInfo(
            name="capability:pred_int",
            description="Can produce prediction intervals (probabilistic forecasts)",
            value_type="bool",
            category="capability",
        ),
        "capability:pred_var": TagInfo(
            name="capability:pred_var",
            description="Can produce variance forecasts",
            value_type="bool",
            category="capability",
        ),
        "capability:insample": TagInfo(
            name="capability:insample",
            description="Can produce in-sample predictions",
            value_type="bool",
            category="capability",
        ),
        "capability:missing_values": TagInfo(
            name="capability:missing_values",
            description="Can handle missing values in input data",
            value_type="bool",
            category="capability",
        ),
        
        # Data type tags
        "scitype:y": TagInfo(
            name="scitype:y",
            description="Supported target data types",
            value_type="str",
            possible_values=["univariate", "multivariate", "both"],
            category="data",
        ),
        "y_inner_mtype": TagInfo(
            name="y_inner_mtype",
            description="Internal data type for y",
            value_type="str",
            category="data",
        ),
        "X_inner_mtype": TagInfo(
            name="X_inner_mtype",
            description="Internal data type for X",
            value_type="str",
            category="data",
        ),
        
        # Fit/predict behavior tags
        "requires-fh-in-fit": TagInfo(
            name="requires-fh-in-fit",
            description="Requires forecasting horizon at fit time",
            value_type="bool",
            category="behavior",
        ),
        "handles-missing-data": TagInfo(
            name="handles-missing-data",
            description="Can handle missing data in time series",
            value_type="bool",
            category="behavior",
        ),
        "ignores-exogeneous-X": TagInfo(
            name="ignores-exogeneous-X",
            description="Ignores exogenous variables if passed",
            value_type="bool",
            category="behavior",
        ),
        
        # Transformation tags
        "transform-returns-same-time-index": TagInfo(
            name="transform-returns-same-time-index",
            description="Transform output has same time index as input",
            value_type="bool",
            category="transformation",
        ),
        "skip-inverse-transform": TagInfo(
            name="skip-inverse-transform",
            description="Inverse transform should be skipped in pipelines",
            value_type="bool",
            category="transformation",
        ),
        "univariate-only": TagInfo(
            name="univariate-only",
            description="Only works with univariate time series",
            value_type="bool",
            category="constraint",
        ),
        
        # Classification tags
        "capability:multivariate": TagInfo(
            name="capability:multivariate",
            description="Can handle multivariate time series",
            value_type="bool",
            category="capability",
        ),
        "capability:unequal_length": TagInfo(
            name="capability:unequal_length",
            description="Can handle unequal length time series",
            value_type="bool",
            category="capability",
        ),
        "capability:missing_values": TagInfo(
            name="capability:missing_values",
            description="Can handle missing values",
            value_type="bool",
            category="capability",
        ),
        
        # Python requirements
        "python_version": TagInfo(
            name="python_version",
            description="Required Python version constraint",
            value_type="str",
            category="requirements",
        ),
        "python_dependencies": TagInfo(
            name="python_dependencies",
            description="Required Python package dependencies",
            value_type="list",
            category="requirements",
        ),
    }
    
    def __init__(self):
        """Initialize the tag resolver."""
        self._registry = get_registry()
    
    def get_tag_info(self, tag_name: str) -> Optional[TagInfo]:
        """
        Get information about a specific tag.
        
        Args:
            tag_name: The tag name to look up
        
        Returns:
            TagInfo if known, None otherwise
        """
        return self.TAG_DEFINITIONS.get(tag_name)
    
    def get_tag_description(self, tag_name: str) -> str:
        """
        Get human-readable description of a tag.
        
        Args:
            tag_name: The tag name
        
        Returns:
            Description string, or generic message if unknown
        """
        info = self.get_tag_info(tag_name)
        if info:
            return info.description
        return f"Tag '{tag_name}' (no description available)"
    
    def get_tags_by_category(self, category: str) -> List[TagInfo]:
        """
        Get all known tags in a specific category.
        
        Args:
            category: Category name (e.g., "capability", "data", "behavior")
        
        Returns:
            List of TagInfo objects in that category
        """
        return [
            tag for tag in self.TAG_DEFINITIONS.values()
            if tag.category == category
        ]
    
    def get_all_categories(self) -> List[str]:
        """Get list of all tag categories."""
        categories = set(tag.category for tag in self.TAG_DEFINITIONS.values())
        return sorted(list(categories))
    
    def explain_tags(self, tags: Dict[str, Any]) -> Dict[str, str]:
        """
        Get human-readable explanations for a set of tags.
        
        Args:
            tags: Dictionary of tag names to values
        
        Returns:
            Dictionary of tag names to explanation strings
        """
        explanations = {}
        
        for tag_name, tag_value in tags.items():
            info = self.get_tag_info(tag_name)
            if info:
                if info.value_type == "bool":
                    status = "Yes" if tag_value else "No"
                    explanations[tag_name] = f"{info.description}: {status}"
                else:
                    explanations[tag_name] = f"{info.description}: {tag_value}"
            else:
                explanations[tag_name] = f"{tag_name}: {tag_value}"
        
        return explanations
    
    def filter_estimators_by_capability(
        self,
        task: Optional[str] = None,
        probabilistic: Optional[bool] = None,
        handles_missing: Optional[bool] = None,
        multivariate: Optional[bool] = None,
    ) -> List[EstimatorNode]:
        """
        Filter estimators by common capability requirements.
        
        This is a convenience method that translates human-friendly
        requirements into the appropriate tag queries.
        
        Args:
            task: Task type filter
            probabilistic: Require probabilistic predictions
            handles_missing: Require missing data handling
            multivariate: Require multivariate support
        
        Returns:
            List of matching EstimatorNode objects
        """
        tags = {}
        
        if probabilistic is not None:
            tags["capability:pred_int"] = probabilistic
        
        if handles_missing is not None:
            tags["handles-missing-data"] = handles_missing
        
        if multivariate is not None:
            tags["capability:multivariate"] = multivariate
        
        return self._registry.get_all_estimators(task=task, tags=tags if tags else None)
    
    def check_compatibility(
        self,
        estimator: EstimatorNode,
        requirements: Dict[str, Any],
    ) -> Dict[str, bool]:
        """
        Check if an estimator meets specific requirements.
        
        Args:
            estimator: The estimator to check
            requirements: Dictionary of required tag values
        
        Returns:
            Dictionary mapping requirement names to whether they are met
        """
        results = {}
        
        for req_name, req_value in requirements.items():
            actual_value = estimator.tags.get(req_name)
            results[req_name] = actual_value == req_value
        
        return results
    
    def suggest_similar_estimators(
        self,
        estimator: EstimatorNode,
        max_results: int = 5,
    ) -> List[EstimatorNode]:
        """
        Find estimators with similar capabilities.
        
        Args:
            estimator: Reference estimator
            max_results: Maximum number of results
        
        Returns:
            List of similar estimators (same task, similar tags)
        """
        # Get all estimators of the same task
        same_task = self._registry.get_all_estimators(task=estimator.task)
        
        # Score by tag similarity
        scored = []
        for candidate in same_task:
            if candidate.name == estimator.name:
                continue
            
            # Count matching tags
            score = 0
            for tag_name, tag_value in estimator.tags.items():
                if candidate.tags.get(tag_name) == tag_value:
                    score += 1
            
            scored.append((candidate, score))
        
        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return [s[0] for s in scored[:max_results]]


# Singleton instance
_resolver_instance: Optional[TagResolver] = None


def get_tag_resolver() -> TagResolver:
    """Get the singleton tag resolver instance."""
    global _resolver_instance
    if _resolver_instance is None:
        _resolver_instance = TagResolver()
    return _resolver_instance
