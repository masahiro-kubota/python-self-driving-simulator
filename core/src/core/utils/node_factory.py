import importlib
import types
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin

from pydantic import BaseModel

from core.interfaces.node import Node
from core.utils.paths import get_project_root

T = TypeVar("T")


class NodeFactory:
    """Factory for creating Node instances dynamically."""

    def __init__(self) -> None:
        self.workspace_root = get_project_root()

    def create(
        self,
        node_type: str,
        rate_hz: float,
        params: dict[str, Any],
        priority: int,
    ) -> Node:
        """Create a Node instance dynamically.

        Args:
            node_type: Entry point name (e.g., "PurePursuit") or Class path
            rate_hz: Execution frequency in Hz
            params: Node configuration parameters
            priority: Execution priority (lower values execute first, default: 100)

        Returns:
            Instantiated Node
        """
        # 1. Resolve Node Class
        node_class = self._resolve_node_class(node_type)

        # 2. Resolve Configuration Class (T) from Node[T]
        config_class = self._resolve_config_class(node_class)

        # 3. Prepare Parameters (Path resolution based on Config type)
        resolved_params = self._resolve_paths(params, config_class)

        # 4. Instantiate Node
        return node_class.from_dict(
            rate_hz=rate_hz,
            config_class=config_class,
            config_dict=resolved_params,
            priority=priority,
        )

    def _resolve_node_class(self, node_type: str) -> type[Node]:
        """Import and return the Node class using Entry Points or dynamic import."""
        # 1. Try Entry Points
        eps = importlib.metadata.entry_points(group="e2e_aichallenge.node")
        # In Python 3.10+, entry_points returns a SelectableGroups, we can access by name or iteration
        # For compatibility, iterate
        for ep in eps:
            if ep.name == node_type:
                return ep.load()

        # 2. Fallback: Dynamic Import (if it looks like a module path)
        if "." in node_type:
            try:
                module_name, class_name = node_type.rsplit(".", 1)
                module = importlib.import_module(module_name)
                cls = getattr(module, class_name)
                if not issubclass(cls, Node):
                    raise TypeError(f"Class {cls} is not a subclass of Node")
                return cls
            except (ValueError, ImportError, AttributeError):
                pass  # Fall through to error raising

        raise ValueError(
            f"Node type '{node_type}' not found in entry points and valid import failed."
        )

    def _resolve_config_class(self, node_class: type[Node]) -> type:
        """Extract the Config class from Node[Config] generic type."""
        config_class = None
        if hasattr(node_class, "__orig_bases__"):
            for base in node_class.__orig_bases__:
                if (
                    hasattr(base, "__origin__")
                    and base.__origin__ is Node
                    and hasattr(base, "__args__")
                    and base.__args__
                ):
                    config_class = base.__args__[0]
                    break

        if config_class is None:
            raise ValueError(f"Could not determine config class for {node_class}")

        return config_class

    def _resolve_paths(
        self, params: dict[str, Any], config_class: type[BaseModel]
    ) -> dict[str, Any]:
        """Resolve path parameters relative to workspace root based on Config type definition."""
        resolved = params.copy()

        for name, field in config_class.model_fields.items():
            if name in resolved:
                value = resolved[name]
                if isinstance(value, str) and self._is_path_type(field.annotation):
                    resolved[name] = str(self.workspace_root / value)

        return resolved

    def _is_path_type(self, annotation: Any) -> bool:
        """Check if the type annotation implies a Path."""
        if annotation is Path:
            return True

        # Handle Optional[Path] or Path | None (Union)
        origin = get_origin(annotation)
        if origin is types.UnionType or str(origin) == "typing.Union":
            # UnionType is for A | B syntax in 3.10+
            args = get_args(annotation)
            for arg in args:
                if self._is_path_type(arg):
                    return True

        # Check sub-types if needed (e.g. FilePath from pydantic)
        # Pydantic types might not be exactly Path but behave like it?
        # For now, we explicitly use pathlib.Path in configs.
        # If annotation is a class and inherits Path?
        try:
            if isinstance(annotation, type) and issubclass(annotation, Path):
                return True
        except TypeError:
            pass

        return False
