"""Obstacle editor server using FastAPI.

This server provides REST API endpoints for:
- Loading YAML configuration files
- Parsing OSM map files
- Saving obstacle data back to YAML files
"""

import logging
from pathlib import Path
from typing import Any

import yaml
from core.utils.config import get_nested_value
from core.utils.osm_parser import parse_osm_for_visualization
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Obstacle Editor API")

# Enable CORS for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (HTML, CSS, JS)
TOOLS_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=TOOLS_DIR), name="static")


class ObstacleData(BaseModel):
    """Obstacle configuration data."""

    obstacles: list[dict[str, Any]] = Field(..., description="List of obstacle configurations")


class ConfigResponse(BaseModel):
    """Response containing configuration data."""

    obstacles: list[dict[str, Any]] = Field(default_factory=list)
    map_path: str | None = Field(None, description="Resolved map path")


class MapResponse(BaseModel):
    """Response containing map visualization data."""

    map_lines: list[dict[str, Any]] = Field(default_factory=list)
    map_polygons: list[dict[str, Any]] = Field(default_factory=list)


def resolve_yaml_references(data: dict[str, Any], root_data: dict[str, Any]) -> dict[str, Any]:
    """Recursively resolve ${...} references in YAML data.

    Args:
        data: Current data node to process
        root_data: Root data for reference resolution

    Returns:
        Data with resolved references
    """
    if isinstance(data, dict):
        return {k: resolve_yaml_references(v, root_data) for k, v in data.items()}
    if isinstance(data, list):
        return [resolve_yaml_references(item, root_data) for item in data]
    if isinstance(data, str) and data.startswith("${") and data.endswith("}"):
        # Extract reference path (e.g., "system.map_path" from "${system.map_path}")
        ref_path = data[2:-1]
        try:
            return get_nested_value(root_data, ref_path)
        except (KeyError, ValueError):
            logger.warning("Could not resolve reference: %s", data)
            return data
    return data


@app.get("/")
async def root() -> FileResponse:
    """Serve the main HTML page."""
    return FileResponse(TOOLS_DIR / "index.html")


@app.get("/api/config", response_model=ConfigResponse)
async def get_config(yaml_path: str) -> ConfigResponse:
    """Load YAML configuration and extract obstacles and map path.

    Args:
        yaml_path: Path to the YAML configuration file (relative or absolute)

    Returns:
        Configuration data including obstacles and map path

    Raises:
        HTTPException: If file not found or parsing fails
    """
    try:
        # Get project root (e2e_aichallenge_playground)
        # This file is in experiment/tools/obstacle_editor_server.py
        project_root = Path(__file__).parent.parent.parent

        # Resolve path (support both relative and absolute paths)
        config_path = Path(yaml_path)
        if not config_path.is_absolute():
            config_path = project_root / yaml_path

        config_path = config_path.resolve()

        if not config_path.exists():
            raise HTTPException(status_code=404, detail=f"YAML file not found: {yaml_path}")

        with config_path.open() as f:
            config_data = yaml.safe_load(f)

        if not config_data:
            raise HTTPException(status_code=400, detail="Empty YAML file")

        # Load system config to resolve ${system.*} references
        # The module config references a system config via experiment config
        # We need to find the system config that references this module
        system_data = {}

        # Try to find system config in parent directory
        systems_dir = config_path.parent.parent / "systems"
        if systems_dir.exists():
            # Look for system configs that reference this module
            for system_file in systems_dir.glob("*.yaml"):
                try:
                    with system_file.open() as f:
                        sys_config = yaml.safe_load(f)
                        if sys_config and "system" in sys_config:
                            # Check if this system config references our module
                            module_ref = sys_config["system"].get("module", "")
                            if config_path.name in module_ref or str(config_path) in module_ref:
                                system_data = sys_config
                                logger.info(f"Found system config: {system_file}")
                                break
                except Exception as e:
                    logger.warning(f"Error reading {system_file}: {e}")
                    continue

        # Merge system data for reference resolution
        full_data = {**config_data, **system_data}

        # Resolve references
        resolved_data = resolve_yaml_references(config_data, full_data)

        # Extract obstacles from module.nodes[].params.obstacles
        obstacles = []
        try:
            nodes = get_nested_value(resolved_data, "module.nodes")
            for node in nodes:
                if "params" in node and "obstacles" in node["params"]:
                    obstacles = node["params"]["obstacles"]
                    break
        except (KeyError, ValueError, TypeError):
            logger.info("No obstacles found in configuration")

        # Extract map_path from module.nodes[].params.map_path
        map_path = None
        try:
            nodes = get_nested_value(resolved_data, "module.nodes")
            for node in nodes:
                if "params" in node and "map_path" in node["params"]:
                    map_path = node["params"]["map_path"]
                    break
        except (KeyError, ValueError, TypeError):
            logger.info("No map_path found in configuration")

        return ConfigResponse(obstacles=obstacles, map_path=map_path)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error loading config")
        raise HTTPException(status_code=500, detail=f"Error loading config: {e}") from e


@app.get("/api/map", response_model=MapResponse)
async def get_map(osm_path: str) -> MapResponse:
    """Load and parse OSM map file for visualization.

    Args:
        osm_path: Path to the OSM map file (relative or absolute)

    Returns:
        Map visualization data (lines and polygons)

    Raises:
        HTTPException: If file not found or parsing fails
    """
    try:
        # Get project root
        project_root = Path(__file__).parent.parent.parent

        # Resolve path (support both relative and absolute paths)
        map_file = Path(osm_path)
        if not map_file.is_absolute():
            map_file = project_root / osm_path

        map_file = map_file.resolve()

        if not map_file.exists():
            raise HTTPException(status_code=404, detail=f"OSM file not found: {osm_path}")

        map_lines, map_polygons = parse_osm_for_visualization(map_file)

        return MapResponse(map_lines=map_lines, map_polygons=map_polygons)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error loading map")
        raise HTTPException(status_code=500, detail=f"Error loading map: {e}") from e


@app.post("/api/obstacles")
async def save_obstacles(yaml_path: str, data: ObstacleData) -> dict[str, str]:
    """Save obstacles to YAML configuration file.

    Args:
        yaml_path: Path to the YAML configuration file (relative or absolute)
        data: Obstacle data to save

    Returns:
        Success message

    Raises:
        HTTPException: If file not found or saving fails
    """
    try:
        # Get project root
        project_root = Path(__file__).parent.parent.parent

        # Resolve path (support both relative and absolute paths)
        config_path = Path(yaml_path)
        if not config_path.is_absolute():
            config_path = project_root / yaml_path

        config_path = config_path.resolve()

        if not config_path.exists():
            raise HTTPException(status_code=404, detail=f"YAML file not found: {yaml_path}")

        # Read the original file content
        with config_path.open() as f:
            original_content = f.read()
            config_data = yaml.safe_load(original_content)

        if not config_data:
            raise HTTPException(status_code=400, detail="Empty YAML file")

        # Find the node with obstacles parameter
        try:
            nodes = get_nested_value(config_data, "module.nodes")
            obstacle_node_found = False
            for node in nodes:
                if "params" in node and "obstacles" in node["params"]:
                    obstacle_node_found = True
                    break

            if not obstacle_node_found:
                raise HTTPException(
                    status_code=400,
                    detail="No node with obstacles parameter found in YAML",
                )
        except (KeyError, ValueError, TypeError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid YAML structure: {e}") from e

        # Convert obstacles to YAML string with proper indentation
        obstacles_yaml = yaml.dump(
            data.obstacles, default_flow_style=False, sort_keys=False, allow_unicode=True
        )

        # Add proper indentation (10 spaces for obstacles list items)
        indented_obstacles = "\n".join(
            "      " + line if line.strip() else line for line in obstacles_yaml.splitlines()
        )

        # Use regex to replace only the obstacles section
        import re

        # Pattern to match the obstacles section
        # This matches from "obstacles:" to the next top-level key or end of params
        pattern = r"(obstacles:\s*\n)((?:[ \t]+.*\n)*?)(\s*(?:\w+:|$))"

        def replace_obstacles(match):
            return match.group(1) + indented_obstacles.rstrip() + "\n" + match.group(3)

        # Replace the obstacles section
        new_content = re.sub(pattern, replace_obstacles, original_content, count=1)

        # Write back to file
        with config_path.open("w") as f:
            f.write(new_content)

        logger.info("Saved obstacles to %s", config_path)
        return {"message": "Obstacles saved successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error saving obstacles")
        raise HTTPException(status_code=500, detail=f"Error saving obstacles: {e}") from e


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
