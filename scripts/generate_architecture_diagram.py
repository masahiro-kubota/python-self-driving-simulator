#!/usr/bin/env python3
import pathlib
import tomllib
from typing import NamedTuple

PROJECT_ROOT = pathlib.Path(__file__).parent.parent


class ProjectInfo(NamedTuple):
    name: str
    description: str
    path: pathlib.Path
    dependencies: list[str]
    group: str  # e.g., "core", "ad_components", "simulators"


def get_project_group(path: pathlib.Path, root: pathlib.Path) -> str:
    rel_path = path.relative_to(root)
    parts = rel_path.parts
    if parts[0] in ["core", "dashboard", "experiment"]:
        return parts[0]
    if parts[0] in ["ad_components", "simulators"]:
        # return top level dir for these as they have sub categories
        # Actually proper grouping might be better.
        # ad_components/core -> ad_components
        # simulators/core -> simulators
        return parts[0]
    return "other"


def find_projects(root: pathlib.Path) -> dict[str, ProjectInfo]:
    projects = {}

    # Exclude root pyproject.toml
    root_pyproject = root / "pyproject.toml"

    for path in root.rglob("pyproject.toml"):
        if path == root_pyproject:
            continue

        # Skip hidden directories and build artifacts
        if any(part.startswith(".") for part in path.parts) or "build" in path.parts:
            continue

        try:
            with open(path, "rb") as f:
                data = tomllib.load(f)

            project_table = data.get("project", {})
            name = project_table.get("name")
            if not name:
                continue

            desc = project_table.get("description", "")
            deps = project_table.get("dependencies", [])
            # Extract package names from dependencies (handle version specifiers)
            clean_deps = []
            for dep in deps:
                # Basic parsing: take chars until first non-name char (unsafe but ok for now)
                # Or just split by symbols. simpler:
                dep_name = dep.split(">")[0].split("<")[0].split("=")[0].strip()
                clean_deps.append(dep_name)

            projects[name] = ProjectInfo(
                name=name,
                description=desc,
                path=path,
                dependencies=clean_deps,
                group=get_project_group(path.parent, root),
            )
        except Exception as e:
            print(f"Error parsing {path}: {e}")

    return projects


def generate_mermaid(projects: dict[str, ProjectInfo]) -> str:
    lines = ["graph TD"]

    # Define groups
    groups = {}
    for p in projects.values():
        groups.setdefault(p.group, []).append(p)

    project_names = set(projects.keys())

    # Styling definitions (matched to README style)
    lines.append("    %% Styling")
    lines.append("    classDef core fill:#f9f,stroke:#333,stroke-width:2px;")
    lines.append("    classDef base fill:#fbb,stroke:#333,stroke-width:2px;")
    lines.append("    classDef impl fill:#bbf,stroke:#333,stroke-width:2px;")
    lines.append("    classDef app fill:#bfb,stroke:#333,stroke-width:2px;")
    lines.append("    classDef default fill:#fff,stroke:#333,stroke-width:1px;")

    # Nodes
    for group_name, members in groups.items():
        lines.append(f"    subgraph {group_name}")
        for p in members:
            # Escape description for mermaid
            desc = p.description.replace('"', "'")
            # formatting: Name[name<br/>description]
            short_desc = (desc[:20] + "..") if len(desc) > 20 else desc
            if short_desc:
                label = f"{p.name}<br/>{short_desc}"
            else:
                label = p.name

            node_id = p.name.replace("-", "_").replace(".", "_")
            lines.append(f'        {node_id}["{label}"]')

            # Apply classes based on heuristic or name
            if p.name == "core":
                lines.append(f"        class {node_id} core;")
            elif "core" in p.name:
                lines.append(f"        class {node_id} base;")
            elif "runner" in p.name:
                lines.append(f"        class {node_id} app;")
            else:
                lines.append(f"        class {node_id} impl;")

        lines.append("    end")

    # Edges
    lines.append("    %% Dependencies")
    for p in projects.values():
        p_node_id = p.name.replace("-", "_").replace(".", "_")
        for dep in p.dependencies:
            # canonicalize name (replace underscores with dashes etc if needed, but here exact match)
            # check if dependency is internal project
            target = None
            if dep in project_names:
                target = dep
            elif dep.replace("-", "_") in project_names:
                target = dep.replace("-", "_")
            elif dep.replace("_", "-") in project_names:
                target = dep.replace("_", "-")

            if target:
                t_node_id = target.replace("-", "_").replace(".", "_")
                if p_node_id != t_node_id:
                    lines.append(f"    {p_node_id} --> {t_node_id}")

    return "\n".join(lines)


def update_readme(mermaid_content: str):
    readme_path = PROJECT_ROOT / "README.md"
    content = readme_path.read_text()

    start_marker = "<!-- ARCHITECTURE_DIAGRAM_START -->"
    end_marker = "<!-- ARCHITECTURE_DIAGRAM_END -->"

    # Check if markers exist, if not, try to replace existing mermaid block
    if start_marker not in content:
        # Fallback: Find existing mermaid block
        if "```mermaid" in content:
            # We will rewrite the file to include markers around the mermaid block
            # But simpler to just replace the whole mermaid block with markers + generated content
            import re

            pattern = re.compile(r"```mermaid\n(.*?)```", re.DOTALL)
            match = pattern.search(content)
            if match:
                new_block = f"{start_marker}\n```mermaid\n{mermaid_content}\n```\n{end_marker}"
                new_content = content[: match.start()] + new_block + content[match.end() :]
                readme_path.write_text(new_content)
                print("Updated README.md with new markers and diagram.")
                return

        print("Could not find insertion point in README.md")
        return

    # Replace content between markers
    start_idx = content.find(start_marker) + len(start_marker)
    end_idx = content.find(end_marker)

    new_content = (
        content[:start_idx] + f"\n```mermaid\n{mermaid_content}\n```\n" + content[end_idx:]
    )
    readme_path.write_text(new_content)
    print("Updated README.md diagram.")


def main():
    projects = find_projects(PROJECT_ROOT)
    mermaid_graph = generate_mermaid(projects)
    update_readme(mermaid_graph)


if __name__ == "__main__":
    main()
