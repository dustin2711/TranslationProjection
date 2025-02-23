import pkg_resources
import requests
from packaging.specifiers import SpecifierSet
from helper import *


def get_python_compatibility(package_name: str, python_version: str) -> str:
    url = f"https://pypi.org/pypi/{package_name}/json"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        releases = data.get("releases", {})

        for version, release_info in releases.items():
            for file_info in release_info:
                requires_python = file_info.get("requires_python")
                if requires_python:  # Skip None values
                    spec = SpecifierSet(requires_python)
                    if spec.contains(python_version):
                        return "Compatible"
        return "Incompatible or Unknown"
    except Exception as e:
        return f"Error: {str(e)}"


def get_incompatabiles_packages(python_version: str) -> tuple[list[str], list[str]]:
    packages = []

    for dist in pkg_resources.working_set:
        package_name = dist.project_name
        compatible = get_python_compatibility(package_name, python_version)
        if compatible != "Compatible":
            packages.append(package_name)

    return packages


# # Check for Python 3.7 compatibility
# for version in ["3.7", "3.8", "3.9", "3.10", "3.11", "3.12"]:
#     incompatible = get_incompatabiles_packages(version)
#     print(f"Incompatible for Python {version}\n{join_string(incompatible)}\n\n")


