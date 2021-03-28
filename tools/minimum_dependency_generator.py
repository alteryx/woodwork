import argparse
import json
import re

import requests
from packaging.requirements import Requirement
from packaging.specifiers import Specifier
from packaging.version import Version, parse


def versions(package_name):
    url = "https://pypi.org/pypi/%s/json" % (package_name,)
    r = requests.get(url)
    data = r.json()
    releases = data["releases"].keys()
    releases = [x for x in list(releases) if 'rc' not in x and 'b' not in x and 'post' not in x]
    versions = list(releases)
    versions = [Version(x) for x in versions]
    versions.sort(reverse=False)
    return versions


def create_strict_min(package_version):
    version = None
    if isinstance(package_version, Version):
        version = package_version.public
    elif isinstance(package_version, Specifier):
        version = package_version.version
    return Specifier('==' + version)


def verify_python_environment(requirement):
    package = Requirement(requirement)
    if package.marker and package.marker.evaluate():
        # evaluate --> Return the boolean from evaluating the given marker against the current Python process environment
        return True
    return False


def find_min_requirement(requirement):
    if '#' in requirement:
        # remove everything after comment character
        requirement = requirement.split('#')[0]
    if not verify_python_environment(requirement):
        return None
    if '>=' in requirement:
        # mininum version specified (ex - 'package <= 0.0.4')
        requirement = re.sub(r',<\d+\.\d+\.\d+$', '', requirement)
        package = Requirement(requirement)
        mininum = [x for x in package.specifier][0]
        mininum = create_strict_min(mininum)
    elif '==' in requirement:
        # version strictly specified
        package = Requirement(requirement)
        mininum = [x for x in package.specifier][0]
        mininum = create_strict_min(mininum)
    elif '<' in requirement:
        # mininum version not specified (ex - 'package < 0.0.4')
        package = Requirement(requirement)
        version_not_specified = [x for x in package.specifier][0]
        exclusive_upper_bound_version = parse(version_not_specified.version)
        all_versions = versions(package.name)
        valid_versions = []
        for version in all_versions:
            if version < exclusive_upper_bound_version:
                valid_versions.append(version)
        mininum = min(valid_versions)
        mininum = create_strict_min(mininum)
    else:
        # version not specified (ex - 'package')
        package = Requirement(requirement)
        all_versions = versions(package.name)
        mininum = min(all_versions)
        mininum = create_strict_min(mininum)
    if len(package.extras) > 0:
        return Requirement(package.name + "[" + package.extras.pop() + "]" + str(mininum))
    return Requirement(package.name + str(mininum))


def write_min_requirements(requirements_path, output_path):
    all_requirements = []
    min_requirements = []

    with open(requirements_path) as f:
        all_requirements = f.readlines()

    for req in all_requirements:
        min_version = find_min_requirement(req)
        min_requirements.append(str(min_version) + "\n")

    with open(output_path, "w") as f:
        f.writelines(min_requirements)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="reads a requirements file and outputs the minimized requirements")
    parser.add_argument('requirements_path', help='path for requirements to minimize')
    parser.add_argument('output_path', help='path to output minimized requirements')
    args = parser.parse_args()
    write_min_requirements(args.requirements_path, args.output_path)
