import argparse
from collections import defaultdict

import requests
from packaging.requirements import Requirement
from packaging.specifiers import Specifier
from packaging.version import Version, parse


def versions(package_name):
    url = "https://pypi.org/pypi/%s/json" % (package_name,)
    r = requests.get(url)
    data = r.json()
    releases = []
    for rel in list(data["releases"].keys()):
        for avoid_tag in ['rc', 'b', 'post']:
            if rel not in avoid_tag:
                releases.append(rel)
    versions = [Version(x) for x in releases]
    versions.sort(reverse=False)
    return versions


def create_strict_min(package_version):
    version = None
    if isinstance(package_version, Version):
        version = package_version.public
    elif isinstance(package_version, str):
        version = package_version
    return Specifier('==' + version)


def verify_python_environment(requirement):
    package = Requirement(requirement)
    if not package.marker:
        # no python version specified in requirement
        return True
    elif package.marker and package.marker.evaluate():
        # evaluate --> evaluating the given marker against the current Python process environment
        return True
    return False


def remove_comment(requirement):
    if '#' in requirement:
        # remove everything after comment character
        requirement = requirement.split('#')[0]
    return requirement


def find_operator_version(package, operator):
    version = None
    for x in package.specifier:
        if x.operator == operator:
            version = x.version
            break
    return version


def find_min_requirement(requirement):
    requirement = remove_comment(requirement)
    if not verify_python_environment(requirement):
        return None
    if '>=' in requirement:
        # mininum version specified (ex - 'package >= 0.0.4')
        package = Requirement(requirement)
        version = find_operator_version(package, '>=')
        mininum = create_strict_min(version)
    elif '==' in requirement:
        # version strictly specified
        package = Requirement(requirement)
        version = find_operator_version(package, '==')
        mininum = create_strict_min(version)
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


def write_min_requirements(output_filepath, requirements_paths):
    requirements_to_specifier = defaultdict(list)
    min_requirements = []

    for path in requirements_paths:
        requirements = []
        with open(path) as f:
            requirements.extend(f.readlines())
        for req in requirements:
            package = Requirement(remove_comment(req))
            name = package.name
            if name in requirements_to_specifier:
                prev_req = Requirement(requirements_to_specifier[name])
                new_req = prev_req.specifier & package.specifier
                requirements_to_specifier[name] = name + str(new_req)
            else:
                requirements_to_specifier[name] = name + str(package.specifier)

    for req in list(requirements_to_specifier.values()):
        min_version = find_min_requirement(req)
        min_requirements.append(str(min_version) + "\n")

    with open(output_filepath, "w") as f:
        f.writelines(min_requirements)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="reads a requirements file and outputs the minimized requirements")
    parser.add_argument('output_filepath', help='path to output minimized requirements')
    parser.add_argument('--requirements_paths', nargs='+', help='path for requirements to minimize', required=True)
    args = parser.parse_args()
    write_min_requirements(args.output_filepath, args.requirements_paths)
