import argparse
import json
import re
import subprocess
import requirements

def get_pipgrip_and_req_parser() -> list:
    final_list = []
    for package in ['pipgrip', 'requirements-parser']:
        args = ['pipgrip', '--tree-json', package, '--max-depth', '3']
        subprocess.run(args, capture_output=True)
        sub_output = subprocess.check_output(args, encoding='utf-8')
        final_list.extend(dict_to_list(list(json.loads(sub_output).values())[0]))
    return final_list

def deps_to_json(package: str, filename: str, write: bool) -> json:
    # writes the json of the min dependencies in the working directory
    print("Getting tree json...")
    args = ['pipgrip', '--tree-json', package, '--max-depth', '3']
    subprocess.run(args, capture_output=True)
    sub_output = subprocess.check_output(args, encoding='utf-8')
    # only write if we want to write
    # primarily used for local testing
    if write:
        with open(filename, "w+") as f:
            f.write(sub_output)
        f.close()
    print("Finished grabbing tree json!")
    return sub_output

def read_deps(path: str) -> json:
    # reads the dependencies as json file from the json path provided
    with open(path, "r+") as f:
        read_json = json.load(f)
    return read_json

def dict_to_list(json_dict: dict) -> list:
    # returns the json as a list of values
    # values will be of format ('<package><equality><version>`)
    # where equality and version are optional (depending on what pipgrip returns)
    list_to_return = []
    for package, sub_deps in json_dict.items():
        list_to_return.append(package)
        list_to_return.extend(dict_to_list(sub_deps))
    return list_to_return

def get_all_package_versions(package_name: str) -> list:
    # grabs all package versions using pip index
    args = ['pip', 'index', 'versions', package_name]
    version_substring = "Available versions: "

    subprocess.run(args, capture_output=True)
    try:
        sub_output = subprocess.check_output(args).decode('utf-8')
    except subprocess.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
    # this gets the string versions available from the `pip index` command
    string_versions = [output_split[len(version_substring):].split(", ") 
                        for output_split in sub_output.split("\n") if version_substring in output_split][0]
    # we filter out versions that might have letters in it (ie 22.2.post1, etc) to only compare versions that we can convert to float
    float_versions = list(filter(lambda version_number: not re.search(r'[A-Za-z]', version_number), 
                          string_versions))
    # sorting for ease later on
    return sorted(float_versions)

def get_version_logic(vers_equality: list) -> list:
    # handles grabbing all the logic for version control
    # expects list of tuples for [(equality, version)]
    version_logic = []
    for ve in vers_equality:
        value = ve[1]
        # replace the '~=' equality with '>=' for simplicity
        # since we are finding the minimum requirement that satisfies the equality
        # this logic should be about the same
        equality = ve[0].replace("~", ">")
        version_logic.append("{} '{}'".format(equality, value))
    return version_logic

def get_min_version_by_logic(available_versions: list, version_logic: list) -> list:
    # the list of available versions is sorted in descending order, which is how `pip index` returns
    if not len(version_logic):
        return available_versions[0]
    for index in range(len(available_versions)):
        # grab the 0 index to get the float value of the version
        # creates the expression that we evaluate to determine if the version satisfies the logic
        eval_string = "'{}'".format(available_versions[index]) + f" and '{available_versions[index]}'".join(version_logic)
        if eval(eval_string):
            return available_versions[index]

def add_versions_to_dict(version_dict: dict, package: str, version: tuple):
    # handles the logic for adding a version to the version dictionary we track
    if package in version_dict:
        if version > version_dict[package]:
            # minimal dependency for this package is greater than another, so we take the greater
            version_dict.update({package: version})
    else:
        version_dict.update({package: version})

def get_min_version_string(version_dict: dict, delim: str, write: bool, output_name: str) -> str:
    # returns the min versions of all packages as a string delimited by the delim value.
    return_string = ""
    for package, version in version_dict.items():
        return_string += f"{package}=={version}"
        return_string += delim
    if (write):
        with open(output_name, "w+") as f:
            f.write(return_string)
        f.close()
    return return_string

def install_min_deps():
    # handles the install of all min dependencies
    print("Installing the minimum requirements generated")
    process = ['pip', 'install']
    process.extend(min_reqs.split(delim)[:-1])
    process = [x for x in process if ('wheel' not in x and 'pip==' not in x)]
    subprocess.run(process, capture_output=False)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get min dependencies of min dependencies")
    parser.add_argument('--package', default='woodwork', required=False)
    parser.add_argument('--json-filename', default='mindep.json', required=False)
    parser.add_argument('--write-txt', default=False, required=False)
    parser.add_argument('--output-name', default='min_min_dep.txt', required=False)
    parser.add_argument('--delimiter', default='\n', required=False)
    parser.add_argument('--install', default=True, required=False)
    parser.add_argument('--path', default='', required=False)
    args = parser.parse_args()

    # get the arguments from the parser
    package = args.package
    json_name = args.json_filename
    delim = args.delimiter
    write = bool(args.write_txt=='True')
    output_name = args.output_name
    install = bool(args.install=='True')
    path = args.path

    # grabs the dependencies of the current package and sets it as a list
    json_packages = deps_to_json(package, json_name, write)
    package_deps = list(json.loads(json_packages).values())[0]
    deps_list = dict_to_list(package_deps)

    with open('woodwork/tests/requirement_files/minimum_test_requirements.txt', 'r') as f:
        testing_deps = f.read()
    testing_deps = testing_deps.split("\n")
    # the dependencies of pipgrip and requirements-parser
    # we want to ensure the package dependencies are supported for these
    pipgrip_and_req_deps = get_pipgrip_and_req_parser()
    deps_list.extend(pipgrip_and_req_deps)
    deps_list.extend(testing_deps)
    version_dict = {}

    # iterate through each dependency to determine the minimum version allowed
    for package_value in deps_list:
        req = tuple(requirements.parse(package_value))
        try:
            package_name = req[0].name
            version_logic = get_version_logic(req[0].specs)
        except IndexError:
            print(package_value)
            continue
        all_versions = get_all_package_versions(package_name)
        min_version = get_min_version_by_logic(all_versions, version_logic)
        add_versions_to_dict(version_dict, package_name, min_version)

    # min_reqs will represent the string version of all min requirements for the package
    # this does not include testing requirements, which we will need to install prior
    min_reqs = get_min_version_string(version_dict, delim, write, output_name)
    if install:
        install_min_deps()
