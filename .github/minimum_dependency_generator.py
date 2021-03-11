import argparse


def write_min_requirements(requirements_path, output_path):
    all_requirements = []
    min_requirements = []

    with open(requirements_path) as f:
        all_requirements = f.readlines()

    for req in all_requirements:
        min_req = req.replace(">=", "==")
        min_requirements.append(min_req)

    with open(output_path, "w") as f:
        f.writelines(min_requirements)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="reads a requirements file and outputs the minimized requirements")
    parser.add_argument('requirements_path', help='path for requirements to minimize')
    parser.add_argument('output_path', help='path to output minimized requirements')
    args = parser.parse_args()
    write_min_requirements(args.requirements_path, args.output_path)