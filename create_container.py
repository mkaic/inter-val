import subprocess
import argparse
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument("--image_name", "-n", default="basic_training")
parser.add_argument("--image_tag", "-t", default=None)
parser.add_argument("--container_name", "-c", default="basic_training")

# parse args, get rid of the "args." prefix
args = parser.parse_args()
image_name = args.image_name
tag = args.image_tag
container_name = args.container_name

# set default container name to image name, allow for user to override with -c
container_name = container_name if container_name is not None else image_name

image = (
    f"docker_images/{image_name}:{tag}"
    if tag is not None
    else f"docker_images/{image_name}"
)

try:
    subprocess.check_output(["docker", "image", "inspect", image])
    image_exists = True
except subprocess.CalledProcessError:
    image_exists = False

if image_exists:
    print(f"The image {image} already exists, creating container...")
else:
    # if the image doesn't already exist, build it
    context_dir = pathlib.Path(__file__).parent.resolve()
    dockerfile_path = context_dir / "Dockerfile"
    print("Context dir:", context_dir)
    print("Dockerfile path:", dockerfile_path)
    print(f"Building image {image}...")
    output = subprocess.run(
        [
            "docker",
            "build",
            "-f",
            str(dockerfile_path),
            "-t",
            image,
            str(context_dir),
        ],
        capture_output=True,
    )

    if output.stdout is not None:
        print("Output from docker build:")
        print(output.stdout.decode("utf-8"))

    # if an error has ocurred and the error message is not empty, print it
    if output.stderr is not None and output.stderr.decode("utf-8") != "":
        print("Error from docker build:")
        print(output.stderr.decode("utf-8"))

    print("Creating container...")

output = subprocess.run(
    [
        "docker",
        "run",
        "--name",
        container_name,
        "--gpus",
        "all",
        "--detach",
        "-v",
        "/workspace:/workspace",
    ]
)

if output.stdout is not None:
    print("Output from docker run:")
    print(output.stdout.decode("utf-8"))
if output.stderr is not None:
    print("Error from docker run:")
    print(output.stderr.decode("utf-8"))

subprocess.run(f"docker ps | grep -w {container_name}", shell=True)
