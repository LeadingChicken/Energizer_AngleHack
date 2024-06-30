import os
import subprocess


def analyze_image_with_llava(model_path, image_path, load_4bit=False):
    command = [
        "python", "-m", "llava.serve.cli",
        "--model-path", model_path,
        "--image-file", image_path
    ]

    if load_4bit:
        command.append("--load-4bit")

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print("Error running the model:")
        print(result.stderr)
        return None

    return result.stdout


if __name__ == "__main__":
    model_path = "/home/lucy/Documents/ai-ml/model/llava/llava-v1.6-34b"
    image_path = "/home/lucy/Documents/ai-ml/code/angelhack24heineiken/heineiken_raw_imgs/66502739_1705745936616.jpg"

    output = analyze_image_with_llava(model_path, image_path, load_4bit=True)

    if output:
        print("Model Output:")
        print(output)
