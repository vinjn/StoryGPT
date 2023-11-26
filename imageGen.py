import os
import io
import warnings
from PIL import Image
import webuiapi


# This function assumes that an environment variable 'STABILITY_KEY' has already been set.
def stability_setup():
    api = webuiapi.WebUIApi(host="127.0.0.1", sampler="DPM++ 2M Karras", port=7860)
    return api


def generate_image(api, img_prompt, dims: tuple):
    answers = api.txt2img(
        prompt=img_prompt,
        negative_prompt="EasyNegative",
        cfg_scale=8.0,
        # seed=1003,
        width=dims[0],  # Generation width, defaults to 256 if not included.
        height=dims[1],  # Generation height, defaults to 256 if not included.
    )

    print(answers.parameters)

    # Set up our warning to print to the console if the adult content classifier is tripped.
    # If adult content classifier is not tripped, save generated images.
    return answers.image
