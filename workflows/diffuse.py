"""A simple Flyte example."""

from dataclasses_json import dataclass_json
from dataclasses import dataclass
from diffusers import StableDiffusionPipeline
from flytekit import task, workflow, Resources, map_task
from typing import List
import base64
from io import BytesIO


@dataclass_json
@dataclass
class InferArgs:

    model: StableDiffusionPipeline
    payload: List[str]


@task(requests=Resources(mem="10Gi", gpu="1"))
def forward_batch(infer: InferArgs) -> List[bytes]:
    model = infer.model.to("cuda")
    output = model(infer.payload)
    images = output.images[0]
    buffers = []
    for image in images:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue())
        buffers.append(img_str)
    return buffers

@task(disable_deck=False)
def render_deck(img_bytes: List[List[bytes]]):
    html_images = []
    for img_batch in img_bytes:
        for img_byte in img_batch:
            html_images.append(f"<img style='display:block; width:1000px;height:1000px;' id='base64image' src='data:image/jpeg;base64, {img_byte.decode('utf-8')}' />")
    print( f"""<!DOCTYPE html>
                    <html>
                      <head>
                        <title>Display Image</title>
                      </head>
                      <body>
                      {" ".join(html_images)}
                      </body>
                    </html>""")

@task(cache=True, cache_version="1.1")
def prepare_inference_args(prompts: List[List[str]], model: StableDiffusionPipeline) -> List[InferArgs]:
    return [InferArgs(model=model, payload=prompt) for prompt in prompts]

@task(requests=Resources(mem="10Gi", storage="10Gi"), cache=True, cache_version="1.1")
def load_model() -> StableDiffusionPipeline:
    """A task the counts the length of a greeting."""
    print("about to load model")
    model = StableDiffusionPipeline.from_pretrained("OFA-Sys/small-stable-diffusion-v0" )
    print("loaded model")
    return model

@task
def generate_sentences(prompt: str, batch_size: int) -> List[List[str]]:
    sentence_splits = prompt.split(".")
    sentence_batches = []
    for i in range(len(sentence_splits) // batch_size):
        sentence_batches.append(sentence_splits[i * batch_size: (i + 1) * batch_size])
    return sentence_batches

@task
def start_process() -> None:
    print("Starting process")

@workflow
def diffuse(prompt: str):
    start_process()
    model = load_model()
    sentences = generate_sentences(prompt=prompt, batch_size=10)
    input_args = prepare_inference_args(prompts=sentences, model=model)
    model_output = map_task(forward_batch, concurrency=1)(infer=input_args)
    render_deck(img_bytes=model_output)

