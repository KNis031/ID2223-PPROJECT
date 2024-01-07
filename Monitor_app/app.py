import gradio as gr
from PIL import Image
import hopsworks
import json


N = 5

project = hopsworks.login()

dataset_api = project.get_dataset_api()

resource_path = "Resources/Freesound/"

dataset_api.download(resource_path + "prd_lbl_tokens.json")
for i in range(0,N):
    dataset_api.download(resource_path + f"sound_{i}.ogg")
    dataset_api.download(resource_path + f"img_{i}.png")

with open("prd_lbl_tokens.json", "r") as jsf:
    prd_lbl_tokens_dict = json.load(jsf)

keys = list(prd_lbl_tokens_dict.keys())
keys.sort()
ids = keys[-N:]
tags = []
for id in ids:
    tags.append(prd_lbl_tokens_dict[id])

print(ids)
print(tags)

with gr.Blocks() as demo:
    with gr.Row():
      with gr.Column():
          gr.Label("Spectrogram")
          gr.Image(f"img_0.png")
          gr.Image(f"img_1.png")
          gr.Image(f"img_2.png")
          gr.Image(f"img_3.png")
          gr.Image(f"img_4.png")
      with gr.Column():
          gr.Label("Sound")
          gr.Audio(f"sound_0.ogg")
          gr.Audio(f"sound_1.ogg")
          gr.Audio(f"sound_2.ogg")
          gr.Audio(f"sound_3.ogg")
          gr.Audio(f"sound_4.ogg")
      with gr.Column():
          gr.Label("Tags")
          gr.Label(str(tags[0]))
          gr.Label(str(tags[1]))
          gr.Label(str(tags[2]))
          gr.Label(str(tags[3]))
          gr.Label(str(tags[4]))

demo.launch(share=True)