import gradio as gr
from PIL import Image
import torch
import utils

def pipe(audio):
    scaler = utils.load_scaler("scaler_top_1000.pkl")
    x = utils.get_x(audio, scaler)
    utils.save_fig(x, 'img', 'Your spectrogram')
    x = torch.from_numpy(x)
    x = torch.unsqueeze(x, 0)
    model = torch.load('entire_model')
    out = model.forward(x)
    prd_labels = utils.out2labels("id2token_top_1000.json", out)

    
    return 'img.png', str(prd_labels)



demo = gr.Interface(pipe, gr.Audio(min_length=1, max_length=10, type='filepath'), outputs= [gr.Image(type='filepath'), "text"], examples = "example_sounds/")
    
if __name__ == '__main__':
    demo.launch(share=True)