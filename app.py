import gradio as gr
import os, shutil, glob

from utils import *
from models.MModel import MModel


mmodel = MModel().to(DEVICE)
mmodel.load_state_dict(torch.load(f'{ABAW5_MODELS_DIR}/multiModal_0.3975.pt'))
mmodel.eval()


FILE_STORE_DIR = r'./flagged/file_objects'
if not os.path.exists(WAV_DIR):
    os.mkdir(WAV_DIR)
    os.mkdir(FEATURE_DIR)
    os.mkdir(PLOTS_DIR)


def estimate_emotional_reaction_intensity(file_path = None):
    features, OpenFace_landmarks = extract_features(file_path=file_path) # REWORK

    # postprocess extracted features
    predictions = get_predictions(model=mmodel, features=features)
    save_plot_results(estimated_intensities=predictions, plot_name=file_path.split('\\')[-1].replace('.mp4', ''))

    # get emotional reaction intesities
    image_name = file_path.split("\\")[-1].replace(".mp4", "")
    image = cv2.imread(f'{PLOTS_DIR}/{image_name}.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # get postprocessed video via OpenFace
    video = f'./open-face/{image_name}.avi'

    return image, video


input_video = gr.PlayableVideo(label='input video')
output_graph = gr.Image(label='ERI estimation result')
output_video = gr.PlayableVideo(label='output video') 

demo = gr.Interface(
    fn=estimate_emotional_reaction_intensity,
    inputs=input_video,
    outputs=[output_graph, output_video],
)

demo.launch()