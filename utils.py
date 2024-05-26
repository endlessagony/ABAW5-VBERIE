import sys, os, shutil, glob, random
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import math, timm, time
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from   torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from   torch.utils.data import Dataset, DataLoader, TensorDataset
from   torch.utils.data.dataloader import default_collate
from   torch.optim.optimizer import Optimizer
from   torch.nn import TransformerEncoderLayer
from   torch.nn import Parameter

import torchvision
import torchvision.transforms as transforms

import einops
from   einops import rearrange, repeat
from   einops.layers.torch import Rearrange

import librosa, fairseq
import soundfile as sf
from   dataclasses import dataclass
from   facenet_pytorch import MTCNN
import cv2
from   scipy.stats import pearsonr

import matplotlib.pyplot as plt

from   models.MModel import MModel


LOCAL_DIR = os.getcwd()
WAV_DIR = './wav_files'
FEATURE_DIR = './features'
FILE_STORE_DIR = './files'
PLOTS_DIR = './plots'


FEATURE_EXTRACTORS_DIR = './models_checkpoints/feature_extractors'
ABAW5_MODELS_DIR = './models_checkpoints/abaw5'
OPENFACE_DIR = r'C:\Users\nikita\OpenFace_2.2.0_win_x64'
EMOTION2VEC_MODEL_DIR = r'C:\Users\nikita\diploma_work_abaw\emotion2vec\upstream'


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
EMO2IDX = {'Adoration': 0, 'Amusement': 1, 'Anxiety': 2, 'Disgust': 3, 'Empathic-Pain': 4, 'Fear': 5, 'Surprise': 6}


@dataclass
class UserDirModule:
    user_dir: str


def calc_pearsons(predictions:np.array=None, ground_truth:np.array=None):
    '''
    Function calculates Pearson's Correlation Coefficient.
    
            Parameters:
                predictions (np.array): Model's forecasts;
                ground_truth (np.array): The fact.
    '''
    # Replace NaN values with 0
    predictions = np.nan_to_num(predictions, 1e-7)
    ground_truth = np.nan_to_num(ground_truth, 1e-7)
    
    pcc = pearsonr(predictions, ground_truth)
    return pcc[0]


def mean_pearsons(predictions:np.array=None, ground_truth:np.array=None, n_classes:int=7):
    '''
    Function calculates mean PCC between predictions and fact.
    
            Parameters:
                predictions (np.array): Model's forecasts;
                ground_truth (np.array): The fact;
                n_classes (int): number of classes.
    '''
    predictions, ground_truth = predictions.detach().cpu().numpy(), ground_truth.detach().cpu().numpy()
    predictions = np.nan_to_num(predictions, 1e-7)
    ground_truth = np.nan_to_num(ground_truth, 1e-7)
    
    class_wise_pcc = np.array([calc_pearsons(predictions[:, i], ground_truth[:, i]) for i in range(n_classes)])
    mean_pcc = np.mean(class_wise_pcc)
    
    return mean_pcc, class_wise_pcc


class FaceAligner(object):
    def __init__(self, desiredLeftEye: tuple = (0.30, 0.30)):
        self.desiredLeftEye = desiredLeftEye
        
    def align(self, image: np.ndarray = None, left_eye: np.ndarray = None, right_eye: np.ndarray = None, width: int = None, height: int = None):
        # compute the angle between the eye centroids
        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX))
        
        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        desiredRightEyeX = 1.0 - self.desiredLeftEye[0]
        
        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        dist = np.sqrt((dX ** 2) + (dY ** 2))
        desiredDist = (desiredRightEyeX - self.desiredLeftEye[0])
        desiredDist *= width
        scale = desiredDist / dist
        
        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        eyesCenter = (
            int((left_eye[0] + right_eye[0]) // 2),
            int((left_eye[1] + right_eye[1]) // 2)
        )
        # grab the rotation matrix for rotating and scaling the face
        M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
        # update the translation component of the matrix
        tX = width * 0.5
        tY = height * self.desiredLeftEye[1]

        M[0, 2] += (tX - eyesCenter[0])
        M[1, 2] += (tY - eyesCenter[1])
        
        # apply the affine transformation
        output = cv2.warpAffine(image, M, (width, width), flags=cv2.INTER_CUBIC)
        
        return output


def configure_feature_extraction_model_visual(
    feature_extractor_model_path: str = None, device: torch.device = None, return_initial: bool = True
) -> np.ndarray:
    '''
    Function configure feature-extraction model
    
            Parameters:
                extraction_model_path (str): path to feature-extraction model;
                device (torch.device): torch device (default=torch.cuda);
                return_initial (bool): Return the initial model or not
            Returns:
                features-extraction model
    '''
    feature_extractor_model = torch.load(feature_extractor_model_path)
    feature_extractor_model.classifier = torch.nn.Identity()
    feature_extractor_model.to(device)
    feature_extractor_model.eval()

    if return_initial:
        return feature_extractor_model, torch.load(feature_extractor_model_path)
    else:
        return feature_extractor_model
    

def configure_emotion2vec_feature_extractor_model(
    emotion2vec_model_dir: str = None, emotion2vec_model_checkpoint: str = None, device: torch.device = None                                      
):
    '''
    Function cofigureates emotion2vec as feature extractor model.
    
            Parameters:
                emotion2vec_model_dir (str): Path to the emotion2vec directory;
                emotion2vec_model_checkpoint (str): Path to the model checkpoint;
                device (torch.device): Available torch device;
    '''
    emotion2vec_model_path = UserDirModule(emotion2vec_model_dir)
    fairseq.utils.import_user_module(emotion2vec_model_path)
    emotion2vec_checkpoint = emotion2vec_model_checkpoint

    emotion2vec_model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([emotion2vec_checkpoint])
    emotion2vec_model = emotion2vec_model[0].eval()
    emotion2vec_model = emotion2vec_model.to(device)
    
    return emotion2vec_model, cfg, task


def detect_face_and_align(face_recongizer: MTCNN = None, frame = None, face_aligner: FaceAligner = None) -> np.ndarray:
    '''
    Function that detect face in the particular frame.
    
            Parameters:
                face_recognizer (MTCNN): MTCNN;
                frame: current frame of particular video;
                face_aligner (FaceAligner): Current FaceAligner algorithm
            Returns:
                face_image: numpy ndarrray;
    '''
    bounding_boxes, probs, landmarks = face_recongizer.detect(frame, landmarks=True)
    if None in probs: return False, False
    aligned_face_image, box = False, False
    if probs is None: return False, False
    else:
        bounding_boxes = bounding_boxes[probs > 0.9]
        landmarks = landmarks[probs > 0.9]
        if bounding_boxes == []: return False
        
        for bbox, landmark in zip(bounding_boxes, landmarks):
            box = bbox.astype(int)
            x1, y1, x2, y2 = box[:4]
            
            cropped_square_area = square_area(x1, y1, x2, y2)
            if cropped_square_area is None:
                continue
                
            left_eye_landmark, right_eye_landmark = landmark[0], landmark[1]
            width, height = abs(x1 - x2), abs(y1 - y2)
            aligned_face_image = face_aligner.align(frame, left_eye_landmark, right_eye_landmark, width, height)
        
        return aligned_face_image


def square_area(x1: float = None, y1: float = None, x2: float = None, y2: float = None):
    """
    Calculate the area of a square given the coordinates of two opposite vertices.
    
    Parameters:
        x1, y1: Coordinates of the first vertex.
        x2, y2: Coordinates of the second vertex.
        
    Returns:
        Area of the square.
    """
    # Ensure the points form a square (opposite sides are equal)
    side1 = abs(x2 - x1)
    side2 = abs(y2 - y1)
    
    area = side1 * side2
    return area


def get_prob(
    features:np.ndarray=None, classifier_weights:np.ndarray=None, classifier_bias:np.ndarray=None, 
    logits:bool=True
) -> np.ndarray:
    '''
    Function for the getting probabilities of the classes of the feature_extraction model.

            Parameters:
                features (np.ndarray): Current extracted features;
                classifier_weights (np.ndarray): Classifier weights;
                classifier_bias(np.ndarray): Classifier bias;
                logits (bool): Get the logits or not.
    '''
    xs = np.dot(features, np.transpose(classifier_weights)) + classifier_bias

    if logits:
        return xs
    else:
        e_x = np.exp(xs - np.max(xs, axis=1)[:,np.newaxis])
        return e_x / e_x.sum(axis=1)[:, None]


def extract_visual_features(file_path: str = None, stride: int = 5):
    """
    Function extracts features for visual-modality only.

            Parameters:
                file_path (str): Path to the particular video file;
                face_detector (MTCNN): Face detector on MTCNN-basis;
                face_aligner (FaceAligner): Face align algorithm.
    """
    vid, frames_count = cv2.VideoCapture(file_path), 0
    whole_visual_features = []
    current_frames, current_batch = [], 0

    efficientnetb0_feature_extractor, efficientnetb0_initial = configure_feature_extraction_model_visual(
        feature_extractor_model_path=f'{FEATURE_EXTRACTORS_DIR}/efficientnet_affectnet.pt', device=DEVICE, return_initial=True)
    mtcnn = MTCNN(keep_all=True, post_process=False, min_face_size=40, device=DEVICE, select_largest=True)
    face_aligner = FaceAligner()
    image_transforms = image_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((224, 224)),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    while (True):
        success, frame = vid.read()
        if not success: break
            
        if ((frames_count % stride == 0) or (frames_count == 0)):
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cropped_aligned_face = detect_face_and_align(face_recongizer=mtcnn, frame=rgb_frame, face_aligner=face_aligner)

            if ((isinstance(cropped_aligned_face, np.ndarray)) and (0 not in cropped_aligned_face.shape)):
                current_frames.append(image_transforms(cropped_aligned_face))

                if len(current_frames) >= 64: 
                    current_frames = torch.stack(current_frames, dim=0).to(DEVICE)
                    batch_features = efficientnetb0_feature_extractor(current_frames)
                    batch_features = batch_features.data.cpu().numpy()

                    current_scores = get_prob(batch_features, efficientnetb0_initial.classifier.weight.cpu().data.numpy(), efficientnetb0_initial.classifier.bias.cpu().data.numpy())
                    batch_features = np.concatenate((batch_features, current_scores), axis=1)
                    whole_visual_features.append(batch_features)

                    current_frames = []
                    current_batch += 1
        frames_count += 1
        
    if len(current_frames):
        current_frames = torch.stack(current_frames, dim=0).to(DEVICE)
        
        batch_features = efficientnetb0_feature_extractor(current_frames)
        batch_features = batch_features.data.cpu().numpy()

        current_scores = get_prob(batch_features, efficientnetb0_initial.classifier.weight.cpu().data.numpy(), efficientnetb0_initial.classifier.bias.cpu().data.numpy())
        batch_features = np.concatenate((batch_features, current_scores), axis=1)

        whole_visual_features.append(batch_features)
        current_frames = []
        
    del vid
    torch.cuda.empty_cache()

    return np.concatenate(whole_visual_features)


def pad_features(tensor: torch.Tensor = None, seq_length: int = 64, padding_values: tuple = (0, 0)):
    """
    Function for padding given tensor with given values

            Parameters:
                tensor (torch.Tensor): Given tensor for padding;
                seq_length (int): Given sequence length;
                padding_values (tuple): Given padding values.
    """
    tensor_seq_length, _ = tensor.shape
    if tensor_seq_length > seq_length: 
        tensor = tensor[:seq_length]
    else:
        tensor = np.pad(tensor, pad_width=((0, seq_length - tensor_seq_length), padding_values))
    
    return tensor


def extract_features(file_path: str = None, stride: int = 5) -> dict:
    """
    Function extracts features from the give file_path.

            Parameters:
                file_path (str): Path to the particular video file;
                stride (int): Value of number of passed through frames;
    """
    start_time = time.time()
    # First of all, we need to separate the visual and acoustic modality, so we extract .wav files from the .mp4
    assert file_path.endswith('.mp4'), f'Incorrect extension for the {file_path}, ".mp4" expected'
    wav_file_path = file_path.split("\\")[-1].replace(".mp4", ".wav")
    wav_file_path = f'{WAV_DIR}/{wav_file_path}'

    if not os.path.exists(wav_file_path):
        cmd = "ffmpeg -i " + file_path + " -ac 1 -ar 16000 -vn " + wav_file_path
        os.system(command=cmd)

    # Secondly, we extract features from the visual modality with the help of MTCNN and HSEmotion
    with torch.no_grad():
        HSEmotion_visual_features = extract_visual_features(file_path=file_path, stride=stride)
        HSEmotion_visual_features = pad_features(tensor=HSEmotion_visual_features)

    # and also OpenFace framework
    os.chdir(OPENFACE_DIR)
    temp_openface_feature_path = fr'{LOCAL_DIR}/open-face'
    cmd = f'FeatureExtraction.exe -f "{file_path}" -out_dir "{temp_openface_feature_path}" -2Dfp -aus -pose'
    os.system(cmd)
    OpenFace_output = pd.read_csv(f'{temp_openface_feature_path}/{wav_file_path.split("/")[-1].replace(".wav", ".csv")}')

    # REWORK, now i'll use the facial landmark toolkit by OpenFace, but in future it should be done by myself
    cmd = f'FaceLandmarkVidMulti.exe -f "{file_path}" -out_dir "{temp_openface_feature_path}"'
    os.system(cmd)
    os.chdir(LOCAL_DIR)

    # Here we extract features for the neural network and other points for visualization
    OpenFace_visual_features = pad_features(tensor=OpenFace_output.iloc[::5, 147:])
    OpenFace_landmarks = OpenFace_output.iloc[:, :] # CHECK

    # and noew we extract features from the audio-modality
    wav, sampling_rate = sf.read(wav_file_path)
    channel = sf.info(wav_file_path).channels
    assert sampling_rate == 16e3, f"Sample rate should be 16kHz, but got {sampling_rate} in file {wav_file_path}"
    assert channel == 1, f"Channel should be 1, but got {channel} in file {wav_file_path}"

    emotion2vec_model, cfg, task = configure_emotion2vec_feature_extractor_model(
        emotion2vec_model_dir=EMOTION2VEC_MODEL_DIR, emotion2vec_model_checkpoint=f'{FEATURE_EXTRACTORS_DIR}/emotion2vec_base.pt', device=DEVICE
    )

    with torch.no_grad():
        source = torch.from_numpy(wav).float().cuda()
        if task.cfg.normalize: source = F.layer_norm(source, source.shape)
        source = source.view(1, -1)

        audio_features = emotion2vec_model.extract_features(source, padding_mask=None)
        audio_features = audio_features['x'].squeeze(0).cpu().numpy()

        dataframe = pd.DataFrame(audio_features)
        audio_features = dataframe.groupby(dataframe.index // 10).mean().values
        audio_features = pad_features(tensor=audio_features)
        del dataframe
    
    features = {'visual': HSEmotion_visual_features, 'acoustic': audio_features, 'AUs': OpenFace_visual_features}

    end_time = time.time()
    print(f'Feature extraction time elapsed: {(end_time - start_time):.2f} sec.')
    return features, OpenFace_landmarks


def get_predictions(model: nn.Module = None, features: dict = None) -> dict:
    """
    Function gets predictions with the model.

            Parameters:
                model (nn.Module): Current Multi-Modal Model;
                features (dict): Extracted features;
    """
    start_time = time.time()
    visual_features, acoustic_features, AU_features = features['visual'], features['acoustic'], features['AUs']
    visual_features = torch.tensor(data=visual_features, dtype=torch.float).unsqueeze(0).to(DEVICE)
    acoustic_features = torch.tensor(data=acoustic_features, dtype=torch.float).unsqueeze(0).to(DEVICE)
    AU_features = torch.tensor(data=AU_features, dtype=torch.float).unsqueeze(0).to(DEVICE)

    model_outputs = model(visual_features, acoustic_features, AU_features)
    predictions = {emo: intensity.item() for emo, intensity in zip(list(EMO2IDX.keys()), model_outputs[0, :])}

    end_time = time.time()
    print(f'Feature extraction time elapsed: {(end_time - start_time):.2f} sec.')
    return predictions

def save_plot_results(estimated_intensities: dict = None, plot_name: str = None):
    estimated_intensities = {emotion: intensity for emotion, intensity in sorted(estimated_intensities.items(), key=lambda item: item[1], reverse=True)}

    figure = plt.figure(figsize=(12, 6))
    axis = figure.add_subplot(1, 1, 1)
    axis.bar(x=list(estimated_intensities.keys()), height=list(estimated_intensities.values()), color='#a0c4ff', alpha=0.9, linewidth=0.4, edgecolor='black')

    axis.grid(which='major', color='#666666', linestyle='-', linewidth=0.3, zorder=0)
    axis.grid(which='minor', color='#999999', linestyle='-', alpha=0.2, linewidth=0.4, zorder=0)
    axis.minorticks_on()
    axis.set_axisbelow(True)

    axis.set_ylim(bottom=0, top=1.05)
    axis.set_ylabel('intensity')

    plt.title(f'Result of emotional reaction intensity estimation', fontsize=12)
    for bar in plt.gca().patches:
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f'{bar.get_height():.2f}', ha='center', va='bottom', 
        )

    plt.savefig(f'{PLOTS_DIR}/{plot_name}.png')