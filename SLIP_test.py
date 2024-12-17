import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import DataLoader
import clip
from PIL import Image
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import sys
from SLIP_models import SLIP
from sklearn.metrics.pairwise import cosine_similarity
import random
import logging
import time
from datetime import datetime
from pytz import timezone

logging.Formatter.converter = lambda *args: datetime.now(tz=timezone('Asia/Taipei')).timetuple()

file_handler = logging.FileHandler(filename='/home/shellyhsu/SLIP/logger/' + 'test_C_nosie_blur.log')
stdout_handler = logging.StreamHandler(stream=sys.stdout)
handlers = [file_handler, stdout_handler]
date = '%(asctime)s %(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=date, handlers=handlers)

def Find_Optimal_Cutoff(TPR, FPR, threshold):
    y = TPR + (1 - FPR)
    Youden_index = np.argmax(y)
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point

def normalize_array(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    scaled_arr = (arr - min_val) / (max_val - min_val)  # Scale to [0, 1]
    normalized_arr = (scaled_arr * 255).astype(np.uint8)  # Scale to [0, 255] and convert to unsigned byte
    return normalized_arr

def NormalizeTorchData(data):
    return (data - torch.min(data)) / (torch.max(data) - torch.min(data))


class MyDataset_test(data.Dataset):
    def __init__(self, images, labels, preprocess):
        self.images = images
        self.labels = labels
        self.preprocess = preprocess

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        return self.preprocess(self.images[idx]), self.labels[idx]

bestACER = 100.0
followingAUC = 100.0
bestAPCER = 100.0
bestBPCER = 100.0

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
clipmodel, preprocess = clip.load("ViT-B/32", device=device, jit=False)
clipmodel.eval()
clip.model.convert_weights(clipmodel)

model = SLIP().to(device)

# protocol = int(sys.argv[1])
# model_path = f"/SLIP/{protocol}/"
model_path = "/shared/.../SLIP/"

# dataset1 = 'casia'
dataset1 = 'casia'

live_path1 = '/shared/.../data/domain-generalization/' + dataset1 + '_images_live.npy'
spoof_path1 = '/shared/.../data/domain-generalization/' + dataset1 + '_images_spoof.npy'

test_live_data = np.load(live_path1)
test_spoof_data = np.load(spoof_path1)
# data3 = np.load(live_path3)
# data_npy = np.concatenate((data1, data2, data3), axis=0)

# test_live_data = np.load(f"/SLIP/test_images_live.npy")
# test_spoof_data = np.load(f"/SLIP/test_images_spoof.npy")
    
test_data = np.vstack([test_live_data, test_spoof_data]) 

test_images = [Image.fromarray(normalize_array(image).astype(np.uint8)) for image in test_data]

labels0 = np.zeros(test_live_data.shape[0])
labels1 = np.ones(test_spoof_data.shape[0])

test_labels = np.hstack([labels0, labels1])

test_labels_tensor = torch.tensor(test_labels).float()

test_dataset = MyDataset_test(test_images, test_labels_tensor, preprocess)

test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)
test_set_live = ["Here's a genuine human face.", 
                    "This is a live face.", 
                    "This represents a real human face."]
test_set_print = ["Here's a photo of a face on paper.", 
                    "This paper features a sketched face.", 
                    "This is a face depicted on paper.",
                    "A face is illustrated on this paper.",
                    "This paper displays a face."]
test_set_replay = ["This screen displays a face.", 
                    "A face appears on this screen.", 
                    "Here's a face presented on a screen.", 
                    "This screen features a facial image.", 
                    "A face is visible on this monitor."]
test_set_mask = ["Here's a face covered with a mask.", 
                    "This is a masked face.", 
                    "A face is concealed by a mask here.", 
                    "This face is adorned with a mask.",
                    "This depicts a face with a mask on."]


import glob 
paths = glob.glob(model_path + '/*.pth')
count = 0

# for i in range(len(paths)):
#     # model.load_state_dict(torch.load(model_path + f'/model_count_{i+1:03d}.pth'))
#     model.load_state_dict(torch.load(model_path + f'/model_count_{i+1:03d}.pth', weights_only=True))
#     model.eval()
#     map_scores = []
#     new_labels = []

#     with torch.no_grad():
#         cnt = 0
#         for images, labels in test_loader:
#             images = images.to(device)
#             s = images.shape[0]
#             sentences_token1 = clip.tokenize(random.choice(test_set_live)).to(device)
#             sentences_token1 = sentences_token1.repeat(s, 1)
#             sentences_token2 = clip.tokenize(random.choice(test_set_print)).to(device)
#             sentences_token2 = sentences_token2.repeat(s, 1)
#             sentences_token3 = clip.tokenize(random.choice(test_set_replay)).to(device)
#             sentences_token3 = sentences_token3.repeat(s, 1)
#             sentences_token4 = clip.tokenize(random.choice(test_set_mask)).to(device)
#             sentences_token4 = sentences_token4.repeat(s, 1)

#             image_map, _, _, i_f, tf1, tf2, tf3, tf4, _, _ = model(images, sentences_token1, sentences_token2, sentences_token3, sentences_token4)

#             scores_cues = torch.sum(image_map, dim=(2, 3)) / (image_map.shape[2] * image_map.shape[3])
#             scores_cues = torch.squeeze(scores_cues, 1)

#             for k in range(0, images.size(0)):
#                 map_scores.append(scores_cues[k].cpu().numpy() / 255.0)
#                 new_labels.append(labels[k].cpu().numpy())


#     fpr, tpr, thresholds = roc_curve(new_labels, map_scores, pos_label=1)
#     threshold_cs, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds)

#     TP = 0.0000001
#     TN = 0.0000001
#     FP = 0.0000001
#     FN = 0.0000001

#     for j in range(len(map_scores)):
#         score = map_scores[j]
#         if (score >= threshold_cs and new_labels[j] == 1):
#             TP += 1
#         elif (score < threshold_cs and new_labels[j] == 0):
#             TN += 1
#         elif (score >= threshold_cs and new_labels[j] == 0):
#             FP += 1
#         elif (score < threshold_cs and new_labels[j] == 1):
#             FN += 1

#     APCER = FP / (TN + FP)
#     NPCER = FN / (FN + TP)

#     if np.round((APCER + NPCER) / 2, 4) < bestACER:
#         bestACER = np.round((APCER + NPCER) / 2, 4)
#         followingAUC = np.round(roc_auc_score(new_labels, map_scores), 4)
#         bestAPCER = APCER
#         bestBPCER = NPCER
#     # print(f"Savestep  {i+1:03d} ACER: {np.round((APCER + NPCER) / 2, 4)} AUC: {np.round(roc_auc_score(new_labels, map_scores),4)}")
#     logging.info(f"Savestep {i+1:03d} ACER: {np.round((APCER + NPCER) / 2, 4)} AUC: {np.round(roc_auc_score(new_labels, map_scores), 4)}")

# print(f"ACER: {bestACER}")

for i in range(len(paths)):
    # Load the model state
    model.load_state_dict(torch.load(model_path + f'/model_count_{i+1:03d}.pth', weights_only=True))
    model.eval()

    # Initialize lists to collect scores and labels, stored as GPU tensors
    map_scores = torch.tensor([], device=device)
    new_labels = torch.tensor([], device=device)

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            s = images.shape[0]
            
            # Tokenization (perform once for each batch)
            sentences_token1 = clip.tokenize(random.choice(test_set_live)).to(device).repeat(s, 1)
            sentences_token2 = clip.tokenize(random.choice(test_set_print)).to(device).repeat(s, 1)
            sentences_token3 = clip.tokenize(random.choice(test_set_replay)).to(device).repeat(s, 1)
            sentences_token4 = clip.tokenize(random.choice(test_set_mask)).to(device).repeat(s, 1)

            # Model prediction
            image_map, _, _, i_f, tf1, tf2, tf3, tf4, _, _ = model(images, sentences_token1, sentences_token2, sentences_token3, sentences_token4)
            
            # Calculate scores and normalize, keeping scores on the GPU
            scores_cues = torch.sum(image_map, dim=(2, 3)) / (image_map.shape[2] * image_map.shape[3])
            scores_cues = torch.squeeze(scores_cues, 1) / 255.0  # Normalize

            # Append batch scores and labels to GPU-based lists
            map_scores = torch.cat((map_scores, scores_cues))
            new_labels = torch.cat((new_labels, labels))

    # Move scores and labels to CPU for final evaluation
    map_scores_cpu = map_scores.cpu().numpy()
    new_labels_cpu = new_labels.cpu().numpy()

    # Calculate ROC and other performance metrics on CPU
    fpr, tpr, thresholds = roc_curve(new_labels_cpu, map_scores_cpu, pos_label=1)
    threshold_cs, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds)

    # Initialize metrics for GPU-based counting
    TP = torch.tensor(1e-7, device=device)
    TN = torch.tensor(1e-7, device=device)
    FP = torch.tensor(1e-7, device=device)
    FN = torch.tensor(1e-7, device=device)

    # Calculate TP, TN, FP, FN on GPU
    TP = torch.sum((map_scores >= threshold_cs) & (new_labels == 1)).float()
    TN = torch.sum((map_scores < threshold_cs) & (new_labels == 0)).float()
    FP = torch.sum((map_scores >= threshold_cs) & (new_labels == 0)).float()
    FN = torch.sum((map_scores < threshold_cs) & (new_labels == 1)).float()

    # Compute APCER, NPCER, and ACER on GPU
    APCER = FP / (TN + FP)
    NPCER = FN / (FN + TP)
    current_ACER = (APCER + NPCER) / 2

    # Move final metrics to CPU if needed for further processing
    current_ACER = current_ACER.cpu().item()
    current_AUC = roc_auc_score(new_labels_cpu, map_scores_cpu)
    
    if current_ACER < bestACER:
        bestACER = current_ACER
        followingAUC = current_AUC
        bestAPCER = APCER.cpu().item()
        bestBPCER = NPCER.cpu().item()

    logging.info(f"Savestep {i+1:03d} ACER: {current_ACER:.4f} AUC: {current_AUC:.4f}")

print(f"Best ACER: {bestACER}")