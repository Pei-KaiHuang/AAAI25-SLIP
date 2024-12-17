import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as data
import clip
from PIL import Image
import numpy as np
import os
import random
import sys
from SLIP_models import SLIP
from einops import rearrange
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def normalize_array(arr):
    min_val = np.min(arr)
    max_val = np.max(arr)
    scaled_arr = (arr - min_val) / (max_val - min_val)  # Scale to [0, 1]
    normalized_arr = (scaled_arr * 255).astype(np.uint8)  # Scale to [0, 255] and convert to unsigned byte
    return normalized_arr

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

model = SLIP().to(device)

# protocol = int(sys.argv[1])
# model_path = f"/SLIP/{protocol}/"
model_path = "/shared/.../SLIP/"
os.system("rm -r " + model_path)
# data_npy = np.load(f"/home/SLIP/train_images_live.npy")

####levae-one-out

dataset1 = 'MSU'
dataset2 = 'replay'
# dataset3 = 'Oulu'
dataset3 = 'casia'

live_path1 = '/shared/.../domain-generalization/' + dataset1 + '_images_live.npy'
live_path2 = '/shared/.../domain-generalization/' + dataset2 + '_images_live.npy'
live_path3 = '/shared/.../domain-generalization/' + dataset3 + '_images_live.npy'
data1 = np.load(live_path1)
data2 = np.load(live_path2)
data3 = np.load(live_path3)
data_npy = np.concatenate((data1, data2, data3), axis=0)


l = data_npy.shape[0]
live_prompt_set = ["Here's a genuine human face.", "This is a live face.", "This represents a real human face."]
content_prompt_set = ["This image shows a human face.", "This is the visage of a human.", "This is a man.", "This is a women."]

positions = ["upper",
             "lower",
             "left",
             "right",
             "upper left",
             "upper right",
             "lower left",
             "lower right"]

objects = ["fish", "flower", "airplane", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "paper", "screen", "mask", "monitor"]

sentences = []
for _ in range(l):
    k = random.randint(0, 7)
    t = random.randint(0, 2)
    t1 = random.choice(live_prompt_set)
    object = random.choice(objects)
    
    t2 = "This image illustrates the " + positions[k] + " part of the face being occluded by a " + random.choice(objects) + "."

    t3 = random.choice(content_prompt_set)
    t4 = t1 + ' ' + t2
    sentences.append([t1, t2, t3, t4, k, t])
    
sentences = np.array(sentences)

def generate_map(region, map_size=32, value_range=(0, 255), offset_range=2, t=9):
    # Initialize the map
    response_map = np.zeros((map_size, map_size))
    
    # Define region centers based on the input region number
    region_centers = {
        1: (map_size * 0.25, map_size * 0.5),  # Upper
        2: (map_size * 0.75, map_size * 0.5),  # Lower
        3: (map_size * 0.5, map_size * 0.25),  # Left
        4: (map_size * 0.5, map_size * 0.75),  # Right
        5: (map_size * 0.25, map_size * 0.25), # Upper left
        6: (map_size * 0.25, map_size * 0.75), # Upper right
        7: (map_size * 0.75, map_size * 0.25), # Lower left
        8: (map_size * 0.75, map_size * 0.75), # Lower right
        9: (map_size * 0.5, map_size * 0.5)
    }

    if region in region_centers:
        # Get the center for the Gaussian peak
        center_y, center_x = region_centers[region]
        
        # Introduce randomness in the center within the specified offset range
        random_offset_y = np.random.randint(-offset_range, offset_range + 1)
        random_offset_x = np.random.randint(-offset_range, offset_range + 1)
        
        # Ensure that the center plus offset stays within bounds
        center_y = max(0, min(map_size - 1, center_y + random_offset_y))
        center_x = max(0, min(map_size - 1, center_x + random_offset_x))

        response_map[int(center_y), int(center_x)] = 1

        # Apply Gaussian filter to simulate the response
        if t == 0:
            sigma1 = np.random.randint(6, 7)
            sigma2 = np.random.randint(6, 7)
        elif t == 1:
            sigma1 = np.random.randint(4, 5)
            sigma2 = np.random.randint(4, 5)
        elif t == 2:
            sigma1 = np.random.randint(2, 3)
            sigma2 = np.random.randint(2, 3)
        else:
            sigma1 = np.random.randint(3, 7)
            sigma2 = np.random.randint(3, 7)
        response_map = gaussian_filter(response_map, sigma=(sigma1, sigma2))
        
        # Normalize and scale the map to the specified range
        response_map = np.interp(response_map, (response_map.min(), response_map.max()), value_range)

    return response_map

class MyDataset_train(data.Dataset):
    def __init__(self, data_npy, sentences, preprocess):
        self.data_npy = data_npy
        self.sentences = sentences
        self.preprocess = preprocess

    def __len__(self):
        return data_npy.shape[0]

    def __getitem__(self, idx):
        array = normalize_array(data_npy[idx])
        array = array.astype(np.uint8)
        image = Image.fromarray(array, 'RGB')
        sentence1, sentence2, sentence3, sentence4, k, t = self.sentences[idx]
        k = int(k)
        t = int(t)

        return self.preprocess(image), sentence1, sentence2, sentence3, sentence4, generate_map(region = k + 1, t = t)


learning_rate = 5e-5
batch_size = 8
savecount = 0
savestep = int(l / (batch_size * 8)) + 1
print("Savestep: ",savestep)

# Create the dataset and data loader
dataset = MyDataset_train(data_npy, sentences, preprocess)
dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training settings
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2)
epochs = 50

COS = nn.CosineSimilarity()
MSE = nn.MSELoss()

for epoch in range(epochs):
    step = 0
    total_loss = 0
    for batch in dataloader:
        step = step + 1
        images, sentences1, sentences2, sentences3, sentences4, m_s_GT = batch
        s = images.shape[0]
        images = images.to(device)
        m_s_GT = rearrange(m_s_GT, 'b h w -> b 1 h w').to(device)
        m_s_GT = m_s_GT.float()

        sentences1 = clip.tokenize(sentences1).to(device)
        sentences2 = clip.tokenize(sentences2).to(device)
        sentences3 = clip.tokenize(sentences3).to(device)
        sentences4 = clip.tokenize(sentences4).to(device)

        label_1 = torch.ones((s, 1))
        label_0 = torch.zeros((s, 1))
        label = torch.vstack([label_1, label_1, label_0]).to(device)
        label = torch.squeeze(label)

        m_l_GT = (m_s_GT * 0.0).to(device)

        image_map, text_map1, text_map2, image_f, text_f1, text_f2, text_f3, text_f4, text_f_ent, image_map_ent = model(images, sentences1, sentences2, sentences3, sentences4, "I")

        loss_L = MSE(image_map, m_l_GT) + MSE(text_map1, m_l_GT)
        loss_S = MSE(text_map2, m_s_GT)

        loss_FA = -torch.mean(COS(image_f, text_f1)) + torch.mean(COS(image_f, text_f2))
        loss_FD = loss_FA * 0.0
        loss_D = loss_FD + loss_FA
        
        loss_A = MSE(image_map_ent, m_s_GT)

        loss_I = (loss_L + loss_S + loss_A) + 0.8 * (loss_D)

        # Backpropagation
        optimizer.zero_grad()
        loss_I.backward()
        model.clipModel.float()
        optimizer.step()
        clip.model.convert_weights(model.clipModel)

        image_map, text_map1, text_map2, image_f, text_f1, text_f2, text_f3, text_f4, text_f_ent, image_map_ent = model(images, sentences1, sentences2, sentences3, sentences4, "T")


        loss_L = MSE(text_map1, m_l_GT)
        loss_S = MSE(text_map2, m_s_GT)

        loss_FA = -torch.mean(COS(image_f, text_f1)) + torch.mean(COS(image_f, text_f2))
        loss_FD = torch.mean(COS(text_f1, text_f2)) + torch.mean(COS(text_f1, text_f3)) + torch.mean(COS(text_f2, text_f3))
        loss_D = loss_FD + loss_FA

        loss_A = torch.mean(COS(text_f4, text_f_ent)) + MSE(image_map_ent, m_s_GT)

        loss_T = (loss_L + loss_S + loss_A) + 0.8 * (loss_D)

        # Backpropagation
        optimizer.zero_grad()
        loss_T.backward()
        model.clipModel.float()
        optimizer.step()
        clip.model.convert_weights(model.clipModel)

        if step % savestep == 0:
            if not os.path.exists(model_path):
                os.makedirs(model_path) 
            torch.save(model.state_dict(), model_path + f'/model_count_{savecount+1:03d}.pth')
            savecount += 1

        total_loss += loss_I.item() + loss_T.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch + 1}, Loss: {avg_loss}")