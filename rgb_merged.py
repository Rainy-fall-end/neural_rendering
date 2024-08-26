import time
import torch
import torch.nn as nn
import numpy as np
import tracemalloc
import numpy as np
import torch
from camera import show_camera
from networks.networks_merged import NgpNet
from utils import get_memory_usage
from loss import loss_fn3
from torch.utils.data import DataLoader
import torch.nn as nn
from dataset import RenderDatasetSph

tracemalloc.start()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_points = 200

def train(model_path,save=False,data_path="C:/works/soc/rainy/neural_rendering/test_neural_rgb_c.txt"):
    # Device configuration

    dataset = RenderDatasetSph(data_dir=data_path,max_len=5000000)
    # dataset_loader = DataLoader(dataset,batch_size=2**13,shuffle=True)
    dataset_loader = DataLoader(dataset,batch_size=2048,shuffle=True)
    # Controlla lo stato della memoria iniziale
    print(f"Memory usage: {get_memory_usage()}")
    model = NgpNet().to(device)
    print(f"Memory usage: {get_memory_usage()}")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    #scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    # Train the model
    total_step = len(dataset_loader)
    for epoch in range(1):
        epoch_train_loss = 0
        for i, lines in enumerate(dataset_loader):
            lines = "\n".join(lines)
            data = np.fromstring(lines, sep=' ').reshape(-1, 9)
            para = torch.tensor(data[:, :4], device=device, dtype=torch.float32)
            labels = torch.tensor(data[:, 4], device=device, dtype=torch.float32).view(-1, 1)
            rgb = torch.tensor(data[:, 5:8], device=device, dtype=torch.float32) / 255
            dist = torch.tensor(data[:, -1], device=device, dtype=torch.float32).view(-1, 1)
            optimizer.zero_grad()
            output_label,output_hits,output_rgb = model(para)
            loss = loss_fn3(output_label,labels,output_hits,dist,output_rgb,rgb)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            if i % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                            .format(epoch+1, 100, i+1, total_step, loss.item()))
    # save model
    if(save):
        model.eval()
        torch.save(model.state_dict(),model_path)
    show_camera(model)
if __name__ == "__main__":
    train(model_path="models/ngp_model.pth")