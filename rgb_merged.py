import torch
import numpy as np
import tracemalloc
import numpy as np
import torch
from camera import show_camera
from networks.networks_merged import NgpNet
from utils import get_memory_usage
from loss import loss_fn3
from torch.utils.data import DataLoader
from dataset import RenderDatasetSph
import wandb
need_wandb = True
tracemalloc.start()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_points = 200

def train(model_path,config,save=False,data_path="C:/works/soc/rainy/neural_rendering/test_neural_rgb_c.txt"):
    # Device configuration
    dataset = RenderDatasetSph(data_dir=data_path,max_len=config["dataset_num"])
    # dataset_loader = DataLoader(dataset,batch_size=2**13,shuffle=True)
    dataset_loader = DataLoader(dataset,batch_size=config["batch_size"],shuffle=True)
    # Controlla lo stato della memoria iniziale
    print(f"Memory usage: {get_memory_usage()}")
    model = NgpNet().to(device)
    print(f"Memory usage: {get_memory_usage()}")
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
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
            if need_wandb:
                wandb.log({"loss":loss})
            if i % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                            .format(epoch+1, 1, i+1, total_step, loss.item()))
    if need_wandb:
        wandb.finish()
    # save model
    if(save):
        model.eval()
        torch.save(model.state_dict(),model_path)
    show_camera(model)
if __name__ == "__main__":
    config={
        "dataset_num": 5000000,
        "learning_rate": 0.01,
        "epochs": 1,
        "batch_size": 2048*2
        }
    if need_wandb:
        wandb.init(
        # set the wandb project where this run will be logged
        project="Neural_Rendering",
        # track hyperparameters and run metadata
        config=config
    )
    train(model_path="models/ngp_model.pth",config=config,data_path="C:/works/soc/rainy/neural_rendering/test_neural_Rgb_new.txt")