import time
import torch
import torch.nn as nn
import numpy as np
import tracemalloc
import numpy as np
import torch
from camera import show_camera
from networks.network_hit import HitNet
from networks.network_rgb import RgbNet
from utils import get_memory_usage
from loss import loss_fn,loss_fn2
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import torch.nn as nn
from dataset import RenderDatasetSph

tracemalloc.start()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_points = 200

def train(model_path1,model_path2,save=False,data_path="C:/works/soc/rainy/neural_rendering/test_neural_rgb_c.txt"):
    # Device configuration

    dataset = RenderDatasetSph(data_dir=data_path,max_len=500000)
    # dataset_loader = DataLoader(dataset,batch_size=2**13,shuffle=True)
    dataset_loader = DataLoader(dataset,batch_size=2048,shuffle=True)
    # Controlla lo stato della memoria iniziale
    print(f"Memory usage: {get_memory_usage()}")
    model = HitNet().to(device)
    model2 = RgbNet().to(device)
    print(f"Memory usage: {get_memory_usage()}")


    test_set = []
    test_set_labels = []
    test_set_rgb = []
    test_set_dist = []
    lines = dataset.test_data()
    # Per ogni riga del file
    for line in lines:
        line = line.strip()
        tokens = line.split()
        tokens = [float(token) for token in tokens]

        points = torch.Tensor(tokens[:4])
        label = torch.Tensor([tokens[4]])
        rgb = torch.Tensor(tokens[5:8]) / 255
        dist = torch.Tensor([tokens[-1]])

        test_set.append(points)
        test_set_rgb.append(rgb)
        test_set_labels.append(label)
        test_set_dist.append(dist)

    test_set = torch.stack(test_set).to(device)
    test_set_labels = torch.tensor(test_set_labels).to(device)
    test_set_rgb = torch.stack(test_set_rgb).to(device)
    test_set_dist = torch.stack(test_set_dist).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.01)
    #scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)


    # Train the model
    total_step = len(dataset_loader)

    add_time_total = 0
    for epoch in range(1):
        epoch_train_loss = 0

        add_time = 0

        
        for i, lines in enumerate(dataset_loader):
           
            lines = "\n".join(lines)
            data = np.fromstring(lines, sep=' ').reshape(-1, 9)

            para = torch.tensor(data[:, :4], device=device, dtype=torch.float32)
            labels = torch.tensor(data[:, 4], device=device, dtype=torch.float32).view(-1, 1)
            rgb = torch.tensor(data[:, 5:8], device=device, dtype=torch.float32) / 255
            dist = torch.tensor(data[:, -1], device=device, dtype=torch.float32).view(-1, 1)

            optimizer.zero_grad()

            now = time.time()
            output_label, all_outputs, _ = model(para, dist)
            add_time += time.time() - now

            loss1 = loss_fn(output_label, labels, dist, all_outputs)
            loss1.backward()
            optimizer.step()

            epoch_train_loss += loss1.item()

            if i % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                            .format(epoch+1, 100, i+1, total_step, loss1.item()))
                
        add_time_total += add_time
        print(f"Add time: {add_time}")

        model.eval()

        all_preds = []
        all_preds_dist = []
        with torch.no_grad():
            for i in range(0, len(test_set), 4096):
                output, _, dist = model(test_set[i:i+4096], test_set_dist[i:i+4096])
                all_preds.append(output)
                all_preds_dist.append((dist / n_points).view(-1, 1))

        all_preds = torch.cat(all_preds, dim=0)
        all_preds_dist = torch.cat(all_preds_dist, dim=0)

        #remove the preds_rgb that have not been hit
        mask = (all_preds > 0.5).squeeze()
        
        all_preds_dist = all_preds_dist[mask]
        all_dist_true = test_set_dist[mask]


        
        all_preds[all_preds > 0.5] = 1
        all_preds[all_preds <= 0.5] = 0
        all_preds = all_preds.cpu()
        all_labels = test_set_labels.cpu()


        f1 = f1_score(all_labels, all_preds)
        dist_loss = nn.L1Loss()(all_preds_dist, all_dist_true).item()


        print("MODEL 1")
        print(f"F1: {f1}")
        print(f"Dist L1: {dist_loss}")
    
    for epoch in range(1):
        epoch_train_loss = 0

        add_time = 0
        
        for i, lines in enumerate(dataset_loader):
           
            lines = "\n".join(lines)
            data = np.fromstring(lines, sep=' ').reshape(-1, 9)

            para = torch.tensor(data[:, :4], device=device, dtype=torch.float32)
            labels = torch.tensor(data[:, 4], device=device, dtype=torch.float32).view(-1, 1)
            rgb = torch.tensor(data[:, 5:8], device=device, dtype=torch.float32) / 255
            dist = torch.tensor(data[:, -1], device=device, dtype=torch.float32).view(-1, 1)

            optimizer2.zero_grad()

            with torch.no_grad():
                output, _, indices_first = model(para, dist)
                mask = (output > 0.5).squeeze()

            output_rgb = model2(para[mask], indices_first[mask])

            loss1 = loss_fn2(output_rgb, rgb[mask])
            loss1.backward()
            optimizer2.step()

            epoch_train_loss += loss1.item()

            if i % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                            .format(epoch+1, 100, i+1, total_step, loss1.item()))
                
        add_time_total += add_time
        print(f"Add time: {add_time}")
        model.eval()
        all_preds = []
        all_preds_rgb = []
        all_preds_dist = []
        with torch.no_grad():
            for i in range(0, len(test_set), 4096):
                output, _, indices_first = model(test_set[i:i+4096], test_set_dist[i:i+4096])
                output_rgb = model2(test_set[i:i+4096], indices_first)
                all_preds.append(output)
                all_preds_rgb.append(output_rgb)
        all_preds = torch.cat(all_preds, dim=0)
        all_preds_rgb = torch.cat(all_preds_rgb, dim=0)
        #remove the preds_rgb that have not been hit
        mask = (all_preds > 0.5).squeeze()
        all_preds_rgb = all_preds_rgb[mask]
        all_rgb_true = test_set_rgb[mask]
        rgb_loss = nn.L1Loss()(all_preds_rgb, all_rgb_true).item()
        print("MODEL 2")
        print(f"RGB L1: {rgb_loss}")
    # save model
    if(save):
        model.eval()
        model2.eval()
        torch.save(model.state_dict(),model_path1)
        torch.save(model2.state_dict(),model_path2)
    show_camera(model, model2)
if __name__ == "__main__":
    train(model_path1="models/hit_model.pth",model_path2="models/rgb_model.pth")