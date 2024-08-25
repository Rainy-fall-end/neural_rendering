import torch
import torch.nn as nn
decay_rate = 1e-3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_points = 200
indices_decay = torch.arange(int(n_points), dtype=torch.float32, device=device)
decay = -decay_rate * indices_decay
max_decay = torch.max(decay)
stable_decay = decay - max_decay
exponential_decay = torch.exp(stable_decay).to(device)
def find_index_with_exponential_decay(tensor):
    threshold = 0.5
    masks = torch.sigmoid(50*(tensor - threshold))
    
    probs = exponential_decay * masks
    sum = torch.sum(probs, dim=1).unsqueeze(1)
    probs = probs / sum
    
    weighted_index = torch.sum(indices_decay * probs, dim=1)
    
    return weighted_index

def loss_fn(output_label, labels, dist, all_outputs):

    mask = (labels > 0.5).squeeze()

    # Applichiamo la soglia morbida
    indices_first = find_index_with_exponential_decay(all_outputs.view(-1, int(n_points))).view(-1, 1) / (int(n_points))
    
    # L1 Loss tra gli indici soft e dist
    loss1 = nn.L1Loss()(indices_first[mask], dist[mask])
    loss0 = nn.BCELoss()(output_label, labels)

    return loss0 + loss1


def loss_fn2(output_rgb, rgb):

    loss2 = nn.MSELoss()(output_rgb, rgb)

    return loss2
