import torch
import torch.nn as nn
from gridencoder import GridEncoder
import torch
from utils import sample_points_along_ray
n_points = 200
class HitNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.level_dim = 1
        self.num_levels = 5
        base_resolution = 8
        self.encoder_label = GridEncoder(input_dim=3, num_levels=self.num_levels, level_dim=self.level_dim, base_resolution=base_resolution)

        self.model_label = nn.Sequential(
            nn.Linear(self.num_levels, 1),
            nn.Sigmoid()
        )

    def predict(self, x):
        return self.grids(x, self.model, self.encoder)

    
    def forward(self, x, dist_true = None):

        num_points = x.size(0)
        with torch.no_grad():
            points_encoded = sample_points_along_ray(x[:, :2], x[:, 2:4])

        output = self.encoder_label(points_encoded).reshape(num_points, int(n_points), -1)
        output = self.model_label(output)
        output_hits, _ = torch.max(output, dim=1)

        # get the first voxel hitted
        all_labels = (output > 0.5).float()
        indices_first = torch.argmax(all_labels, dim=1).view(-1)

        return output_hits, output, indices_first
  