import torch
import torch.nn as nn
from gridencoder import GridEncoder
import torch
from utils import sample_points_along_ray
n_points = 200
class NgpNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.level_dim1 = 1
        self.level_dim2 = 4
        self.num_levels1 = 5
        self.num_levels2 = 6
        self.base_resolution1 = 8
        self.base_resolution2 = 4
        self.encoder_label = GridEncoder(input_dim=3, num_levels=self.num_levels1, level_dim=self.level_dim1, base_resolution=self.base_resolution1)
        self.encoder_rgb = GridEncoder(input_dim=3, num_levels=self.num_levels2, level_dim=self.level_dim2, base_resolution=self.base_resolution2)
        self.model_label = nn.Sequential(
            nn.Linear(self.num_levels1, 1),
            nn.Sigmoid()
        )
        self.model_rgb = nn.Sequential(
            nn.Linear(self.num_levels2 * (self.level_dim2) * 5, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )
    def forward(self, x):
        num_points = x.size(0)
        with torch.no_grad():
            points_encoded = sample_points_along_ray(x[:, :2], x[:, 2:4])
        output_label = self.encoder_label(points_encoded).reshape(num_points, int(n_points), -1)
        output_label = self.model_label(output_label)
        output_hits, _ = torch.max(output_label, dim=1)

        # get the first voxel hitted
        all_labels = (output_label > 0.5).float()
        indices_first = torch.argmax(all_labels, dim=1).view(-1)
        
        points_encoded = self.encoder_rgb(points_encoded)

        indices_prev1 = torch.clamp(indices_first - 2, 0, 299)
        indices_prev2 = torch.clamp(indices_first - 1, 0, 299)
        indices_next1 = torch.clamp(indices_first + 1, 0, 299)
        indices_next2 = torch.clamp(indices_first + 2, 0, 299)

        num_points_ls = torch.arange(num_points)
        points_encoded_first = torch.cat([
                                            points_encoded[num_points_ls, indices_prev1],
                                            points_encoded[num_points_ls, indices_prev2],
                                            points_encoded[num_points_ls, indices_first], 
                                            points_encoded[num_points_ls, indices_next1],
                                            points_encoded[num_points_ls, indices_next2]
                                        ], dim=-1)

        output_rgb = self.model_rgb(points_encoded_first)

        return output_hits,output_label,output_rgb
  