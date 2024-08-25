import torch
import torch.nn as nn
from gridencoder import GridEncoder
import torch
from utils import sample_points_along_ray
class RgbNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.level_dim2 = 4
        self.num_levels2 = 6
        base_resolution2 = 4
        self.encoder_rgb = GridEncoder(input_dim=3, num_levels=self.num_levels2, level_dim=self.level_dim2, base_resolution=base_resolution2)

        self.model_rgb = nn.Sequential(
            nn.Linear(self.num_levels2 * (self.level_dim2) * 5, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )
    
    def forward(self, x, indices_first):

        num_points = torch.arange(x.size(0))
        with torch.no_grad():
            points_encoded = sample_points_along_ray(x[:, :2], x[:, 2:4])

        # get the first voxel hitted
        points_encoded = self.encoder_rgb(points_encoded)

        indices_prev1 = torch.clamp(indices_first - 2, 0, 299)
        indices_prev2 = torch.clamp(indices_first - 1, 0, 299)
        indices_next1 = torch.clamp(indices_first + 1, 0, 299)
        indices_next2 = torch.clamp(indices_first + 2, 0, 299)

        points_encoded_first = torch.cat([
                                            points_encoded[num_points, indices_prev1],
                                            points_encoded[num_points, indices_prev2],
                                            points_encoded[num_points, indices_first], 
                                            points_encoded[num_points, indices_next1],
                                            points_encoded[num_points, indices_next2]
                                        ], dim=-1)

        output_rgb = self.model_rgb(points_encoded_first)

        return output_rgb
  