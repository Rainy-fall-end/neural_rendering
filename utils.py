import torch
import psutil
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_points = 200
t_values = torch.linspace(0, 1, n_points).reshape(1, n_points, 1).to(device)
def convert_spherical_to_cartesian(theta, phi):
    sin_theta = torch.sin(theta)

    x = sin_theta * torch.cos(phi)
    y = sin_theta * torch.sin(phi)
    z = torch.cos(theta)
    
    return torch.cat((x.unsqueeze(-1), y.unsqueeze(-1), z.unsqueeze(-1)), dim=-1)


def sample_points_along_ray(int1, int2):

    point1 = convert_spherical_to_cartesian(int1[:, 0], int1[:, 1])
    point2 = convert_spherical_to_cartesian(int2[:, 0], int2[:, 1])
    
    diff = point2 - point1
    points = point1.unsqueeze(1) + diff.unsqueeze(1) * t_values
    
    return points

def get_memory_usage():
    # Ottieni l'uso della memoria del processo corrente
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss  # Utilizza la memoria fisica residente (RSS)
