from torch.utils.data import Dataset
import numpy as np
class RenderDatasetSph(Dataset):
    def __init__(self,data_dir="",max_len=100000,transform=None) -> None:
        super().__init__()

        self.transform = transform
        self.data = []
        self.test_datas = []

        with open(data_dir, 'r') as file:
            lines = file.readlines()

            np.random.shuffle(lines)
            test_data = lines[:10000]
            train_data = lines[10000:max_len]
            train_data = train_data[:max_len]
            self.data = train_data
            self.test_datas = test_data

        del lines

    def test_data(self):
        return self.test_datas

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)