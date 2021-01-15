from datasets.metadataset import MetaDataset
import pandas as pd
import numpy as np
import torch
from torchvision.transforms import transforms
from PIL import Image


class MiniImagenet(MetaDataset):
    def __init__(self, task_num, k_shot, k_query, n_way, path="../datasets/MiniImagenet", mode='train',
                 imgsize=84, channels=3, dynamic=True):
        super(MiniImagenet, self).__init__(task_num, k_shot, k_query, n_way)
        self.mode = mode
        self.imgsize = imgsize
        self.channels = channels
        self.path = path
        self.dynamic = dynamic

        if self.mode == "test":
            self.table = pd.read_csv(path + "/test.csv")
        elif self.mode == "validation":
            self.table = pd.read_csv(path + "/val.csv")
        else:
            self.table = pd.read_csv(path + "/train.csv")

        self.classes = self.table['label'].unique()

        if not dynamic:
            self.tasks = np.array([np.random.choice(self.classes, size=self.n_way, replace=False)
                                   for _ in range(self.task_num)])

            self.x_s, self.x_q, self.y_s, self.y_q, self.lookup_table = self.sample_images(self.tasks[0])

            for task in self.tasks[1:]:
                x_s, x_q, y_s, y_q, table = self.sample_images(task)
                self.x_s = np.append(self.x_s, x_s, axis=0)
                self.x_q = np.append(self.x_q, x_q, axis=0)
                self.y_s = np.append(self.y_s, y_s, axis=0)
                self.y_q = np.append(self.y_q, y_q, axis=0)
                self.lookup_table = np.append(self.lookup_table, table, axis=0)
            self.y_s = torch.Tensor(self.y_s).view(task_num, n_way * k_shot).long()
            self.y_q = torch.Tensor(self.y_q).view(task_num, n_way * k_query).long()

    def __getitem__(self, item):
        if item >= self.task_num:
            raise StopIteration
        if self.dynamic:
            task = np.random.choice(self.classes, size=self.n_way, replace=False)
            x_s, x_q, y_s, y_q, lookup_table = self.sample_images(task)
            x_s, x_q, y_s, y_q = map(lambda x: x.flatten(), (x_s, x_q, y_s, y_q))
            y_s, y_q = torch.Tensor(y_s).long(), torch.Tensor(y_q).long()
        else:
            x_s = self.x_s[item]  # (meta-batch, k_shot*n_way)
            x_q = self.x_q[item]
            y_s = self.y_s[item]
            y_q = self.y_q[item]

        x_s = self.load_images(x_s)  # (meta-batch, k_shot*n_way, channels, size, size)
        x_q = self.load_images(x_q)

        return x_s, x_q, y_s, y_q

    def sample_images(self, task):
        support_idx = np.array([])
        query_idx = np.array([])
        for cls in task:
            support = np.random.choice(self.table[self.table['label'] == cls].index, self.k_shot)
            query = np.random.choice(self.table[self.table['label'] == cls].index, self.k_query)
            support_idx = np.append(support_idx, support)
            query_idx = np.append(query_idx, query)

        x_s = np.array(self.table['filename'].iloc[support_idx]).reshape((1, -1))
        x_q = np.array(self.table['filename'].iloc[query_idx]).reshape((1, -1))
        table, y_s = np.unique(np.array(self.table['label'].iloc[support_idx]), return_inverse=True)
        __, y_q = np.unique(np.array(self.table['label'].iloc[query_idx]), return_inverse=True)

        y_s = y_s.reshape(1, -1)
        y_q = y_q.reshape(1, -1)
        table = table.reshape(1, -1)

        return x_s, x_q, y_s, y_q, table

    def load_images(self, imageset):
        images = torch.zeros(*imageset.shape, self.channels, self.imgsize, self.imgsize)
        for index in np.ndindex(imageset.shape):
            images[index] = self.load_image(imageset[index])
        return images

    def load_image(self, filename):
        path = self.path + "/images/" + filename
        transform = transforms.Compose([lambda x: Image.open(x).convert('RGB'),
                                        transforms.Resize((self.imgsize, self.imgsize)),
                                        transforms.ToTensor()
                                        ])
        return transform(path)


if __name__ == '__main__':
    pass
