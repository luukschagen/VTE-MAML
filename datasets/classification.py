import torch
import numpy as np
from sklearn.datasets import make_moons, make_circles, make_classification
from datasets.metadataset import MetaDataset


def rotation_matrix(thetas):
    top = torch.stack([torch.cos(thetas), -torch.sin(thetas)], dim=1).view(-1)
    bottom = torch.stack([torch.sin(thetas), torch.cos(thetas)], dim=1).view(-1)
    return torch.stack([top, bottom], dim=1)


class MoonClassification(MetaDataset):
    def __init__(self, task_num, k_shot, k_query, noise=0.2):
        super(MoonClassification, self).__init__(task_num, k_shot, k_query)
        self.noise = noise

    def __getitem__(self, item):
        if item >= self.task_num:
            raise StopIteration
        angle = torch.rand(1) * 2 * np.pi
        rotation = rotation_matrix(angle)

        x_s, y_s = make_moons(self.k_shot, noise=self.noise)
        x_q, y_q = make_moons(self.k_query, noise=self.noise)

        x_s, x_q, y_s, y_q = (torch.Tensor(x) for x in (x_s, x_q, y_s, y_q))

        x_s = torch.matmul(rotation, x_s.permute(1, 0)).permute(1, 0)
        x_q = torch.matmul(rotation, x_q.permute(1, 0)).permute(1, 0)
        y_s = y_s.view(self.k_shot, 1)
        y_q = y_q.view(self.k_query, 1)

        return x_s, x_q, y_s, y_q


class CircleClassification(MetaDataset):
    def __init__(self, task_num, k_shot, k_query, distance=0.5, noise=0.1):
        super(CircleClassification, self).__init__(task_num, k_shot, k_query)
        self.noise = noise
        self.distance = distance

    def __getitem__(self, item):
        if item >= self.task_num:
            raise StopIteration
        translation = torch.rand(2)*5-2.5
        scale = torch.rand(1)*2+0.5
        x_s, y_s = make_circles(self.k_shot, factor=self.distance, noise=self.noise)
        x_q, y_q = make_circles(self.k_query, factor=self.distance, noise=self.noise)

        x_s, x_q, y_s, y_q = (torch.Tensor(x) for x in (x_s, x_q, y_s, y_q))

        x_s, x_q = x_s*scale+translation, x_q*scale+translation
        y_s, y_q = [y.view(-1, 1) for y in (y_s, y_q)]
        return x_s, x_q, y_s, y_q


class LinearClassification(MetaDataset):
    def __init__(self, task_num, k_shot, k_query):
        super(LinearClassification, self).__init__(task_num, k_shot, k_query)

    def __getitem__(self, item):
        if item >= self.task_num:
            raise StopIteration
        x, y = make_classification(self.k_shot + self.k_query, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1)
        x_s, y_s = x[:self.k_shot], y[:self.k_shot]
        x_q, y_q = x[self.k_shot:], y[self.k_shot:]
        x_s, x_q, y_s, y_q = (torch.Tensor(x) for x in (x_s, x_q, y_s, y_q))
        y_s, y_q = [y.view(-1, 1) for y in (y_s, y_q)]

        return x_s, x_q, y_s, y_q


class MultiModalClassification(MetaDataset):
    def __init__(self, task_num, k_shot, k_query, modes=3):
        super(MultiModalClassification, self).__init__(task_num, k_shot, k_query)
        self.modes = modes
        self._counter = 0
        tasks_per_mode = task_num//modes
        self.datasets = [MoonClassification(tasks_per_mode, k_shot, k_query),
                         CircleClassification(tasks_per_mode, k_shot, k_query)]

        if modes >= 3:
            self.datasets.append(LinearClassification(tasks_per_mode, k_shot, k_query))

        if modes > 3:
            raise NotImplementedError("4 modes is not part of the experiments")
        if modes > 5:
            raise NotImplementedError("5 modes is the maximum")

    def __getitem__(self, item):
        if item >= self.task_num:
            raise StopIteration
        index = self._counter % self.modes
        self._counter += 1
        return self.datasets[index][0]


if __name__ == '__main__':
    pass
