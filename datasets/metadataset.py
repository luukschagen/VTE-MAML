import torch.utils.data as data
from math import pi
import torch


class MetaDataset(data.Dataset):

    def __init__(self, task_num, k_shot, k_query, n_way=None):
        super(MetaDataset, self).__init__()
        self.task_num = task_num
        self.k_shot = k_shot
        self.k_query = k_query
        self.n_way = n_way
        self.x_s, self.x_q, self.y_s, self.y_q = (None for _ in range(4))

    def __len__(self):
        return self.task_num

    def __getitem__(self, item):
        return self.x_s[item], self.x_q[item], self.y_s[item], self.y_q[item]


class Sinusoid(MetaDataset):
    def __init__(self, task_num, k_shot, k_query, amp_range=(0.1, 5),
                 phase_range=(0, 2 * pi), freq_range=(1, 1), noise=0.3):
        super(Sinusoid, self).__init__(task_num=task_num, k_shot=k_shot, k_query=k_query)
        self.amp_range = amp_range
        self.phase_range = phase_range
        self.freq_range = freq_range
        self.noise = noise

    def __getitem__(self, item):
        if item >= self.task_num:
            raise StopIteration
        x_s = torch.rand((1, self.k_shot)) * 10 - 5
        x_q = torch.rand((1, self.k_query)) * 10 - 5
        amp = (torch.rand(1) * (self.amp_range[1] - self.amp_range[0]) + self.amp_range[0]).view(-1, 1)
        phase = (torch.rand(1) * (self.phase_range[1] - self.phase_range[0]) + self.phase_range[0]).view(-1, 1)
        freq = (torch.rand(1) * (self.freq_range[1] - self.freq_range[0]) + self.freq_range[0]).view(-1, 1)
        e_s = torch.distributions.Normal(0, torch.Tensor([self.noise for _ in range(1)])).sample(
            [self.k_shot]).view(self.k_shot, 1).transpose(0, 1)
        e_q = torch.distributions.Normal(0, torch.Tensor([self.noise for _ in range(1)])).sample(
            [self.k_query]).view(self.k_query, 1).transpose(0, 1)
        y_s = (amp * torch.sin(freq * x_s + phase)) + e_s
        y_q = (amp * torch.sin(freq * x_q + phase)) + e_q
        x_s = x_s.view(self.k_shot, 1)
        x_q = x_q.view(self.k_query, 1)
        y_s = y_s.view(self.k_shot, 1)
        y_q = y_q.view(self.k_query, 1)

        return x_s, x_q, y_s, y_q


class Linear(MetaDataset):
    def __init__(self, task_num, k_shot, k_query, alpha_range=(-3, 3), beta_range=(-3, 3), noise=0.3):
        super(Linear, self).__init__(task_num, k_shot, k_query)
        self.alpha_range = alpha_range
        self.beta_range = beta_range
        self.noise = noise

    def __getitem__(self, item):
        if item >= self.task_num:
            raise StopIteration
        x_s = torch.rand((1, self.k_shot)) * 10 - 5
        x_q = torch.rand((1, self.k_query)) * 10 - 5
        alpha = (torch.rand(1) * (self.alpha_range[1] - self.alpha_range[0]) + self.alpha_range[0]).view(-1, 1)
        beta = (torch.rand(1) * (self.beta_range[1] - self.beta_range[0]) + self.beta_range[0]).view(-1, 1)
        e_s = torch.distributions.Normal(0, torch.Tensor([self.noise for _ in range(1)])).sample(
            [self.k_shot]).view(self.k_shot, 1).transpose(0, 1)
        e_q = torch.distributions.Normal(0, torch.Tensor([self.noise for _ in range(1)])).sample(
            [self.k_query]).view(self.k_query, 1).transpose(0, 1)
        y_s = alpha * x_s + beta + e_s
        y_q = alpha * x_q + beta + e_q

        x_s = x_s.view(self.k_shot, 1)
        x_q = x_q.view(self.k_query, 1)
        y_s = y_s.view(self.k_shot, 1)
        y_q = y_q.view(self.k_query, 1)

        return x_s, x_q, y_s, y_q


class Quadratic(MetaDataset):
    def __init__(self, task_num, k_shot, k_query, alpha_range=(0.02, 0.15),
                 beta_range=(-3, 3), c_range=(-3, 3), noise=0.3):
        super(Quadratic, self).__init__(task_num, k_shot, k_query)
        self.alpha_range = alpha_range
        self.beta_range = beta_range
        self.c_range = c_range
        self.noise = noise

    def __getitem__(self, item):
        if item >= self.task_num:
            raise StopIteration
        x_s = torch.rand((1, self.k_shot)) * 10 - 5
        x_q = torch.rand((1, self.k_query)) * 10 - 5
        alpha = (torch.rand(1) * (self.alpha_range[1] - self.alpha_range[0]) + self.alpha_range[0]).view(-1, 1)
        sign = (-1 if torch.randint(2, tuple([1])) == 0 else 1)
        alpha = alpha * sign
        beta = (torch.rand(1) * (self.beta_range[1] - self.beta_range[0]) + self.beta_range[0]).view(-1, 1)
        c = (torch.rand(1) * (self.c_range[1] - self.c_range[0]) + self.c_range[0]).view(-1, 1)
        e_s = torch.distributions.Normal(0, torch.Tensor([self.noise for _ in range(1)])).sample(
            [self.k_shot]).view(self.k_shot, 1).transpose(0, 1)
        e_q = torch.distributions.Normal(0, torch.Tensor([self.noise for _ in range(1)])).sample(
            [self.k_query]).view(self.k_query, 1).transpose(0, 1)
        y_s = alpha * (x_s - c) ** 2 + beta + e_s
        y_q = alpha * (x_q - c) ** 2 + beta + e_q
        x_s = x_s.view(self.k_shot, 1)
        x_q = x_q.view(self.k_query, 1)
        y_s = y_s.view(self.k_shot, 1)
        y_q = y_q.view(self.k_query, 1)

        return x_s, x_q, y_s, y_q


class L1Norm(MetaDataset):
    def __init__(self, task_num, k_shot, k_query, alpha_range=(-3, 3),
                 beta_range=(-3, 3), c_range=(-3, 3), noise=0.3):
        super(L1Norm, self).__init__(task_num, k_shot, k_query)
        self.alpha_range = alpha_range
        self.beta_range = beta_range
        self.c_range = c_range
        self.noise = noise

    def __getitem__(self, item):
        if item >= self.task_num:
            raise StopIteration
        x_s = torch.rand((1, self.k_shot)) * 10 - 5
        x_q = torch.rand((1, self.k_query)) * 10 - 5
        alpha = (torch.rand(1) * (self.alpha_range[1] - self.alpha_range[0]) + self.alpha_range[0]).view(-1, 1)
        beta = (torch.rand(1) * (self.beta_range[1] - self.beta_range[0]) + self.beta_range[0]).view(-1, 1)
        c = (torch.rand(1) * (self.c_range[1] - self.c_range[0]) + self.c_range[0]).view(-1, 1)
        e_s = torch.distributions.Normal(0, torch.Tensor([self.noise for _ in range(1)])).sample(
            [self.k_shot]).view(self.k_shot, 1).transpose(0, 1)
        e_q = torch.distributions.Normal(0, torch.Tensor([self.noise for _ in range(1)])).sample(
            [self.k_query]).view(self.k_query, 1).transpose(0, 1)
        y_s = alpha * torch.abs(x_s - c) + beta + e_s
        y_q = alpha * torch.abs(x_q - c) + beta + e_q
        x_s = x_s.view(self.k_shot, 1)
        x_q = x_q.view(self.k_query, 1)
        y_s = y_s.view(self.k_shot, 1)
        y_q = y_q.view(self.k_query, 1)

        return x_s, x_q, y_s, y_q


class Tanh(MetaDataset):
    def __init__(self, task_num, k_shot, k_query, alpha_range=(-3, 3),
                 beta_range=(-3, 3), c_range=(-3, 3), noise=0.3):
        super(Tanh, self).__init__(task_num, k_shot, k_query)
        self.alpha_range = alpha_range
        self.beta_range = beta_range
        self.c_range = c_range
        self.noise = noise

    def __getitem__(self, item):
        if item >= self.task_num:
            raise StopIteration
        x_s = torch.rand((1, self.k_shot)) * 10 - 5
        x_q = torch.rand((1, self.k_query)) * 10 - 5
        alpha = (torch.rand(1) * (self.alpha_range[1] - self.alpha_range[0]) + self.alpha_range[0]).view(-1, 1)
        beta = (torch.rand(1) * (self.beta_range[1] - self.beta_range[0]) + self.beta_range[0]).view(-1, 1)
        c = (torch.rand(1) * (self.c_range[1] - self.c_range[0]) + self.c_range[0]).view(-1, 1)
        e_s = torch.distributions.Normal(0, torch.Tensor([self.noise for _ in range(1)])).sample(
            [self.k_shot]).view(self.k_shot, 1).transpose(0, 1)
        e_q = torch.distributions.Normal(0, torch.Tensor([self.noise for _ in range(1)])).sample(
            [self.k_query]).view(self.k_query, 1).transpose(0, 1)
        y_s = alpha * torch.tanh(x_s - c) + beta + e_s
        y_q = alpha * torch.tanh(x_q - c) + beta + e_q
        x_s = x_s.view(self.k_shot, 1)
        x_q = x_q.view(self.k_query, 1)
        y_s = y_s.view(self.k_shot, 1)
        y_q = y_q.view(self.k_query, 1)

        return x_s, x_q, y_s, y_q


class MultiModal(MetaDataset):
    def __init__(self, task_num, k_shot, k_query, modes=5):
        super(MultiModal, self).__init__(task_num, k_shot, k_query)
        self.modes = modes
        self._counter = 0
        tasks_per_mode = task_num//modes
        self.datasets = [Sinusoid(tasks_per_mode, k_shot, k_query), Linear(tasks_per_mode, k_shot, k_query)]
        if modes >= 3:
            self.datasets.append(Quadratic(tasks_per_mode, k_shot, k_query))
        if modes == 5:
            self.datasets.append(L1Norm(tasks_per_mode, k_shot, k_query))
            self.datasets.append(Tanh(tasks_per_mode, k_shot, k_query))
        if modes == 4:
            raise NotImplementedError("4 modes is not part of the experiments")
        if modes > 5:
            raise NotImplementedError("5 modes is the maximum")
        if modes == 2:
            self.datasets = [Tanh(tasks_per_mode, k_shot, k_query), L1Norm(tasks_per_mode, k_shot, k_query)]

    def __getitem__(self, item):
        if item >= self.task_num:
            raise StopIteration
        index = self._counter % self.modes
        self._counter += 1
        return self.datasets[index][0]


if __name__ == '__main__':
    pass
