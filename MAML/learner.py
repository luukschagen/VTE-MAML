from torch.nn import functional as F
from MAML.helper_networks import *
import pickle


class Learner(nn.Module):
    """

    """

    def __init__(self, config):
        """

        :param config: network config file, type:list of (string, list)
        """
        super(Learner, self).__init__()

        self.config = config

        # this dict contains all tensors needed to be optimized
        self.vars = nn.ParameterList()
        # running_mean and running_var
        self.vars_bn = nn.ParameterList()

        for i, (name, param) in enumerate(self.config):
            if name == 'conv2d':
                # [ch_out, ch_in, kernelsz, kernelsz]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name == 'convt2d':
                # [ch_in, ch_out, kernelsz, kernelsz, stride, padding]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_in, ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[1])))

            elif name == 'linear':
                # [ch_out, ch_in]
                w = nn.Parameter(torch.ones(*param))
                # gain=1 according to cbfinn's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name == 'bn':
                # [ch_out]
                w = nn.Parameter(torch.ones(param[0]))
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

                # must set requires_grad=False
                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.vars_bn.extend([running_mean, running_var])

            elif name in ['tanh', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d',
                          'flatten', 'reshape', 'leakyrelu', 'sigmoid']:
                continue
            else:
                raise NotImplementedError

    def extra_repr(self):
        info = ''

        for name, param in self.config:
            if name == 'conv2d':
                tmp = 'conv2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)' \
                      % (param[1], param[0], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name == 'convt2d':
                tmp = 'convTranspose2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)' \
                      % (param[0], param[1], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name == 'linear':
                tmp = 'linear:(in:%d, out:%d)' % (param[1], param[0])
                info += tmp + '\n'

            elif name == 'leakyrelu':
                tmp = 'leakyrelu:(slope:%f)' % (param[0])
                info += tmp + '\n'


            elif name == 'avg_pool2d':
                tmp = 'avg_pool2d:(k:%d, stride:%d, padding:%d)' % (param[0], param[1], param[2])
                info += tmp + '\n'
            elif name == 'max_pool2d':
                tmp = 'max_pool2d:(k:%d, stride:%d, padding:%d)' % (param[0], param[1], param[2])
                info += tmp + '\n'
            elif name == ['flatten', 'tanh', 'relu', 'upsample', 'reshape', 'sigmoid', 'use_logits', 'bn']:
                tmp = name + ':' + str(tuple(param))
                info += tmp + '\n'
            else:
                raise NotImplementedError

        return info

    def forward(self, x, vars=None, embeddings=None, bn_training=True):
        """
        This function can be called by finetunning, however, in finetunning, we dont wish to update
        running_mean/running_var. Thought weights/bias of bn is updated, it has been separated by fast_weights.
        Indeed, to not update running_mean/running_var, we need set update_bn_statistics=False
        but weight/bias will be updated and not dirty initial theta parameters via fast_weiths.
        :param x: [b, 1, 28, 28]
        :param vars:
        :param bn_training: set False to not update
        :return: x, loss, likelihood, kld
        """

        if vars is None:
            vars = self.vars

        idx = 0
        bn_idx = 0
        embedding_idx = 0

        for name, param in self.config:
            x, idx, bn_idx = self.layer_forward(x, name, param, idx, bn_idx, vars, bn_training)
            if name in ['conv2d', 'convt2d', 'linear'] and embeddings:
                if embedding_idx < len(embeddings):
                    gamma = embeddings[embedding_idx][0] + torch.ones_like(embeddings[embedding_idx][0])
                    beta = embeddings[embedding_idx][1]
                    if len(x.shape) != 2:
                        gamma = gamma.view(1,-1,1,1)
                        beta = beta.view(1,-1,1,1)
                    x = x * gamma + beta
                    embedding_idx += 1

        # make sure variable is used properly
        assert idx == len(vars)
        assert bn_idx == len(self.vars_bn)
        if embeddings:
            assert embedding_idx == len(embeddings)

        return x

    def layer_forward(self, x, name, param, idx, bn_idx, vars, bn_training):
        if name == 'conv2d':
            w, b = vars[idx], vars[idx + 1]
            # remember to keep synchrozied of forward_encoder and forward_decoder!
            x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
            idx += 2
            # print(name, param, '\tout:', x.shape)
        elif name == 'convt2d':
            w, b = vars[idx], vars[idx + 1]
            # remember to keep synchrozied of forward_encoder and forward_decoder!
            x = F.conv_transpose2d(x, w, b, stride=param[4], padding=param[5])
            idx += 2
            # print(name, param, '\tout:', x.shape)
        elif name == 'linear':
            w, b = vars[idx], vars[idx + 1]
            x = F.linear(x, w, b)
            idx += 2
            # print('forward:', idx, x.norm().item())
        elif name == 'bn':
            w, b = vars[idx], vars[idx + 1]
            running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx + 1]
            x = F.batch_norm(x, None, None, weight=w, bias=b, training=bn_training)
            idx += 2
            bn_idx += 2

        elif name == 'flatten':
            # print(x.shape)
            x = x.view(x.size(0), -1)
        elif name == 'reshape':
            # [b, 8] => [b, 2, 2, 2]
            x = x.view(x.size(0), *param)
        elif name == 'relu':
            x = F.relu(x, inplace=param[0])
        elif name == 'leakyrelu':
            x = F.leaky_relu(x, negative_slope=param[0], inplace=param[1])
        elif name == 'tanh':
            x = torch.tanh(x)
        elif name == 'sigmoid':
            x = torch.sigmoid(x)
        elif name == 'upsample':
            x = F.upsample_nearest(x, scale_factor=param[0])
        elif name == 'max_pool2d':
            x = F.max_pool2d(x, param[0], param[1], param[2])
        elif name == 'avg_pool2d':
            x = F.avg_pool2d(x, param[0], param[1], param[2])

        else:
            raise NotImplementedError

        return x, idx, bn_idx

    def zero_grad(self, vars=None):
        """

        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self, recurse=True):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars

    def save(self, name, path="model", ):
        with open(path + "/" + name + ".pkl", 'wb') as file:
            pickle.dump(self.vars, file)

    @classmethod
    def load(cls, path, config, **kwargs):
        model = cls(config)
        with open(path, 'rb') as file:
            vars = pickle.load(file)
        model.vars = vars
        return model


class Modulation_network(nn.Module):

    def __init__(self, input_size, output_size, layers, hidden_size, output_dimensions, bidirectional=True,
                 decoder_layers=1, **kwargs):
        super(Modulation_network, self).__init__()
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size * (2 if bidirectional else 1)

        self.encoder = nn.LSTM(input_size + output_size, hidden_size, layers, bidirectional=self.bidirectional)

        self.decoders = nn.ModuleList()

        for dim in output_dimensions:
            layers = nn.ModuleList()
            for i in range(decoder_layers-1):
                layers.append(nn.Linear(self.hidden_size, self.hidden_size))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(self.hidden_size, dim * 2))
            self.decoders.append(Sequential(layers))

    def encode(self, x_support, y_support):
        flat_x = x_support.view(x_support.shape[0], -1)
        flat_y = y_support.view(y_support.shape[0], -1)
        task = torch.cat((flat_x, flat_y), dim=1).view(x_support.shape[0], 1, -1)
        out, (h_n, c_n) = self.encoder(task)
        if not self.bidirectional:
            embedding = out.view(out.shape[0], -1)[-1].view(1, -1)
        else:
            out = out.view(out.shape[0], out.shape[1], 2, out.shape[2] // 2)
            embedding = torch.cat((out[-1, :, 0], out[0, :, 1]), dim=1)
        return embedding

    def decode(self, embedding):
        outputs = []
        for decoder in self.decoders:
            output = decoder(embedding).view(2, -1)
            outputs.append(output)
        return outputs

    def forward(self, x_support, y_support):
        # reshape inputs
        encoding = self.encode(x_support, y_support)
        return self.decode(encoding)

    def save(self, name, path="model", ):
        torch.save(self.state_dict(), path + "/" + name + ".pkl")

    @classmethod
    def load(cls, path, input_size, output_size, layers, hidden_size, output_dimensions, bidirectional=True,
             decoder_layers=1, **kwargs):
        model = cls(input_size, output_size, layers, hidden_size, output_dimensions, bidirectional, decoder_layers, **kwargs)
        model.load_state_dict(torch.load(path))
        return model


class Variational_task_encoder(Modulation_network):
    def __init__(self, input_size, output_size, layers, hidden_size, output_dimensions, bidirectional=True,
                 decoder_layers=1, learnable_prior=False, **kwargs):
        super(Variational_task_encoder, self).__init__(input_size, output_size, layers, hidden_size, output_dimensions,
                                                       bidirectional, decoder_layers, **kwargs)
        self.mu_layer = nn.Linear(hidden_size*(2 if self.bidirectional else 1),
                                  hidden_size*(2 if self.bidirectional else 1))
        self.log_var_layer = nn.Linear(hidden_size*(2 if self.bidirectional else 1),
                                       hidden_size*(2 if self.bidirectional else 1))
        self.learnable_prior = learnable_prior
        if learnable_prior:
            self.prior = nn.Parameter(torch.stack([torch.zeros(hidden_size*(2 if self.bidirectional else 1)),
                                                   torch.zeros(hidden_size*(2 if self.bidirectional else 1))],
                                                  dim=0), requires_grad=True)
            self.register_parameter("prior", self.prior)
        self.kld_loss = 0

    def transform_encoding(self, encoding):
        mu = self.mu_layer(encoding)
        log_var = self.log_var_layer(encoding)
        return mu, log_var

    def sample(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        e = torch.randn_like(std)
        return mu + e * std

    def forward(self, x_support, y_support):
        self.reset_kld()
        encoding = self.encode(x_support, y_support)
        mu, log_var = self.transform_encoding(encoding)
        sample = self.sample(mu, log_var)
        self.update_kld(mu, log_var, sample)
        outputs = self.decode(sample)
        return outputs

    def update_kld(self, mu, log_var, sample):
        if not self.learnable_prior:
            kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        else:
            mu_q = mu
            sd_q = torch.exp(log_var*0.5)
            mu_p = self.prior[0,:]
            sd_p = torch.exp(self.prior[1,:]*0.5)
            kld = (0.5 * ((sd_p/sd_q).pow(2)+(mu_q-mu_p).pow(2)/sd_p.pow(2)-1+2*torch.log(sd_q/sd_p))).sum()
        self.kld_loss += kld

    def reset_kld(self):
        self.kld_loss = 0


class ConvModulation_network(Modulation_network):
    def __init__(self, input_size, output_size, conv_layers, hidden_size, output_dimensions,
                 decoder_layers=1, **kwargs):
        super(ConvModulation_network, self).__init__(1, 1, 2, hidden_size, output_dimensions, False,
                                                     decoder_layers, **kwargs)

        self.encoder = Convnet(input_size, conv_layers, hidden_size)

    def encode(self, x_support, y_support):
        return self.encoder(x_support)

    @classmethod
    def load(cls, path, input_size, output_size, conv_layers, hidden_size, output_dimensions,
             decoder_layers=1, **kwargs):
        model = cls(input_size, output_size, conv_layers, hidden_size, output_dimensions, decoder_layers,
                    **kwargs)
        model.load_state_dict(torch.load(path))
        return model


class ConvVariational_task_encoder(Variational_task_encoder):
    def __init__(self, input_size, output_size, conv_layers, hidden_size, output_dimensions,
                 decoder_layers=1, learnable_prior=False, **kwargs):
        super(ConvVariational_task_encoder, self).__init__(1, 1, 2, hidden_size, output_dimensions, False,
                                                           decoder_layers, learnable_prior, **kwargs)

        self.encoder = Convnet(input_size, conv_layers, hidden_size)

    def encode(self, x_support, y_support):
        return self.encoder(x_support)

    @classmethod
    def load(cls, path, input_size, output_size, conv_layers, hidden_size, output_dimensions,
             decoder_layers=1, learnable_prior=False, **kwargs):
        model = cls(input_size, output_size, conv_layers, hidden_size, output_dimensions, decoder_layers,
                    learnable_prior, **kwargs)
        model.load_state_dict(torch.load(path))
        return model


if __name__ == '__main__':
    pass
