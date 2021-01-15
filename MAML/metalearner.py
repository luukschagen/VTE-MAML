import torch
import torch.nn.functional as F
import torch.utils.data as data
from torch.nn.utils.clip_grad import clip_grad_norm_
from tqdm import tqdm


class MAML(object):

    def __init__(self, model, inner_lr, outer_lr, inner_updates, validation_updates,
                 device=None, loss=F.mse_loss, first_order=False, writer=None, inner_grad_clip=10, **kwargs):
        """
        :type model: MAML.learner.Learner

        """
        super(MAML, self).__init__()
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_updates = inner_updates
        self.validation_updates = validation_updates
        self.loss = loss
        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.outer_lr)
        self.first_order = first_order
        self.inner_grad_clip = inner_grad_clip
        self.writer = writer
        self.iteration = 0
        self.workers = 8
        if device:
            self.device = device
            self.to_device(device)
        else:
            self.device = torch.device('cpu')
            self.to_device(self.device)

    def fit(self, dataset, validationset=None, meta_batch_size=10, epochs=50, verbose=True, save=False, accuracy=False):

        loader = data.DataLoader(dataset, batch_size=meta_batch_size, shuffle=True, num_workers=self.workers, pin_memory=True)

        self.iteration = 0

        if validationset:
            validation_loss = self.meta_evaluate(validationset, accuracy)
            tqdm.write("Validation loss: {}".format(validation_loss))

        for epoch in range(epochs):
            loss_total = 0
            epoch_length = len(dataset) / meta_batch_size
            batch_iterator = tqdm(enumerate(loader), total=epoch_length)
            count = 0
            for count, batch in batch_iterator:
                self.iteration += 1
                for i, tensor in enumerate(batch):
                    batch[i] = tensor.to(self.device)
                loss = self.batch_adapt(batch)
                batch_iterator.set_description("Epoch: {} -- Loss: {}".format(epoch + 1, float(loss)))
                self.meta_update(loss)
                loss_total += float(loss)
                self.write("Trainingset/Query_loss", float(loss))

            epoch_loss = loss_total / (count + 1)

            if verbose:
                tqdm.write("Average meta loss: {}".format(float(epoch_loss)))

            if save:
                self.save("Epoch_{}".format(epoch), save)

            if validationset:
                validation_loss = self.meta_evaluate(validationset, accuracy)
                tqdm.write("Validation loss: {}".format(validation_loss))

    def batch_adapt(self, batch):
        x_s, x_q, y_s, y_q = batch
        meta_loss = torch.zeros(1, device=self.device)

        for task in range(len(x_s)):
            task_meta_loss = self.task_adapt(x_s[task], x_q[task], y_s[task], y_q[task])
            meta_loss = meta_loss + task_meta_loss
        meta_loss = meta_loss / len(x_s)
        return meta_loss

    def meta_update(self, meta_loss):
        self.optim.zero_grad()
        meta_loss.backward()
        clip_grad_norm_(self.model.parameters(), 2.0)
        self.optim.step()

    def task_adapt(self, x_support, x_query, y_support, y_query):
        phi = self.model.parameters()
        for i in range(self.inner_updates):
            phi = self.inner_update(x_support, y_support, phi)
        meta_loss = self.evaluate(x_query, y_query, phi)
        return meta_loss

    def inner_update(self, x_support, y_support, phi, first_order=None):
        if first_order is None:
            first_order = self.first_order
        loss = self.evaluate(x_support, y_support, phi)
        phi = self.inner_step(loss, phi, first_order)
        return phi

    def inner_step(self, loss, phi, first_order):
        grad = torch.autograd.grad(loss, phi, create_graph=(not first_order))
        phi = list(map(lambda x: x[1] - self.inner_lr * x[0].clamp(min=-self.inner_grad_clip, max=self.inner_grad_clip),
                       zip(grad, phi)))
        return phi

    def meta_evaluate(self, validationset, accuracy=False, updates=None):
        validation_score = 0
        support_validation_score = 0
        loader = data.DataLoader(validationset, batch_size=1, num_workers=self.workers, pin_memory=True)
        for x_support, x_test, y_support, y_test in tqdm(loader):
            x_support = x_support.to(self.device).view(x_support.shape[1:])
            x_test = x_test.to(self.device).view(x_test.shape[1:])
            y_support = y_support.to(self.device).view(y_support.shape[1:])
            y_test = y_test.to(self.device).view(y_test.shape[1:])
            phi = self.finetune(x_support, y_support, updates=updates)
            with torch.no_grad():
                support_validation_score += float(self.evaluate(x_support, y_support, phi, accuracy))
                validation_score += float(self.evaluate(x_test, y_test, phi, accuracy))
        mean_validation_score = validation_score/len(validationset)
        mean_support_validation_score = support_validation_score / len(validationset)
        if accuracy:
            self.write("Validationset/Post_adaptation/Support_accuracy", mean_support_validation_score)
            self.write("Validationset/Post_adaptation/Query_accuracy", mean_validation_score)
        else:
            self.write("Validationset/Post_adaptation/Support_loss", mean_support_validation_score)
            self.write("Validationset/Post_adaptation/Query_loss", mean_validation_score)

        return mean_validation_score

    def finetune(self, x_support, y_support, updates=None):
        """
        Used to finetune to a specific task for inference or testing. Don't use for training, as it does not store the
        inner gradients.
        """
        if updates is None:
            updates = self.validation_updates
        phi = self.model.parameters()
        for i in range(updates):
            phi = self.inner_update(x_support, y_support, phi, first_order=True)
        return phi

    def evaluate(self, x, y, phi=None, accuracy=False):
        prediction = self.model.forward(x, vars=phi)
        if accuracy:
            return self.get_accuracy(prediction, y)
        else:
            return self.loss(prediction, y)

    def predict(self, x, phi=None):
        return self.model.forward(x, vars=phi)

    def get_accuracy(self, prediction, y):
        if prediction.shape[-1] == 1:
            classes = torch.round(torch.sigmoid(prediction))
        else:
            classes = torch.argmax(prediction, 1)
        accuracy = float(torch.eq(classes, y).sum()) / len(y)
        return accuracy

    def write(self, name, value):
        if self.writer:
            self.writer.add_scalar(name, value, self.iteration)

    def to_device(self, device):
        self.model.to(device)

    def save(self, name, path):
        self.model.save(name, path)


class MMAML(MAML):

    def __init__(self, model, modulation_model, inner_lr, outer_lr, inner_updates, validation_updates,
                 device=None, loss=F.mse_loss, first_order=False, writer=None, **kwargs):

        self.modulation_model = modulation_model
        super(MMAML, self).__init__(model, inner_lr, outer_lr, inner_updates,
                                    validation_updates, device, loss, first_order, writer=writer)
        self.optim_embed = torch.optim.Adam(self.modulation_model.parameters(), lr=self.outer_lr)

    def meta_update(self, meta_loss):
        self.optim.zero_grad()
        self.optim_embed.zero_grad()
        meta_loss.backward()
        clip_grad_norm_(self.model.parameters(), 2.0)
        clip_grad_norm_(self.modulation_model.parameters(), 2.0)
        self.optim.step()
        self.optim_embed.step()

    def task_adapt(self, x_support, x_query, y_support, y_query):
        phi = self.model.parameters()
        embeddings = self.modulation_model(x_support, y_support)
        for i in range(self.inner_updates):
            phi = self.inner_update(x_support, y_support, phi, embeddings)
        meta_loss = self.evaluate(x_query, y_query, phi, embeddings)
        return meta_loss

    def inner_update(self, x_support, y_support, phi, embeddings=None, first_order=None):
        if first_order is None:
            first_order = self.first_order
        loss = self.evaluate(x_support, y_support, phi, embeddings)
        phi = self.inner_step(loss, phi, first_order)
        return phi

    def meta_evaluate(self, validationset, accuracy=False, updates=None, pre_adapt_score=True):
        pre_validation_score = 0
        pre_support_score = 0
        validation_score = 0
        support_score = 0
        loader = data.DataLoader(validationset, batch_size=1, num_workers=self.workers, pin_memory=True)
        for x_support, x_test, y_support, y_test in tqdm(loader):

            x_support = x_support.to(self.device).view(x_support.shape[1:])
            x_test = x_test.to(self.device).view(x_test.shape[1:])
            y_support = y_support.to(self.device).view(y_support.shape[1:])
            y_test = y_test.to(self.device).view(y_test.shape[1:])
            phi, embeddings = self.finetune(x_support, y_support, updates=updates)

            with torch.no_grad():
                if pre_adapt_score:
                    pre_validation_score += float(self.evaluate(x_test, y_test,
                                                                embeddings=embeddings, accuracy=accuracy))
                    pre_support_score += float(self.evaluate(x_support, y_support,
                                                             embeddings=embeddings, accuracy=accuracy))
                validation_score += float(self.evaluate(x_test, y_test, phi, embeddings, accuracy))
                support_score += float(self.evaluate(x_support, y_support, phi, embeddings, accuracy))

        pre_validation_score = pre_validation_score/len(validationset)
        pre_support_score = pre_support_score/len(validationset)
        validation_score = validation_score/len(validationset)
        support_score = support_score/len(validationset)
        if pre_adapt_score:
            if accuracy:
                self.write("Validationset/Pre_adaptation/Support_accuracy", pre_support_score)
                self.write("Validationset/Pre_adaptation/Query_accuracy", pre_validation_score)
            else:
                self.write("Validationset/Pre_adaptation/Support_loss", pre_support_score)
                self.write("Validationset/Pre_adaption/Query_loss", pre_validation_score)

        if accuracy:
            self.write("Validationset/Post_adaptation/Support_accuracy", support_score)
            self.write("Validationset/Post_adaptation/Query_accuracy", validation_score)
        else:
            self.write("Validationset/Post_adaptation/Support_loss", support_score)
            self.write("Validationset/Post_adaptation/Query_loss", validation_score)

        if pre_adapt_score:
            return pre_validation_score, validation_score
        else:
            return validation_score

    def finetune(self, x_support, y_support, updates=None):
        """
        Used to finetune to a specific task for inference or testing. Don't use for training, as it does not store the
        inner gradients.
        """
        if updates is None:
            updates = self.validation_updates
        phi = self.model.parameters()
        embeddings = self.modulation_model(x_support, y_support)
        for i in range(updates):
            phi = self.inner_update(x_support, y_support, phi, embeddings, first_order=True)
        return phi, embeddings

    def evaluate(self, x, y, phi=None, embeddings=None, accuracy=False):
        prediction = self.model.forward(x, vars=phi, embeddings=embeddings)
        if accuracy:
            return self.get_accuracy(prediction, y)
        else:
            return self.loss(prediction, y)

    def predict(self, x, phi=None, embeddings=None):
        return self.model.forward(x, vars=phi, embeddings=embeddings)

    def to_device(self, device):
        super(MMAML, self).to_device(device)
        self.modulation_model.to(device)

    def save(self, name, path):
        self.model.save(name+"_model", path)
        self.modulation_model.save(name+"_modulation", path)


class VTE_MAML(MMAML):
    def __init__(self, model, modulation_model, inner_lr, outer_lr, inner_updates, validation_updates,
                 device=None, loss=F.mse_loss, first_order=False, writer=None, **kwargs):
        super(VTE_MAML, self).__init__(model, modulation_model, inner_lr, outer_lr, inner_updates,
                                       validation_updates, device, loss, first_order, writer)

        self.loss = lambda pred, y: loss(pred, y, reduction='sum')

    def evaluate(self, x, y, phi=None, embeddings=None, accuracy=False):
        prediction = self.model.forward(x, vars=phi, embeddings=embeddings)
        if accuracy:
            return self.get_accuracy(prediction, y)
        else:
            likelihood_loss = self.loss(prediction, y)
            kld = self.modulation_model.kld_loss
            return likelihood_loss + kld

    def meta_evaluate(self, validationset, accuracy=False, updates=None, pre_adapt_score=True):
        result = super(VTE_MAML, self).meta_evaluate(validationset, accuracy, updates, pre_adapt_score)
        KL_total = 0
        for task in validationset:
            self.modulation_model.forward(task[0].to(self.device), task[2].to(self.device))
            KL_total += float(self.modulation_model.kld_loss)
            self.modulation_model.reset_kld()
        self.write("Validationset/KL-divergence", KL_total/len(validationset))
        return result


if __name__ == '__main__':
    pass
