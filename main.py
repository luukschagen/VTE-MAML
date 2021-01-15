import datasets
import torch
import torch.nn.functional as F
import os
from datetime import datetime
from tensorboardX import SummaryWriter
from MAML import *
import math
import matplotlib.pyplot as plt
import argparse
import json


def main(args):

    with open(args.config, 'r') as file:
        config = json.load(file)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    network = Learner(config)

    directory = "./logs/{}_{}/{}_{}".format(args.maml_type, args.benchmark,
                                            datetime.now().strftime("%Y%m%d"), args.logname)

    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(directory+"/settings.json", 'w') as file:
        json.dump(vars(args), file)

    validationset = None

    if args.benchmark == 'regression':

        trainingset = datasets.MultiModal(args.task_num, args.k_shot, args.k_query, args.modes)
        if args.validation_tasks:
            validationset = datasets.MultiModal(args.validation_tasks, args.k_shot, args.validation_samples, args.modes)
    elif args.benchmark == 'classification':
        trainingset = datasets.MultiModalClassification(args.task_num, args.k_shot, args.k_query, args.modes)
        if args.validation_tasks:
            validationset = datasets.MultiModalClassification(args.validation_tasks, args.k_shot,
                                                              args.validation_samples, args.modes)
    elif args.benchmark == 'imagenet':

        trainingset = datasets.MiniImagenet(args.task_num, args.k_shot, args.k_query, args.n_way)
        if args.validation_tasks:
            validationset = datasets.MiniImagenet(args.validation_tasks, args.k_shot, args.validation_samples,
                                                  args.n_way, mode='validation')

    if args.loss == 'mse':
        loss = F.mse_loss
    elif args.loss == 'bce':
        loss = F.binary_cross_entropy_with_logits
    elif args.loss == 'crossentropy':
        loss = F.cross_entropy
    elif args.loss == 'gaussian_likelihood':
        loss = log_likelihood
    else:
        if args.benchmark == 'regression':
            loss = F.mse_loss
        elif args.benchmark == 'classification':
            loss = F.binary_cross_entropy_with_logits
        elif args.benchmark == 'imagenet':
            loss = F.cross_entropy
        else:
            loss = F.mse_loss

    writer = SummaryWriter(directory, flush_secs=30)

    metalearner = None

    if args.maml_type == 'maml':
        metalearner = MAML(network, args.inner_lr, args.outer_lr, args.inner_updates, args.validation_updates,
                           device, loss, args.first_order, writer)

    elif args.maml_type == 'mmaml':


        if args.benchmark != 'imagenet':
            if args.hidden_size is None:
                args.hidden_size = 40
            input_size = trainingset[0][0].shape[1]
            output_size = trainingset[0][2].shape[1]
            modulation = Modulation_network(input_size, output_size, args.modulation_layers, args.hidden_size,
                                            args.output_dimensions, not args.unidirectional)

        else:
            if args.hidden_size is None:
                args.hidden_size = 128
            input_size = trainingset[0][0].shape[1:]
            output_size = trainingset[0][2].shape[0]
            modulation = ConvModulation_network(input_size, output_size, args.convolutional_layers, args.hidden_size,
                                                args.output_dimensions, args.decoder_layers)

        metalearner = MMAML(network, modulation, args.inner_lr, args.outer_lr, args.inner_updates,
                            args.validation_updates, device, loss, args.first_order, writer)

    elif args.maml_type == 'vte-maml':

        if args.benchmark != 'imagenet':
            if args.hidden_size is None:
                args.hidden_size = 40
            input_size = trainingset[0][0].shape[1]
            output_size = trainingset[0][2].shape[1]
            modulation = Variational_task_encoder(input_size, output_size, args.modulation_layers, args.hidden_size,
                                                  args.output_dimensions, not args.unidirectional, args.decoder_layers,
                                                  args.learnable_prior)

        else:
            if args.hidden_size is None:
                args.hidden_size = 128
            input_size = trainingset[0][0].shape[1:]
            output_size = trainingset[0][2].shape[0]
            modulation = ConvVariational_task_encoder(input_size, output_size, args.convolutional_layers,
                                                      args.hidden_size, args.output_dimensions, args.decoder_layers,
                                                      args.learnable_prior)

        metalearner = VTE_MAML(network, modulation, args.inner_lr, args.outer_lr, args.inner_updates,
                               args.validation_updates, device, loss, args.first_order, writer)

    metalearner.fit(trainingset, validationset, epochs=args.epochs, save=directory,
                    meta_batch_size=args.meta_batch_size, accuracy=args.report_accuracy)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("benchmark", choices=['regression', 'classification', 'imagenet'])
    parser.add_argument("maml_type", choices=['maml', 'mmaml', 'vte-maml'])

    # General meta-learning requirements
    parser.add_argument("--config", default="model_configs/regression.json",
                        help="path to configuration for neural network")
    parser.add_argument("--task_num", type=int, default=12500, help="number of tasks per epoch")
    parser.add_argument("--k_shot", type=int, default=5, help="number of instances per class for training")
    parser.add_argument("--k_query", type=int, default=15, help="number of instances per class for evaluation "
                                                                "during meta-training")
    parser.add_argument("--validation_tasks", type=int, default=1000, help="number of tasks in validationset, "
                                                                           "0 for no validation")
    parser.add_argument("--validation_samples", type=int, default=100, help="number of samples in per validationtask")

    parser.add_argument("--first_order", action="store_true", help="Use first order approximation of MAML")
    parser.add_argument("--inner_lr", type=float, default=1e-4, help="Learning rate for the inner update steps of MAML")
    parser.add_argument("--outer_lr", type=float, default=1e-4, help="Learning rate for the meta-training process")
    parser.add_argument("--inner_updates", type=int, default=5, help="Number of inner updates during training")
    parser.add_argument("--validation_updates", type=int, default=5, help="Number of inner updates during validation")
    parser.add_argument("--loss", choices=['mse', 'bce', 'crossentropy', 'gaussian_likelihood'],
                        help="Loss function to be used")
    parser.add_argument("--device", choices=['cuda', 'cpu'], help="Whether to use gpu(cuda) or cpu. "
                                                                  "Defaults to gpu if available")

    parser.add_argument("--logname", type=str, default="test", help="Suffix for logging folder, defaults to 'test'")
    parser.add_argument("--epochs", type=int, default=700, help="Number of epochs for training")
    parser.add_argument("--meta_batch_size", type=int, default=125, help="Number of tasks per batch")
    parser.add_argument("--report_accuracy", action="store_true", help="Report the accuracy")

    # Regression arguments
    parser.add_argument("--modes", type=int, default=5, help="number of modes in regression and classification tasks")

    # Imagenet arguments
    parser.add_argument("--n_way", type=int, default=5, help="number of classes per imagenet task")

    # Modulation arguments
    parser.add_argument("--modulation_layers", type=int, default=2, help='number of layers in encoding LSTM')
    parser.add_argument("--convolutional_layers", nargs='+', type=int, default=[32, 64, 128, 256])
    parser.add_argument("--hidden_size", type=int, default=None,
                        help='hidden units in encoding LSTM and task embedding, default is 40 for '
                             'regression/classification, 128 imagenet')
    parser.add_argument("--unidirectional", action="store_true", help="Use unidirectional LSTM for task encoding "
                                                                      "in MMAML and VTE-MAML")
    parser.add_argument("--output_dimensions", nargs='+', type=int, default=[100, 100, 100],
                        help="Output shapes of modulation parameters")
    parser.add_argument("--decoder_layers", type=int, default=1, help="Number of layers to use for decoder")

    parser.add_argument("--learnable_prior", action="store_true", help="Use trainable prior for "
                                                                       "Variational Task Encoder")

    args = parser.parse_args()

    main(args)
