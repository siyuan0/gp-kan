import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from lib.activations import NormaliseGaussian
from lib.activations import ReshapeGaussian
from lib.conv import GP_conv2D
from lib.gp_dist import GP_dist
from lib.layer import LayerFused
from lib.model import GP_Model
from lib.utils import count_parameters
from lib.utils import log_likelihood


def imgTransform(img):
    t1 = ToTensor()(img)
    t2 = torch.transpose(t1, dim0=-3, dim1=-2)
    t3 = torch.transpose(t2, dim0=-2, dim1=-1)
    return t3


def oneHotEncoding(label):
    t1 = torch.tensor(label)
    t2 = torch.nn.functional.one_hot(t1, num_classes=10)  # pylint: disable=not-callable
    return t2


def main():
    MODEL_SAVE_PATH = "model.pt"
    BATCH_SIZE = 128

    model = GP_Model(
        [
            GP_conv2D(28, 28, 1, 1, kernel_size=8, stride=2, num_gp_pts=20),
            NormaliseGaussian(),
            GP_conv2D(11, 11, 1, 1, kernel_size=3, stride=1, num_gp_pts=20),
            NormaliseGaussian(),
            ReshapeGaussian([-1, 81]),
            LayerFused(81, 10, num_gp_pts=10),
            NormaliseGaussian(),
            LayerFused(10, 10, num_gp_pts=10),
            NormaliseGaussian(),
        ]
    )
    print(model)
    print(f"number of parameters: {count_parameters(model)}")

    if os.path.isfile(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))

    optimizerSGD = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    optimizerLBFGS = torch.optim.LBFGS(
        model.parameters(), lr=1e-3, line_search_fn="strong_wolfe"
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizerSGD, gamma=0.9)

    # get MNIST data
    training_data = datasets.MNIST(
        root="downloads/MNIST/train",
        train=True,
        download=True,
        transform=imgTransform,
        target_transform=oneHotEncoding,
    )

    test_data = datasets.MNIST(
        root="downloads/MNIST/test",
        train=False,
        download=True,
        transform=imgTransform,
        target_transform=oneHotEncoding,
    )

    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

    def closure(optimizer):
        optimizer.zero_grad()
        train_features, train_labels = next(iter(train_dataloader))
        batch_size = train_features.shape[0]

        # go through all data
        model_out = model.forward(GP_dist.fromTensor(train_features))
        loglik = log_likelihood(model_out.mean, model_out.var, train_labels)
        obj_val = -torch.sum(loglik) / batch_size
        obj_val.backward()
        return obj_val

    def closureSGD():
        return closure(optimizerSGD)

    # pylint: disable=unused-variable
    def closureLBFGS():
        optimizerLBFGS.zero_grad()
        negloglik = -model.internal_loglik()
        negloglik.backward()
        return negloglik

    def eval_model(train_loss):
        test_features, test_labels = next(iter(test_dataloader))
        batch_size = test_features.shape[0]
        model_out = model.forward(GP_dist.fromTensor(test_features))
        loglik = log_likelihood(model_out.mean, model_out.var, test_labels)
        eval_val = -torch.sum(loglik) / batch_size

        corrects = torch.argmax(model_out.mean, dim=1) == torch.argmax(
            test_labels, dim=1
        )
        accuracy = torch.count_nonzero(corrects) / corrects.numel()
        print(
            f"epoch {epoch}, train negloglik: \t{train_loss.detach().numpy()},    "
            f"\ttest negloglik: {eval_val.detach().numpy()}    "
            f"\taccuracy: {accuracy}"
            f"\tlr: {scheduler.get_lr()}"
        )

    # LBFGS step
    def refine_model():
        # with torch.no_grad():
        #     model.reset_gp_hyp()
        for i in range(10):
            model.freeze_all_params()
            model.set_gp_hyp_trainable()
            optimizerLBFGS.step(closureLBFGS)  # type: ignore
            print(f"  refinement {i}: internal loglik {closureLBFGS()}")

    # refine_model()
    for epoch in range(1000):
        # SGD step
        model.freeze_all_params()
        model.set_all_trainable()
        # model.set_gp_pts_trainable()
        optimizerSGD.step(closureSGD)  # type:ignore
        eval_model(closureSGD())
        if epoch % 50 == 0:
            scheduler.step()

        # save tmp
        torch.save(model.state_dict(), "tmp.pt")
        if epoch % 10 == 0:
            torch.save(model.state_dict(), "tmp10.pt")
        if epoch % 50 == 0:
            torch.save(model.state_dict(), "tmp50.pt")
        if epoch % 100 == 0:
            torch.save(model.state_dict(), "tmp100.pt")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)

    # print some training outcomes
    for layer in model.layers:
        print(layer)
        if isinstance(layer, LayerFused):
            print(f"log length scale {layer.l.detach().numpy()}")
            print(f"log covar scale {layer.s.detach().numpy()}")
            print(f"log jitter scale {layer.jitter.detach().numpy()}")
        if isinstance(layer, GP_conv2D):
            print(f"log length scale {layer.layerFused.l.detach().numpy()}")
            print(f"log covar scale {layer.layerFused.s.detach().numpy()}")
            print(f"log jitter scale {layer.layerFused.jitter.detach().numpy()}")


if __name__ == "__main__":
    main()
