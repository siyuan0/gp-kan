import argparse
import json
import pathlib
import sys

sys.path.append("../../../")  # bring to project top level


import torch.utils.data
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.transforms import v2
from tqdm import tqdm

import lib.activations as activations
from lib.conv import GP_conv2D
from lib.gp_dist import GP_dist
from lib.layer import HYP_CTX
from lib.layer import LayerFused
from lib.model import GP_Model
from lib.utils import count_parameters
from lib.utils import log_likelihood

USE_GPU = True


def true_func(x1, x2):
    return torch.exp(torch.sin(torch.pi * x1) + x2**2)


class myDataSet(torch.utils.data.Dataset):
    def __init__(self, size):
        RANGE_LOW = -0.5
        RANGE_HIGH = 0.5
        self.size = size
        self.inputs = torch.rand(size, 2) * (RANGE_HIGH - RANGE_LOW) + RANGE_LOW
        self.labels = true_func(self.inputs[:, 0], self.inputs[:, 1])

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]


def main(
    run_dir: pathlib.PosixPath,
    batch_size: int,
    epochs: int,
    model: GP_Model,
    global_jitter: float,
    baseline_jitter: float,
    learning_rate: float,
):

    MODEL_SAVE_PATH = run_dir / "model.pt"
    HYP_CTX.BASELINE_JITTER = baseline_jitter
    HYP_CTX.GLOBAL_JITTER = global_jitter

    print(model)
    print(f"number of parameters: {count_parameters(model)}")
    reload_model = MODEL_SAVE_PATH.is_file()
    if reload_model:
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))

    # try GPU
    if USE_GPU:
        model.cuda()

    optimizerSGD = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    optimizerLBFGS = torch.optim.LBFGS(
        model.parameters(), lr=1e-3, line_search_fn="strong_wolfe"
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizerSGD, gamma=0.9)

    # get data
    training_data = myDataSet(30000)
    val_data = myDataSet(1000)
    test_data = myDataSet(1000)

    print(
        f"training on {len(training_data)}, "
        f"validating on {len(val_data)}, "
        f"testing on {len(test_data)}"
    )

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    # pylint: disable=unused-variable
    def closureLBFGS():
        optimizerLBFGS.zero_grad()
        negloglik = -model.internal_loglik()
        negloglik.backward()
        return negloglik

    def eval_model(data_loader):
        se_total = 0
        total_count = 0
        eval_val = 0
        with torch.no_grad():
            for i, [test_features, test_labels] in enumerate(
                tqdm(data_loader, leave=False)
            ):
                batch_size = test_features.shape[0]

                if USE_GPU:
                    test_features = test_features.cuda()
                    test_labels = test_labels.cuda()

                model_out = model.forward(GP_dist.fromTensor(test_features))
                loglik = log_likelihood(
                    model_out.mean,
                    model_out.var,
                    test_labels.reshape(model_out.mean.shape),
                )

                se = (test_labels.reshape(model_out.mean.shape) - model_out.mean) ** 2

                eval_val += -torch.sum(loglik).item()
                se_total += torch.sum(se).item()
                total_count += batch_size
        return eval_val / total_count, se_total / total_count

    # LBFGS step
    def refine_model():
        for i in range(10):
            model.freeze_all_params()
            model.set_gp_hyp_trainable()
            optimizerLBFGS.step(closureLBFGS)  # type: ignore
            print(f"  refinement {i}: internal loglik {closureLBFGS()}")

    # if not reload_model:
    #     refine_model()

    # SGD step
    model.freeze_all_params()
    # model.set_all_trainable()
    # model.set_gp_hyp_trainable()
    model.set_gp_pts_trainable()

    class run_log:
        epoch = []
        lr = []
        train_avg_negloglik = []
        val_avg_negloglik = []
        val_acc = []

    if reload_model:
        with open(run_dir / "run_log.json") as j:
            prev_log = json.load(j)
        run_log.epoch = prev_log["epoch"]
        run_log.lr = prev_log["lr"]
        run_log.train_avg_negloglik = prev_log["train_avg_negloglik"]
        run_log.val_avg_negloglik = prev_log["val_avg_negloglik"]
        run_log.val_acc = prev_log["val_acc"]
        start_epoch = run_log.epoch[-1] + 1
    else:
        start_epoch = 0

    for epoch in range(start_epoch, start_epoch + epochs):
        train_loglik = 0
        total_count = 0
        # print some model neuron
        model.layers[0].save_fig("layer-0.png")
        model.layers[1].save_fig("layer-1.png")
        for i, [train_features, train_labels] in enumerate(
            tqdm(train_dataloader, leave=False)
        ):
            optimizerSGD.zero_grad()
            batch_size = train_features.shape[0]
            # go through all data
            if USE_GPU:
                train_features = train_features.cuda()
                train_labels = train_labels.cuda()
            model_out = model.predict(train_features)
            loglik = log_likelihood(
                model_out.mean,
                model_out.var,
                train_labels.reshape(model_out.mean.shape),
            )
            obj_val = -torch.sum(loglik) / batch_size

            # add penalty on spread of inducing points
            weight = 1000
            obj_val += weight * torch.sum(
                (torch.var(model.layers[0].get_z(), dim=-1) - 0.5) ** 2
            )
            obj_val += weight * torch.sum(
                (torch.var(model.layers[1].get_z(), dim=-1) - 0.5) ** 2
            )

            obj_val.backward()
            optimizerSGD.step()
            with torch.no_grad():
                train_loglik += -torch.sum(loglik).item()
                total_count += batch_size

        val_negloglik, val_mse = eval_model(val_dataloader)

        print(
            f"epoch {epoch}, train negloglik: \t{train_loglik / total_count},    "
            f"\tval negloglik: {val_negloglik}    "
            f"\tmse: {val_mse}"
            # f"\tlr: {scheduler.get_last_lr()}"
            # f"\tin_noise: {model.get_input_noise()}"
        )
        run_log.epoch.append(epoch)
        run_log.lr.append(scheduler.get_last_lr()[0])
        run_log.train_avg_negloglik.append(train_loglik / total_count)
        run_log.val_avg_negloglik.append(val_negloglik)
        run_log.val_acc.append(val_mse)

        if epoch % 20 == 0:
            scheduler.step()

        # save tmp
        torch.save(model.state_dict(), run_dir / "tmp.pt")

        # log progress
        run_dict = {
            "epoch": run_log.epoch,
            "lr": run_log.lr,
            "train_set": len(train_dataloader),
            "val_set": len(val_dataloader),
            "test_set": len(test_dataloader),
            "train_avg_negloglik": run_log.train_avg_negloglik,
            "val_avg_negloglik": run_log.val_avg_negloglik,
            "val_acc": run_log.val_acc,
        }

        with open(run_dir / "run_log.json", "w") as outfile:
            json.dump(run_dict, outfile)

    torch.save(model.state_dict(), MODEL_SAVE_PATH)

    # test the model accuracy
    print("testing the model")
    test_negloglik, test_mse = eval_model(test_dataloader)
    print("test_negloglik", test_negloglik, "test mse", test_mse)

    # examples
    test_inputs, test_labels = next(iter(test_dataloader))
    print(
        "input",
        test_inputs[0:4],
        "model",
        model.predict(test_inputs[0:4].cuda()),
        "label",
        test_labels[0:4],
    )

    # -- one example --
    import matplotlib.pyplot as plt
    import numpy as np

    # set font sizes
    SIZE = 26
    plt.rc("font", size=SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SIZE)  # legend fontsize

    def dsp_g(mean, var, range_min, range_max, ax, yshift=0):
        print(mean, var)
        step = (range_max - range_min) / 100
        x_range = np.arange(range_min, range_max, step)
        g_pdf = (
            1 / np.sqrt(2 * np.pi * var) * np.exp(-((x_range - mean) ** 2) / (2 * var))
            + yshift
        )
        ax.fill_between(x_range, g_pdf, yshift)

    print("input noise", model.get_input_noise())
    test_input0 = test_inputs[0].reshape(1, 2).cuda()
    test_input0_g = GP_dist(
        test_input0, (model.get_input_noise() ** 2) * torch.ones_like(test_input0)
    )
    layer0 = model.layers[0]
    layer1 = model.layers[1]
    t0 = layer0.forward(test_input0_g)
    out = layer1.forward(t0)
    # dsp inputs
    fig, axes = plt.subplots(2, 2, squeeze=False)

    model.layers[0].plot_neuron(axes[0, 0], 0, 0)
    dsp_g(
        test_input0_g.mean.cpu().detach().numpy().reshape(-1)[0],
        test_input0_g.var.cpu().detach().numpy().reshape(-1)[0],
        -2,
        2,
        axes[0, 0],
        yshift=-4,
    )
    model.layers[0].plot_neuron(axes[0, 1], 1, 0)
    dsp_g(
        test_input0_g.mean.cpu().detach().numpy().reshape(-1)[1],
        test_input0_g.var.cpu().detach().numpy().reshape(-1)[1],
        -2,
        2,
        axes[0, 1],
        yshift=-4,
    )
    model.layers[1].plot_neuron(axes[1, 0], 0, 0)
    dsp_g(
        t0.mean.cpu().detach().numpy().reshape(-1)[0],
        t0.var.cpu().detach().numpy().reshape(-1)[0],
        -2,
        2,
        axes[1, 0],
        yshift=-4,
    )

    dsp_g(
        out.mean.cpu().detach().numpy().reshape(-1)[0],
        out.var.cpu().detach().numpy().reshape(-1)[0],
        -5,
        5,
        axes[1, 1],
        yshift=0,
    )

    fig.set_figwidth(10)
    fig.set_figheight(10)
    fig.savefig("layers_inputs.png")

    plt.close(fig)
    SIZE = 15
    plt.rc("font", size=SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SIZE)  # legend fontsize
    epoch_max = 10
    plt.plot(
        run_log.epoch[:epoch_max],
        run_log.train_avg_negloglik[: 2 * epoch_max : 2],
        label="training negative log likelihood",
        linewidth=3,
    )
    plt.plot(
        run_log.epoch[:epoch_max],
        run_log.val_acc[:epoch_max],
        label="validation mean squared-error",
        linewidth=3,
    )
    plt.xlabel("epochs")
    plt.legend()
    plt.savefig("training.png")
    # -- end example --

    run_dict = {
        "epoch": run_log.epoch,
        "lr": run_log.lr,
        "train_set": len(train_dataloader),
        "val_set": len(val_dataloader),
        "test_set": len(test_dataloader),
        "train_avg_negloglik": run_log.train_avg_negloglik,
        "val_avg_negloglik": run_log.val_avg_negloglik,
        "val_acc": run_log.val_acc,
        "test_avg_negloglik": test_negloglik,
        "test_acc": test_mse,
    }

    with open(run_dir / "run_log.json", "w") as outfile:
        json.dump(run_dict, outfile)


# pylint: disable-all
def make_model(layer_infos: list):
    layer_list = []
    for layer_info in layer_infos:
        if "GP_conv2D" in layer_info:
            att = layer_info["GP_conv2D"]
            layer = GP_conv2D(
                *att["IH_IW_IC_OC"],
                kernel_size=att["kernel_size"],
                stride=att["stride"],
                num_gp_pts=att["num_gp_pts"],
                use_double_layer=att["use_double_layer"],
            )
        elif "LayerFused" in layer_info:
            att = layer_info["LayerFused"]
            layer = LayerFused(att["in"], att["out"], att["num_gp_pts"])
        elif "NormaliseGaussian" in layer_info:
            layer = activations.NormaliseGaussian()
        elif "ReshapeGaussian" in layer_info:
            layer = activations.ReshapeGaussian(layer_info["ReshapeGaussian"])
        elif "AvgPool2DGaussian" in layer_info:
            att = layer_info["AvgPool2DGaussian"]
            layer = activations.AvgPool2DGaussian(att["kernel_size"], att["stride"])
        else:
            raise RuntimeError
        layer_list.append(layer)
    return GP_Model(layer_list)


# pylint: enable-all

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_file", help="path to json file containing the run details")
    parser.add_argument("--no-gpu", action="store_true", help="disable GPU usage")
    args = parser.parse_args()

    USE_GPU = not args.no_gpu
    run_path = pathlib.PosixPath(args.run_file).absolute()
    run_dir = run_path.parent
    assert run_path.is_file()

    with open(run_path) as j:
        run_details = json.load(j)

    model = make_model(run_details["MODEL"])
    main(
        run_dir=run_dir,
        batch_size=run_details["BATCH_SIZE"],
        epochs=run_details["EPOCHS"],
        model=model,
        global_jitter=run_details["GLOBAL_JITTER"],
        baseline_jitter=run_details["BASELINE_JITTER"],
        learning_rate=run_details["LEARNING_RATE"],
    )
