import argparse
import json
import pathlib
import sys

TOP_LEVEL = "../../"
sys.path.append(TOP_LEVEL)  # bring to project top level


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


class DataSetWrapper(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


transformations = v2.Compose(
    [
        v2.RandomResizedCrop(size=(28, 28), scale=(0.8, 1.0), antialias=True),
    ]
)


def imgTransform_train(img):
    t1 = ToTensor()(img)
    t2 = transformations(t1)
    t3 = torch.transpose(t2, dim0=-3, dim1=-2)
    t4 = torch.transpose(t3, dim0=-2, dim1=-1)
    return t4


def imgTransform_test(img):
    t1 = ToTensor()(img)
    t2 = torch.transpose(t1, dim0=-3, dim1=-2)
    t3 = torch.transpose(t2, dim0=-2, dim1=-1)
    return t3


def oneHotEncoding(label):
    t1 = torch.tensor(label)
    t2 = torch.nn.functional.one_hot(t1, num_classes=10)  # pylint: disable=not-callable
    return t2


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
    print(f"number of parameters: {count_parameters(model, only_require_grad=True)}")
    reload_model = MODEL_SAVE_PATH.is_file()
    if reload_model:
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))

    # try GPU
    if USE_GPU:
        model.cuda()

    # print some model neuron
    # model.layers[5].save_fig("plot1.png")

    optimizerSGD = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    optimizerLBFGS = torch.optim.LBFGS(
        model.parameters(), lr=1e-3, line_search_fn="strong_wolfe"
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizerSGD, gamma=0.9)

    # get MNIST data
    training_data_all = datasets.MNIST(
        root=TOP_LEVEL + "downloads/MNIST/train",
        train=True,
        download=True,
        transform=None,
        target_transform=oneHotEncoding,
    )

    test_data_all = datasets.MNIST(
        root=TOP_LEVEL + "downloads/MNIST/test",
        train=False,
        download=True,
        transform=imgTransform_test,
        target_transform=oneHotEncoding,
    )

    training_data = DataSetWrapper(
        torch.utils.data.Subset(training_data_all, range(0, 55000)),
        transform=imgTransform_train,
    )
    val_data = DataSetWrapper(
        torch.utils.data.Subset(training_data_all, range(55000, 60000)),
        transform=imgTransform_test,
    )
    test_data = test_data_all

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
        with torch.no_grad():
            correct_count = 0
            total_count = 0
            eval_val = 0
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

                corrects = torch.argmax(model_out.mean, dim=1) == torch.argmax(
                    test_labels, dim=1
                )

                eval_val += -torch.sum(loglik).item()
                correct_count += torch.count_nonzero(corrects).item()
                total_count += batch_size
        return eval_val / total_count, correct_count / total_count

    # LBFGS step
    def refine_model():
        for i in range(10):
            model.freeze_all_params()
            model.set_gp_hyp_trainable()
            optimizerLBFGS.step(closureLBFGS)  # type: ignore
            print(f"  refinement {i}: internal loglik {closureLBFGS()}")

    if not reload_model:
        refine_model()

    # SGD step
    model.freeze_all_params()
    model.set_all_trainable()
    # print(model.get_input_noise())
    # exit()

    # profile memory usage
    # with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof:
    #     model.forward(GP_dist.fromTensor(train_features))
    # print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
    # exit()
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
        run_log.val_avg_negloglik = prev_log["train_avg_negloglik"]
        run_log.val_acc = prev_log["val_acc"]
        start_epoch = run_log.epoch[-1] + 1
    else:
        start_epoch = 0

    for epoch in range(start_epoch, start_epoch + epochs):
        train_loglik = 0
        total_count = 0
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
            loglik = log_likelihood(model_out.mean, model_out.var, train_labels)
            obj_val = -torch.sum(loglik) / batch_size
            obj_val.backward()
            optimizerSGD.step()
            with torch.no_grad():
                train_loglik += -torch.sum(loglik).item()
                total_count += batch_size

        val_negloglik, val_accuracy = eval_model(val_dataloader)

        print(
            f"epoch {epoch}, train negloglik: \t{train_loglik / total_count},    "
            f"\tval negloglik: {val_negloglik}    "
            f"\taccuracy: {val_accuracy}"
            f"\tlr: {scheduler.get_last_lr()}"
        )
        run_log.epoch.append(epoch)
        run_log.lr.append(scheduler.get_last_lr()[0])
        run_log.train_avg_negloglik.append(train_loglik / total_count)
        run_log.val_avg_negloglik.append(val_negloglik)
        run_log.val_acc.append(val_accuracy)

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
    test_negloglik, test_accuracy = eval_model(test_dataloader)
    print("test_negloglik", test_negloglik, "test accuracy", test_accuracy)

    # example tests
    print("example output")
    inputs, labels = next(iter(test_dataloader))
    if USE_GPU:
        inputs = inputs.cuda()
        labels = labels.cuda()
    predictions = model.predict(inputs)
    import matplotlib.pyplot as plt
    import numpy as np

    np.set_printoptions(suppress=True, precision=3)
    fig, axes = plt.subplots(1, 3)
    for i in range(3):
        axes[i].imshow(inputs[i].cpu())
        pred = predictions.mean[i].detach().cpu().numpy()
        pred_max = np.argmax(pred)
        axes[i].set_title(
            f"model prediction: {pred_max}\n raw model output:\n{pred[:5]}\n{pred[5:]}",
            ha="left",
            loc="left",
        )

    fig.set_figwidth(16)
    fig.set_figheight(4)
    fig.tight_layout(w_pad=4)
    fig.savefig("input_img.png")

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
        "test_acc": test_accuracy,
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
