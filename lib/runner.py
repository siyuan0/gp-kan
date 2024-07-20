import json
import pathlib

import torch.utils.data.dataloader
from tqdm import tqdm

import lib.activations as activations
from lib import utils
from lib.conv import GP_conv2D
from lib.gp_dist import GP_dist
from lib.layer import HYP_CTX
from lib.layer import LayerFused
from lib.model import GP_Model


class run_log:
    def __init__(self, log_path: pathlib.PosixPath) -> None:
        self.epoch = []
        self.lr = []
        self.train_obj = []
        self.val_obj = []
        self.val_acc = []
        if log_path.is_file():
            with open(log_path) as j:
                prev_log = json.load(j)
            self.epoch = prev_log["epoch"]
            self.lr = prev_log["lr"]
            self.train_obj = prev_log["train_obj"]
            self.val_obj = prev_log["val_obj"]
            self.val_acc = prev_log["val_acc"]

    def add_epoch(self, lr: float, train_obj: float, val_obj: float, val_acc: float):
        self.epoch.append(len(self.epoch))
        self.lr.append(lr)
        self.train_obj.append(train_obj)
        self.val_obj.append(val_obj)
        self.val_acc.append(val_acc)

    def save_log(self, path: pathlib.PosixPath, additional_info: dict):
        run_dict = {
            "epoch": self.epoch,
            "lr": self.lr,
            "train_obj": self.train_obj,
            "val_obj": self.val_obj,
            "val_acc": self.val_acc,
        }
        run_dict.update(additional_info)

        with open(path, "w") as outfile:
            json.dump(run_dict, outfile)


def pretrain_model(model: GP_Model, iters=10):
    # pretrain the model by maximising the log-likelihood of its inducing points
    # by tuning the lengthscale l and covariance scale s
    optimizerLBFGS = torch.optim.LBFGS(
        model.parameters(), lr=1e-3, line_search_fn="strong_wolfe"
    )

    def closureLBFGS():
        optimizerLBFGS.zero_grad()
        negloglik = -model.internal_loglik()
        negloglik.backward()
        return negloglik

    model.freeze_all_params()
    model.set_gp_hyp_trainable()

    for i in range(iters):
        optimizerLBFGS.step(closureLBFGS)  # type: ignore
        print(f"  refinement {i}: internal loglik {closureLBFGS()}")


def eval_model(
    data_loader: torch.utils.data.dataloader.DataLoader, model: GP_Model, gpu=True
):
    with torch.no_grad():
        correct_count = 0
        total_count = 0
        eval_val = 0.0
        for i, [features, labels] in enumerate(tqdm(data_loader, leave=False)):
            batch_size: int = features.shape[0]

            if gpu:
                features = features.cuda()
                labels = labels.cuda()

            model_out = model.forward(GP_dist.fromTensor(features))
            assert utils.same_shape(model_out.mean.shape, labels.shape)

            loglik = utils.log_likelihood(model_out.mean, model_out.var, labels)

            corrects = torch.argmax(model_out.mean, dim=1) == torch.argmax(
                labels, dim=1
            )

            eval_val += -torch.sum(loglik).item()
            correct_count += torch.count_nonzero(corrects).item()
            total_count += batch_size

    return (eval_val / total_count, correct_count / total_count)


def test_model(
    test_dataloader: torch.utils.data.DataLoader, model: GP_Model, use_gpu: bool = True
):
    # test a model
    print("testing the model on dataset size:", len(test_dataloader))
    test_negloglik, test_accuracy = eval_model(test_dataloader, model, gpu=use_gpu)
    print("test_negloglik", test_negloglik, "test accuracy", test_accuracy)


def train_model(
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    model: GP_Model,
    epochs: int = 30,
    learning_rate: float = 1e-3,
    global_jitter: float = 1e-1,
    baseline_jitter: float = 1e-1,
    pretrain: bool = False,
    use_gpu: bool = True,
    save_tmp: bool = True,
    log_path: pathlib.PosixPath = pathlib.PosixPath("run_log.json"),
):

    HYP_CTX.BASELINE_JITTER = baseline_jitter
    HYP_CTX.GLOBAL_JITTER = global_jitter

    if use_gpu:
        model.cuda()
    if pretrain:
        pretrain_model(model)

    optimizerSGD = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizerSGD, gamma=0.95)

    log = run_log(log_path)
    start_epoch = len(log.epoch)

    for epoch in range(start_epoch, start_epoch + epochs):
        train_loglik = 0.0
        total_count = 0
        for i, [train_features, train_labels] in enumerate(
            tqdm(train_dataloader, leave=False)
        ):
            optimizerSGD.zero_grad()
            batch_size = train_features.shape[0]
            # go through all data
            if use_gpu:
                train_features = train_features.cuda()
                train_labels = train_labels.cuda()
            model_out = model.predict(train_features)
            loglik = utils.log_likelihood(model_out.mean, model_out.var, train_labels)
            obj_value = -torch.sum(loglik) / batch_size
            obj_value.backward()
            optimizerSGD.step()
            with torch.no_grad():
                train_loglik += -torch.sum(loglik).item()
                total_count += batch_size

        val_negloglik, val_accuracy = eval_model(val_dataloader, model, gpu=use_gpu)

        print(
            f"epoch {epoch}, train negloglik: \t{train_loglik / total_count},    "
            f"\tval negloglik: {val_negloglik}    "
            f"\taccuracy: {val_accuracy}"
            f"\tlr: {scheduler.get_last_lr()}"
        )
        log.add_epoch(
            scheduler.get_last_lr()[0],  # lr
            train_loglik / total_count,  # train_obj
            val_negloglik,  # val_obj
            val_accuracy,  # val_acc
        )

        if epoch % 20 == 0:
            scheduler.step()

        # save tmp
        if save_tmp:
            torch.save(model.state_dict(), log_path.parent / "tmp.pt")

        # log progress
        log.save_log(
            log_path,
            {
                "train_set": len(train_dataloader),
                "val_set": len(val_dataloader),
            },
        )


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
