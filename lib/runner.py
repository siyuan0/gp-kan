import json
import pathlib

import torch.utils.data.dataloader

from lib import utils
from lib.gp_dist import GP_dist
from lib.layer import HYP_CTX
from lib.model import GP_Model


class run_log:
    def __init__(self, log_path: pathlib.PosixPath, reuse_log: bool = False) -> None:
        self.epoch = []
        self.lr = []
        self.train_obj = []
        self.val_obj = []
        self.val_acc = []
        if log_path.is_file() and reuse_log:
            with open(log_path, encoding="utf-8") as j:
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

        with open(path, "w", encoding="utf-8") as outfile:
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
        print(f"  pretrain {i}: internal loglik {closureLBFGS().item():.4f}")


def eval_model(
    data_loader: torch.utils.data.dataloader.DataLoader, model: GP_Model, gpu=True
):
    with torch.no_grad():
        correct_count = 0
        total_count = 0
        eval_val = 0.0
        for _, [features, labels] in enumerate(utils.progress_bar(data_loader)):
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
    print("testing the model ...")
    if use_gpu:
        model.cuda()
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
    reuse_log: bool = False,
    log_path: pathlib.PosixPath = pathlib.PosixPath("run_log.json"),
):

    HYP_CTX.BASELINE_JITTER = baseline_jitter
    HYP_CTX.GLOBAL_JITTER = global_jitter

    if use_gpu:
        model.cuda()
    if pretrain:
        pretrain_model(model)

    model.freeze_all_params()
    model.set_all_trainable()

    optimizerSGD = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizerSGD, gamma=0.9)

    log = run_log(log_path, reuse_log)
    start_epoch = len(log.epoch)

    for epoch in range(start_epoch, start_epoch + epochs):
        train_loglik = 0.0
        total_count = 0
        for _, [train_features, train_labels] in enumerate(
            utils.progress_bar(train_dataloader)
        ):
            optimizerSGD.zero_grad()
            batch_size = train_features.shape[0]
            # go through all data
            if use_gpu:
                train_features = train_features.cuda()
                train_labels = train_labels.cuda()
            model_out = model.predict(train_features)
            assert utils.same_shape(model_out.mean.shape, train_labels.shape)
            loglik = utils.log_likelihood(model_out.mean, model_out.var, train_labels)
            obj_value = -torch.sum(loglik) / batch_size
            obj_value.backward()
            optimizerSGD.step()
            with torch.no_grad():
                train_loglik += -torch.sum(loglik).item()
                total_count += batch_size

        val_negloglik, val_accuracy = eval_model(val_dataloader, model, gpu=use_gpu)

        print(
            f"epoch {str(epoch).ljust(3)}, train negloglik: {train_loglik / total_count:.5f},    "
            f"val negloglik: {val_negloglik:.5f}    "
            f"accuracy: {val_accuracy:.4f} "
            f"lr: {scheduler.get_last_lr()}"
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
