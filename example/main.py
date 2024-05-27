import os
from dataclasses import dataclass

import torch

from lib.gp_dist import GP_dist
from lib.model import simple_model
from lib.utils import log_likelihood


@dataclass
class dataset:
    training_input: torch.Tensor
    training_label: torch.Tensor
    test_input: torch.Tensor
    test_label: torch.Tensor

    def random_training_subset(self, size):
        """
        return input shape (N, model_I)
               label shape (N, model_O)
            where N = size
        """
        idx = torch.randperm(self.training_input.shape[0])
        chosen_idx = idx[:size]
        return self.training_input[chosen_idx], self.training_label[chosen_idx]


def true_func(x1, x2):
    # print(f"label int {torch.sin(torch.pi * x1)} , {x2**2}")
    return torch.exp(torch.sin(torch.pi * x1) + x2**2)


def get_data():
    range_x1 = [-0.5, 0.5]
    range_x2 = [-0.5, 0.5]
    num_of_train = 5000
    num_of_test = 100
    inputs = torch.rand(num_of_train + num_of_test, 2) * torch.tensor(
        [range_x1[1] - range_x1[0], range_x2[1] - range_x2[0]]
    ) + torch.tensor([range_x1[0], range_x2[0]])
    labels = true_func(inputs[:, 0], inputs[:, 1])

    return dataset(
        inputs[:num_of_train],
        labels[:num_of_train],
        inputs[num_of_train:],
        labels[num_of_train:],
    )


def main():
    MODEL_SAVE_PATH = "model.pt"

    model = simple_model([2, 2, 1])
    print(model)

    if os.path.isfile(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))

    model.save_fig("display/")

    optimizerSGD = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9)
    optimizerLBFGS = torch.optim.LBFGS(
        model.parameters(), lr=7e-6, line_search_fn="strong_wolfe"
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizerSGD, gamma=0.95)

    data = get_data()

    def closure(optimizer):
        optimizer.zero_grad()
        BATCH_SIZE = 300  # N
        # get random subset of data
        train_inputs, train_labels = data.random_training_subset(BATCH_SIZE)
        train_labels_reshaped = train_labels.reshape(BATCH_SIZE, 1)

        # go through all data
        model_out = model.forward(
            GP_dist(train_inputs, 1e-6 * torch.ones_like(train_inputs))
        )
        loglik = log_likelihood(model_out.mean, model_out.var, train_labels_reshaped)
        obj_val = -torch.sum(loglik)
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
        test_inputs = data.test_input
        test_labels = data.test_label.reshape(-1, 1)
        model_out = model.forward(
            GP_dist(test_inputs, 1e-6 * torch.zeros_like(test_inputs))
        )
        loglik = log_likelihood(model_out.mean, model_out.var, test_labels)
        eval_val = -torch.sum(loglik)

        print(
            f"epoch {epoch}, train negloglik: \t{train_loss.detach().numpy()},    "
            f"\ttest negloglik: {eval_val.detach().numpy()}"
        )

    # # LBFGS step
    # with torch.no_grad():
    #     model.reset_gp_hyp()
    # for i in range(10):
    #     model.freeze_all_params()
    #     model.set_gp_hyp_trainable()
    #     optimizerLBFGS.step(closureLBFGS)
    #     print(f"  refinement {i}: internal loglik {closureLBFGS()}")

    for epoch in range(1000):

        torch.save(model.state_dict(), "tmp.pt")
        model.save_fig("display/")

        # SGD step
        model.freeze_all_params()
        model.set_all_trainable()
        optimizerSGD.step(closureSGD)  # type:ignore
        eval_model(closureSGD())
        if epoch % 10 == 0:
            scheduler.step()

    torch.save(model.state_dict(), MODEL_SAVE_PATH)

    # evaluate
    eval_data = get_data()
    for test_input, test_label in zip(eval_data.test_input, eval_data.test_label):
        prediction = model.predict(test_input.reshape(1, -1))
        print(
            f"evaluation: input {test_input.numpy()}    "
            f"\t output mean {prediction.mean.detach().numpy()}   "
            f"\t output var  {prediction.var.detach().numpy()}    "
            f"\t true label  {true_func(test_input[0], test_input[1]).numpy()}"
        )


if __name__ == "__main__":
    main()
