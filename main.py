import os
from dataclasses import dataclass

import torch

from lib.model import Layer
from lib.neuron import GP_dist
from lib.utils import log_likelihood


@dataclass
class dataset:
    training_input: torch.Tensor
    training_label: torch.Tensor
    test_input: torch.Tensor
    test_label: torch.Tensor


def true_func(x1, x2):
    return torch.exp(torch.sin(torch.pi * x1) + x2**2)


def get_data():
    range_x1 = [-2, 2]
    range_x2 = [-2, 2]
    training_input = torch.rand(100, 2) * torch.tensor(
        [range_x1[1] - range_x1[0], range_x2[1] - range_x2[0]]
    ) + torch.tensor([range_x1[0], range_x2[0]])
    training_labels = true_func(training_input[:, 0], training_input[:, 1])
    print(training_labels)

    test_input = torch.rand(10, 2) * torch.tensor(
        [range_x1[1] - range_x1[0], range_x2[1] - range_x2[0]]
    ) + torch.tensor([range_x1[0], range_x2[0]])
    test_labels = true_func(training_input[:, 0], training_input[:, 1])

    return dataset(training_input, training_labels, test_input, test_labels)


def main():
    MODEL_SAVE_PATH = "model.pt"

    model = Layer(input_size=2, neurons_per_element=5, output_size=1)

    if os.path.isfile(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))

    model.freeze_all_params()
    model.set_gp_pts_trainable()

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-6, momentum=0.9)

    data = get_data()

    for epoch in range(100):
        # model.zero_grad()
        obj_val = torch.zeros(1)
        # go through all data
        for train_idx in range(data.training_input.shape[0]):
            train_input = data.training_input[train_idx, :]
            train_label = data.training_label[train_idx]
            model_out = model.forward(
                GP_dist(train_input, 1e-6 * torch.ones(train_input.shape[-1]))
            )
            obj_val -= log_likelihood(model_out.mean, model_out.var, train_label)

        # update parameters
        optimizer.zero_grad()
        obj_val.backward()
        optimizer.step()

        # evaluate model's accuracy
        eval_val = torch.zeros(1, requires_grad=False)
        for test_idx in range(data.test_input.shape[0]):
            test_input = data.test_input[test_idx, :]
            test_label = data.test_label[test_idx]
            model_out = model.forward(
                GP_dist(test_input, 1e-6 * torch.zeros(test_input.shape[-1]))
            )
            eval_val -= log_likelihood(model_out.mean, model_out.var, test_label)

        print(
            f"epoch {epoch}, train negloglik: \t{obj_val.detach().numpy()[0]},    "
            f"\ttest negloglik: {eval_val.detach().numpy()[0]}"
        )

        torch.save(model.state_dict(), "tmp.pt")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)

    # evaluate
    eval_data = get_data()
    for test_input, test_label in zip(eval_data.test_input, eval_data.test_label):
        prediction = model.predict(test_input)
        print(
            f"evaluation: input {test_input.numpy()}    "
            f"\t output mean {prediction.mean.detach().numpy()}   "
            f"\t output var  {prediction.var.detach().numpy()}    "
            f"\t true label  {test_label.numpy()}"
        )


if __name__ == "__main__":
    main()
