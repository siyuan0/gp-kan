# import os
# from dataclasses import dataclass
# import torch
# from lib.model import Layer
# from lib.deprecated.neuron import GP_dist
# from lib.utils import log_likelihood
# @dataclass
# class dataset:
#     training_input: torch.Tensor
#     training_label: torch.Tensor
#     test_input: torch.Tensor
#     test_label: torch.Tensor
#     def random_subset(self, size):
#         idx = torch.randperm(self.training_input.shape[0])
#         chosen_idx = idx[:size]
#         return self.training_input[chosen_idx], self.training_label[chosen_idx]
# def true_func(x1, x2):
#     # print(f"label int {torch.sin(torch.pi * x1)} , {x2**2}")
#     return torch.exp(torch.sin(torch.pi * x1) + x2**2)
# def get_data():
#     range_x1 = [-0.5, 0.5]
#     range_x2 = [-0.5, 0.5]
#     num_of_train = 5000
#     num_of_test = 100
#     inputs = torch.rand(num_of_train + num_of_test, 2) * torch.tensor(
#         [range_x1[1] - range_x1[0], range_x2[1] - range_x2[0]]
#     ) + torch.tensor([range_x1[0], range_x2[0]])
#     labels = true_func(inputs[:, 0], inputs[:, 1])
#     return dataset(inputs[:num_of_train], labels[:num_of_train], inputs[num_of_train:], labels[num_of_train:])
# def main():
#     MODEL_SAVE_PATH = "model.pt"
#     model = Layer(input_size=2, neurons_per_element=5, output_size=1)
#     if os.path.isfile(MODEL_SAVE_PATH):
#         model.load_state_dict(torch.load(MODEL_SAVE_PATH))
#     model.save_fig("display.png")
#     # for p_name, p in model.named_parameters():
#     #     print(p_name, p)
#     optimizerSGD = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
#     optimizerLBFGS = torch.optim.LBFGS(model.parameters(), lr=7e-6, line_search_fn="strong_wolfe")
#     scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizerSGD, gamma=0.9)
#     data = get_data()
#     def closure(optimizer):
#         optimizer.zero_grad()
#         obj_val = torch.zeros(1)
#         # get random subset of data
#         train_inputs, train_labels = data.random_subset(200)
#         # go through all data
#         for train_idx in range(train_inputs.shape[0]):
#             train_input = train_inputs[train_idx, :]
#             train_label = train_labels[train_idx]
#             # print(f"label: {train_label}, {true_func(train_input[0], train_input[1])}")
#             model_out = model.forward(
#                 GP_dist(train_input, 1e-6 * torch.ones(train_input.shape[-1]))
#             )
#             obj_val -= log_likelihood(model_out.mean, model_out.var, train_label)
#         obj_val.backward()
#         return obj_val
#     def closureSGD():
#         return closure(optimizerSGD)
#     def closureLBFGS():
#         optimizerLBFGS.zero_grad()
#         negloglik = - model.neuron_loglik()
#         negloglik.backward()
#         return negloglik
#     def eval_model(train_loss):
#         eval_val = torch.zeros(1, requires_grad=False)
#         for test_idx in range(data.test_input.shape[0]):
#             test_input = data.test_input[test_idx, :]
#             test_label = data.test_label[test_idx]
#             model_out = model.forward(
#                 GP_dist(test_input, 1e-6 * torch.zeros(test_input.shape[-1]))
#             )
#             eval_val -= log_likelihood(model_out.mean, model_out.var, test_label)
#         print(
#             f"epoch {epoch}, train negloglik: \t{train_loss.detach().numpy()[0]},    "
#             f"\ttest negloglik: {eval_val.detach().numpy()[0]}"
#             # f"\t jitter: {[neuron.jitter.detach().numpy()[0] for neuron in model.group1_neurons + model.group2_neurons]}"
#             # f"\t l: {[neuron.l.detach().numpy()[0] for neuron in model.group1_neurons + model.group2_neurons]}"
#             # f"\t s: {[neuron.s.detach().numpy()[0] for neuron in model.group1_neurons + model.group2_neurons]}"
#         )
#     for epoch in range(100):
#         # LBFGS step
#         for i in range(5):
#             model.freeze_all_params()
#             model.set_gp_hyp_trainable()
#             optimizerLBFGS.step(closureLBFGS)
#             print(f"refinement {i}: internal loglik {closureLBFGS()}")
#         # LBFGS step
#         for i in range(5):
#             model.freeze_all_params()
#             model.set_gp_hyp_trainable()
#             optimizerSGD.step(closureSGD)
#             print(f"refinement {i}: external loglik {closureSGD()}")
#         # SGD step
#         for i in range(5):
#             model.freeze_all_params()
#             # if epoch % 10 == 9 :
#             # model.set_all_trainable()
#             # else:
#             model.set_gp_pts_trainable()
#             optimizerSGD.step(closureSGD)
#             eval_model(closureSGD())
#         torch.save(model.state_dict(), "tmp.pt")
#         model.save_fig("display.png")
#         scheduler.step()
#     torch.save(model.state_dict(), MODEL_SAVE_PATH)
#     # evaluate
#     eval_data = get_data()
#     for test_input, test_label in zip(eval_data.test_input, eval_data.test_label):
#         prediction = model.predict(test_input)
#         print(
#             f"evaluation: input {test_input.numpy()}    "
#             f"\t output mean {prediction.mean.detach().numpy()}   "
#             f"\t output var  {prediction.var.detach().numpy()}    "
#             f"\t true label  {true_func(test_input[0], test_input[1]).numpy()}"
#         )
# if __name__ == "__main__":
#     main()
