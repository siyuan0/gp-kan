import json

import matplotlib.pyplot as plt

runs = {
    "11k": "mnist_conv/run_log.json",
    "24k": "mnist_conv3/run_log.json",
    "33k": "mnist_conv4/run_log.json",
    "43k": "mnist_conv9/run_log.json",
    "80k": "mnist_conv12b/run_log.json",
}

# set font sizes
SIZE = 26
plt.rc("font", size=SIZE)  # controls default text sizes
plt.rc("axes", titlesize=SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=SIZE)  # legend fontsize

fig, (ax_acc, ax_obj) = plt.subplots(1, 2)

NUM_EPOCHS = 45

for label, log_path in runs.items():
    with open(log_path, encoding="utf-8") as j:
        run_log = json.load(j)

    epoch = run_log["epoch"]
    val_acc = run_log["val_acc"]
    train_negloglik = run_log["train_avg_negloglik"]
    if len(train_negloglik) > len(epoch):
        train_negloglik = train_negloglik[: 2 * NUM_EPOCHS : 2]
        epoch = epoch[:NUM_EPOCHS]
        val_acc = val_acc[:NUM_EPOCHS]
    else:
        train_negloglik = train_negloglik[:NUM_EPOCHS]
        epoch = epoch[:NUM_EPOCHS]
        val_acc = val_acc[:NUM_EPOCHS]

    ax_acc.plot(epoch, val_acc, label=label, linewidth=3)
    ax_obj.plot(epoch, train_negloglik, label=label, linewidth=3)


ax_acc.set_title("validation accuracy")
ax_acc.set_xlabel("epoch")
ax_obj.set_title("training negative loglikelihood")
ax_obj.set_xlabel("epoch")
ax_obj.legend()
fig.set_figwidth(16)
fig.set_figheight(8)
fig.tight_layout()
fig.savefig("results.png")
