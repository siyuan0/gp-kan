from lib import model as gplib
from lib import runner
from lib.dataset import MNIST

model = gplib.GP_Model(
    [
        gplib.GP_conv2D(28, 28, 1, 4, kernel_size=5, stride=2, num_gp_pts=10),
        gplib.NormaliseGaussian(),
        gplib.GP_conv2D(12, 12, 4, 12, kernel_size=5, stride=2, num_gp_pts=10),
        gplib.NormaliseGaussian(),
        gplib.ReshapeGaussian([-1, 192]),
        gplib.LayerFused(192, 10, num_gp_pts=10),
        gplib.NormaliseGaussian(),
        gplib.LayerFused(10, 10, num_gp_pts=10),
    ],
    saveload_path="model.pth",
)

train_dataloader, val_dataloader, test_dataloader = MNIST.get_train_val_test(
    batch_size=64, train=55000, val=5000, test=10000
)
model.print_model()
runner.train_model(
    train_dataloader, val_dataloader, model, epochs=30, learning_rate=1e-3, use_gpu=True
)
model.save_model()
runner.test_model(test_dataloader, model, use_gpu=True)
