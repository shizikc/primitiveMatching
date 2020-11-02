# evaluate
import torch

from data.shapeNet import ShapeDiffDataset
from modules.configUtils import get_args
from modules.cuboid import get_cuboid_corner
from train import get_model
from utils.visualization import plot_pc_mayavi

params = get_args()
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model_path = "//home//coopers//models//1102_0949//model"
model, _ = get_model()
model.load_state_dict(torch.load(model_path, map_location=dev))
model.eval()

train_dataset = ShapeDiffDataset(params.train_path, params.bins, dev=dev, seed=0)
train_loader = torch.utils.data.DataLoader(train_dataset, params.batch_size, shuffle=False)

if __name__ == '__main__':
    total_acc = 0.
    for i, (x, d, h) in enumerate(train_loader):
        if i == 1:
            break

        pred, p, z = model(x)

        corners = get_cuboid_corner(params.bins)
        i = 3
        t = p[i] > params.threshold  # bins ^ 3
        t2 = h[i] > 0  # bins ^ 3
        c = corners[t, :]
        s = (t == t2).float() * t2.float()

        plot_pc_mayavi([c, d[1].cpu()],
                       colors=((0., 0., 0.), (.9, .9, .9), (1., 0., 0.)))
