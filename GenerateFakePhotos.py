import torch
from torchvision.utils import save_image
from model import VAE
from torchvision import transforms
from PIL import Image
from VAELoss import VAELoss

checkpoint = torch.load('./AKIEC_result/best_model.pth.tar')
image_size = 256
model = VAE(image_size)
criterion = VAELoss()

# top_params = [p for p in model._modules['fc4'].parameters()]
# base_params = []
# for key, _ in (model._modules.items()):
#     if (key != 'fc4'):
#         base_params += [p for p in model._modules[key].parameters()]

model.load_state_dict(checkpoint['state_dict'])
# device = torch.device("cuda")
device = torch.device("cpu")

if __name__ == '__main__':
    with torch.no_grad():
        sample = torch.randn(64, 100, 1, 1).to(device)
        sample = model.decode(sample).cpu()
        save_image(sample.view(64, 3, 256, 256),
                   'GeneratedPhtos.png')