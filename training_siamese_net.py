import torchvision.transforms as transforms
import torch
import torchvision
import torchvision.datasets as dset
import torch.nn.functional as F
import os
from torch import optim
from torch.autograd import Variable
from ai_pytorch.models.siamese_network import SiameseNetwork
from ai_pytorch.dataset.siamese_dataset import SiameseDataset
from ai_pytorch.losses.constrativeloss import ContrastiveLoss
from ai_pytorch.utils.helper_functions import HelperFunctions
from ai_pytorch.utils.training import ModelTraining
from torch.utils.data import DataLoader
import argparse


# ===== MAIN =====
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('input_dir', help='Input directory of images ()')
    parser.add_argument('output_image', help='Output image name')
    parser.add_argument('--method', help='Stacking method ORB (faster) or ECC (more precise)')
    parser.add_argument('--show', help='Show result image', action='store_true')
    args = parser.parse_args()

    image_folder = args.input_dir
    if not os.path.exists(image_folder):
        print("ERROR {} not found!".format(image_folder))
        exit()

    data_transforms = {
        'training': transforms.Compose([
            transforms.Resize((100, 100)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5]),
        ]),
        'testing': transforms.Compose([
            transforms.Resize((100, 100)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5]),
        ]),
    }

    batch_size = {
        'training': 8,
        'testing': 1,
    }

    data_dir = './data/faces/'
    image_datasets = {
        x: SiameseDataset(
            image_folder_dataset=dset.ImageFolder(
                os.path.join(data_dir, x + '/')
            ),
            transform=data_transforms[x],
            should_invert=False,
        )
        for x in ['training', 'testing']
    }
    dataloaders = {
        x: DataLoader(
            image_datasets[x],
            batch_size=batch_size[x],
            shuffle=True,
            num_workers=8,
        )
        for x in ['training', 'testing']
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ['training', 'testing']}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = SiameseNetwork()

    for param in model.parameters():
        param.requires_grad = True

    model.to(device)  # We can fine-tune on GPU if available

    criterion = ContrastiveLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.0005
    )

    # Decay LR by a factor of 0.3 every several epochs
    exp_lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=7,
        gamma=0.3,
    )

    train_model = ModelTraining(
        model,
        criterion,
        optimizer,
        exp_lr_scheduler,
        num_epochs=25,
        device=device,
    )

    model_tuned = train_model(
        dataloaders
    )

    dataiter = iter(dataloaders['testing'])
    x0, _, _ = next(dataiter)

    for i in range(10):
        _, x1, label2 = next(dataiter)
        concatenated = torch.cat((x0, x1), 0)

        output1, output2 = model_tuned(
            Variable(x0).to(device),
            Variable(x1).to(device)
        )
        euclidean_distance = F.pairwise_distance(output1, output2)
        HelperFunctions.imshow(
            torchvision.utils.make_grid(concatenated),
            'Dissimilarity: {:.2f}'.format(euclidean_distance.item())
        )
