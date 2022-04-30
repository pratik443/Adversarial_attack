import torch
from classifier.models import *
from classifier.dataset import get_dataloader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import argparse


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F #233
import torch.optim as optim
from torchvision import datasets,models,transforms
from PIL import Image

from fgsm import FGSM




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Skecth_Classification')
    parser.add_argument('--backbone_name', type=str, default='Resnet', help='VGG / InceptionV3/ Resnet')
    parser.add_argument('--pool_method', type=str, default='AdaptiveAvgPool2d', help='AdaptiveMaxPool2d / AdaptiveAvgPool2d / AvgPool2d')
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--nThreads', type=int, default=8)
    parser.add_argument('--splitTrain', type=float, default=0.8)
    parser.add_argument('--learning_rate', type=float, default=0.0001)

    hp = parser.parse_args()
    dataloader_Train, dataloader_Test = get_dataloader(hp)
    print(hp)

    hp.FGSM_MNIST = {'epsilon': 0.2, 'order': np.inf, 'clip_max': None, 'clip_min': None}

    model = Sketch_Classification(hp)
    model.to(device)
    model.load_state_dict(torch.load('./classifier/model_best_TUBerlin_Bina.pth', map_location=device))
    model.eval()

    with torch.no_grad():
        True_Accuracy = model.evaluate(dataloader_Test)

    attack = FGSM(model, device='cuda')

    correct, correct_adv, correct_preserved = 0, 0, 0
    test_loss, test_loss_adv = 0, 0
    start_time = time.time()

    for i_batch, batch in enumerate(dataloader_Test):

        print(i_batch)
        images = batch['sketch_img'].to(device)
        images = (images > 0.4).float()

        for img, label in zip(images, batch['sketch_label']):
            target_label = torch.randint(0, 249, (1,))[0].to(device)
            img = img.unsqueeze(0)
            AdvExArray = attack.generate(img, target_label, **hp.FGSM_MNIST)
            # AdvExArray = AdvExArray.unsqueeze_(0).float()

            output = model(AdvExArray)
            test_loss_adv += model.loss(output, label.to(device).unsqueeze(0)).item()
            prediction_adv = output.argmax(dim=1, keepdim=True).to('cpu')
            correct_adv += prediction_adv.eq(label.view_as(prediction_adv)).sum().item()

            AdvExArray_Bina =  (AdvExArray > 0.4).float()
            save_image(torch.cat((img, AdvExArray, AdvExArray_Bina), dim=0), 'images.jpg')
            # print(torch.equal(AdvExArray_Bina, img))
            output = model(AdvExArray_Bina)
            test_loss += model.loss(output, label.to(device).unsqueeze(0)).item()
            prediction = output.argmax(dim=1, keepdim=True).to('cpu')
            correct_preserved += prediction.eq(label.view_as(prediction)).sum().item()

    #        Save images


    Adv_Accuracy = 100. * correct_adv / len(dataloader_Test.dataset)
    Adv_Accuracy_preserved = 100. * correct_preserved / len(dataloader_Test.dataset)
    print('True_Accuracy: {:.4f}, Adv_Accuracy, Adv_Accuracy_preserved: {},  Time_Takes: {}\n'.format(test_loss,
                                                                                                      Adv_Accuracy,
                                                                                                      Adv_Accuracy_preserved,
                                                                                                      (time.time() - start_time)))







