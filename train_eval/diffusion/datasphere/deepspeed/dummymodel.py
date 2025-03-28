import torch
from torch import nn

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from vision_aided_loss import Discriminator
import deepspeed
from torch.utils.data import Dataset
from animator.diffusion.losses import CycleLoss, IdentityLoss
import os
import argparse

def get_trainable_params(model: nn.Module, print_num: bool = False) -> list:
    trainable_params = []
    count, tot_count = 0, 0
    for param in model.parameters():
        if param.requires_grad:
            trainable_params.append(param)
            count += param.numel()
        tot_count += param.numel()
    if print_num:
        print("Number of trainable parametes: {} of {}, {:.2%}".format(count, tot_count, count / tot_count))
    return trainable_params

class MyDataset(Dataset):
    def __init__(self,):
        super().__init__()
    
    def __len__(self,):
        return 4
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        return (torch.rand((3,256,256)) - 0.5) / 0.5, (torch.rand((3,256,256)) - 0.5) / 0.5

class DummyDataset(Dataset):
    def __init__(self,):
        super().__init__()
        self.nums = list(range(10))

    def __len__(self,):
        return len(self.nums)

    def __getitem__(self, index):
        return self.nums[index]
    

class DummyModel(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(3, 64, 3,padding=1),
                                    nn.Conv2d(64, 3, 3, padding=1),
                                    nn.Tanh())
    
    def forward(self, x):
        return self.layers(x)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--st', required = False) # path to S3 storage to store intermediate results
    parser.add_argument("--local_rank", type=int, default=-1, required=False)
    base_param = parser.parse_args()

    # Set s3 storage folder to store cache
    #os.environ['HF_HOME'] = base_param.st + '/cache/'
    #os.environ['TORCH_HOME'] = base_param.st + '/cache/'

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    device = torch.device("cuda", base_param.local_rank) if base_param.local_rank != -1 else torch.device("cpu")

    model1 = DummyModel()
    disc = Discriminator('clip', loss_type='multilevel_sigmoid')
    model2 = DummyModel()
    disc2 = Discriminator('clip', loss_type='multilevel_sigmoid')
    disc.cv_ensemble.models = nn.ModuleList(disc.cv_ensemble.models)
    disc2.cv_ensemble.models = nn.ModuleList(disc2.cv_ensemble.models)

    lpips = LearnedPerceptualImagePatchSimilarity("vgg", "mean", sync_on_compute=False)#.to("cuda")
    
    data = MyDataset()
    dummy_dataset = DummyDataset()
    cycle_loss = CycleLoss(lpips=lpips) #,device="cuda")
    idn_loss = IdentityLoss(lpips=lpips) #, device="cuda")

    modelG1, optG1, loader, _ =  deepspeed.initialize(model=model1, training_data=data, config="ds_config.json")
    modelG2, optG2, _, _ =  deepspeed.initialize(model=model2, config="ds_config.json")
    modelD1, optD1, test_loader, _ = deepspeed.initialize(model=disc,  training_data=dummy_dataset, config="ds_config_disc.json")
    modelD2, optD2, _, _ = deepspeed.initialize(model=disc2, config="ds_config_disc.json")

    for ind in range(3):
        print(ind)
        test_loader.data_sampler.set_epoch(ind)
        for x in test_loader:
            print(x)

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        fakeY = modelG1(x)
        cycle_x = modelG2(fakeY)

        fakeX = modelG2(y)
        cycle_y = modelG1(fakeX)

        modelD2.requires_grad_ = False
        modelD1.requires_grad_ = False

        adv_lossA = modelD1(fakeY, for_G=True).mean()
        adv_lossB = modelD2(fakeX, for_G=True).mean()

        cycle = cycle_loss(cycle_x, cycle_y, x, y)
        identity = idn_loss(modelG2(x), modelG1(y), x, y)

        loss = adv_lossA + adv_lossB + cycle + identity
        loss.backward()

        modelG1._backward_epilogue()
        modelG2._backward_epilogue()
        modelG1.step()
        modelG2.step()

        modelD2.requires_grad_ = True
        modelD1.requires_grad_ = True

        lossA_false = modelD1(fakeY.detach().clone(), for_real=False).mean()
        lossB_false = modelD2(fakeX.detach().clone(), for_real=False).mean()

        lossA_true = modelD1(y, for_real=True).mean()
        lossB_true = modelD2(x, for_real=True).mean()

        lossA = lossA_true + lossA_false
        lossB = lossB_true + lossB_false
        """
        lossA.backward()
        modelD1._backward_epilogue()
        modelD1.step()

        lossB.backward()
        modelD2._backward_epilogue()
        modelD2.step()"""

        modelD1.backward(lossA)
        modelD1.step()

        modelD2.backward(lossB)
        modelD2.step()
