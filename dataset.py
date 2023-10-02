import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms


class Train_dataset(Dataset):

    def __init__(self, train_data, crop_size, scaling_factor):

        self.train_data = np.load(train_data)
        self.crop_size = int(crop_size)
        self.scaling_factor = int(scaling_factor)

        self.pre_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(self.crop_size)
            # transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomVerticalFlip(p=0.5)
        ])

        self.input_transform = transforms.Compose([
            transforms.GaussianBlur((3, 3), sigma=1),
            transforms.Resize(self.crop_size // self.scaling_factor),
        ])

    def __len__(self):

        return self.train_data.shape[0]

    def __getitem__(self, index):

        data_train = self.train_data[index, :, :, :]
        data_train = self.pre_trans(data_train)

        lr_img = self.input_transform(data_train)
        hr_img = data_train

        return lr_img, hr_img


class Test_dataset(Dataset):

    def __init__(self, train_data, scale):

        self.train_data = np.load(train_data)
        self.crop_size = np.array(self.train_data.shape[-3:-1])

        self.crop_size1 = self.crop_size // scale
        self.crop_size = tuple(self.crop_size1 * scale)
        self.crop_size1 = tuple(self.crop_size1)

        self.pre_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(self.crop_size)
        ])

        self.input_transform = transforms.Compose([
            transforms.GaussianBlur((3, 3), sigma=1),
            transforms.Resize(self.crop_size1)
        ])

    def __len__(self):

        return self.train_data.shape[0]

    def __getitem__(self, index):

        data_train = self.train_data[index, :, :, :]
        data_train = self.pre_trans(data_train)

        lr_img = self.input_transform(data_train)
        hr_img = data_train

        return lr_img, hr_img


if __name__ == '__main__':

    test_path = '/'
    trainset = Test_dataset(test_path, scale=4)
    print(trainset[10][0].shape)