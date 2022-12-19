# %%
# set base path to "Shoes Dataset" usig Path from pathlib
# create a dataframe with the image path and the shoe type
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
import os
print("Current working directory: ", os.getcwd())
run_anyway = False
if __name__ == '__main__' or run_anyway:
    base_path = Path('/Shoes Dataset')
    if base_path.exists():
        print("The base path exists")
    else:
        print("The base path does not exist")
        print("Running as jupyter notebook")
        base_path  = Path("../../Shoes_Dataset")
        print("The new base path is: ", base_path)
    print("The base path is: ", base_path)
    # %%
    # make path for Test Train Valid
    test_path = base_path / 'Test'
    train_path = base_path / 'Train'
    valid_path = base_path / 'Valid'

    # Get Shoe types in Train
    shoe_types = [x for x in train_path.iterdir() if x.is_dir()]
    # %%
    # for each show type print how many images there are
    for shoe_type in shoe_types:
        print(shoe_type.name, len(list(shoe_type.iterdir())))
    # %% 
    # plot 16 images in a gr
    for shoe_type in shoe_types:
        print(shoe_type.name)
        images = list(shoe_type.iterdir())
        fig, ax = plt.subplots(4, 4, figsize=(10, 10))
        for i in range(4):
            for j in range(4):
                img = plt.imread(str(images[i * 4 + j]))
                ax[i, j].imshow(img)
                ax[i, j].axis('off')
        plt.show()
    # %%
    data = []
    for shoe_type in shoe_types:
        images = list(shoe_type.iterdir())
        for image in images:
            data.append((str(image), shoe_type.name))
    df = pd.DataFrame(data, columns=['image', 'shoe_type'])
    # %%
    # save the dataframe to a csv file
    shoe_type_id = {shoe_type.name: i for i, shoe_type in enumerate(shoe_types)}
    df['shoe_type_id'] = df['shoe_type'].map(shoe_type_id)
    df.to_csv(base_path /'shoes.csv', index=False)
    df.head()
    # %%
    # read the csv file
    df = pd.read_csv(base_path/'shoes.csv')
    # %%
    # check the number of images and shoe types
    print(df.shape)
    print(df['shoe_type'].unique())
    # %%
    # check the number of images for each shoe type
    print(df['shoe_type'].value_counts())
    # %%
    # plot a histogram for the number of images for each shoe type
    df['shoe_type'].value_counts().plot(kind='bar')
    # %%
    df.head()
# %%
# Make a machine learning model to classify the shoe types
# use py lightning to make the model
# https://pytorch-lightning.readthedocs.io/en/latest/
# 1. Install pytorch lightning
# pip install pytorch-lightning
# 2. Import the libraries
import pytorch_lightning as pl

# 3. make a dataset class using lightning
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
# import lightening data module
from pytorch_lightning import LightningDataModule
# %%
# Make a torch dataset class
class ShoeDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path, shoe_type, shoe_class_id = self.df.iloc[idx]
        image = Image.open(image_path)
        image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, shoe_class_id

# make a datamodule class
class ShoeDataModule(LightningDataModule):
    def __init__(self, df, batch_size=32, filter_with_shoe_type_id=None):
        super().__init__()
        self.df = df
        if filter_with_shoe_type_id is not None:
            print("Filtering with shoe type id: ", filter_with_shoe_type_id)
            print("Before filtering: ", self.df.shape)
            if isinstance(filter_with_shoe_type_id, int):
                filter_with_shoe_type_id = [filter_with_shoe_type_id]
            self.df = self.df[self.df['shoe_type_id'].isin(filter_with_shoe_type_id)]
            print("After filtering: ", self.df.shape)
        else:
            print("Not filtering with shoe type id")
        self.batch_size = batch_size
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.transform = transforms.Compose([
            # transforms.Resize((224, 224)),
            transforms.Resize((256, 256)),
            
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean,
                                 std=self.std)
        ])

    def setup(self, stage=None):
        self.train_df = self.df.sample(frac=0.8, random_state=42)
        self.val_df = self.df.drop(self.train_df.index)

    def train_dataloader(self):
        return DataLoader(ShoeDataset(self.train_df, self.transform),
                          batch_size=self.batch_size,
                          num_workers=4)

    def val_dataloader(self):
        return DataLoader(ShoeDataset(self.val_df, self.transform),
                          batch_size=self.batch_size,
                          num_workers=4)
# %%
# use timm to make a model with pytorch lightning
# 1. Install timm
# pip install timm
# 2. Import the libraries
import timm
import torch
import torch.nn.functional as F

# use wandb to log the model
# 1. Install wandb
# pip install wandb
# 2. Import the libraries
import wandb
from pytorch_lightning.loggers import WandbLogger

# %%
# make a pytorch lightning model
class ShoeClassifier(pl.LightningModule):
    def __init__(self, num_classes, class2name_map, data_mean , data_std , learning_rate=1e-3):
        super().__init__()
        self.model = timm.create_model('resnet50', pretrained=True)
        # add a dropout layer
        self.model.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
             torch.nn.Linear(self.model.fc.in_features, num_classes)
        )
            
        # self.model.fc = torch.nn.Linear(self.model.fc.in_features, num_classes)
        self.class2name_map = class2name_map
        self.learning_rate = learning_rate
        self.data_mean = torch.tensor(data_mean) 
        self.data_std = torch.tensor(data_std) 

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        accuracy = (y_hat.argmax(dim=1) == y).float().mean()
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr= self.learning_rate)
    
    # write a function to write predictions to wandb
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        accuracy = (y_hat.argmax(dim=1) == y).float().mean()
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_accuracy', accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # log the predictions to wandb
        # on every 10th batch
        if batch_idx % 10 == 0:
            # make a grid of images, use matplotlib
            n = 16
            idxs = torch.randint(0, len(x), (n,))
            x = x[idxs].cpu().permute(0, 2, 3, 1)
            # apply inverse normalization
            x = x * self.data_std + self.data_mean
            y = y[idxs].cpu()
            y_hat = y_hat[idxs].cpu()
            fig, axs = plt.subplots(4, 4, figsize=(16, 16))
            for i, ax in enumerate(axs.flatten()):
                ax.imshow(x[i])
                ax.set_title(f'Pred: {self.class2name_map[y_hat[i].argmax().item()]}, Label: {self.class2name_map[y[i].item()]}')
                ax.axis('off')
            plt.tight_layout()
            # log the figure to wandb
            self.logger.experiment.log({'predictions': wandb.Image(fig)})
            plt.close(fig)
# %%
# make an instance of the datamodule
if __name__ == '__main__':
    dm = ShoeDataModule(df, batch_size=32)
    # %%
    # make an instance of the model
    shoe_type_id_inverse = {v: k for k, v in shoe_type_id.items()}
    model = ShoeClassifier(len(shoe_types), shoe_type_id_inverse, data_mean = dm.mean, data_std = dm.std, learning_rate=1e-3)
    # %%
    # train the model
    wandb_logger = WandbLogger(project='shoe-classifier')
    trainer = pl.Trainer(max_epochs=10, gpus=1, logger=wandb_logger)
    trainer.fit(model, dm)
    # # %%
    # model.model
    # # %%
    # model1 = timm.create_model('resnet50', pretrained=True)
    # # %%
    # model1.fc
    # # %%

    # %%
