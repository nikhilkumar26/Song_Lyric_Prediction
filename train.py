## Importing the libraries

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader

import torch.nn.functional as F
import torchvision.models as models

## Adding the path of docformer to system path
import sys
sys.path.append('/home/ec2-user/docformer/src/docformer/')

## Importing the functions from the DocFormer Repo
from dataset import create_features
from modeling import DocFormerEncoder,ResNetFeatureExtractor,DocFormerEmbeddings,LanguageFeatureExtractor
from transformers import BertTokenizerFast


## Hyperparameters

seed = 42
target_size = (500, 384)

## Setting some hyperparameters

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'
## One can change this configuration and try out new combination
config = {
  "coordinate_size": 96,              ## (768/8), 8 for each of the 8 coordinates of x, y
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "image_feature_pool_shape": [7, 7, 256],
  "intermediate_ff_size_factor": 4,
  "max_2d_position_embeddings": 2048,
  "max_position_embeddings": 2048,
  "max_relative_positions": 8,
  "num_attention_heads": 12,
  "num_hidden_layers": 2,
  "pad_token_id": 0,
  "shape_size": 96,
  "vocab_size": 30522,
  "layer_norm_eps": 1e-12,
}



from tqdm.auto import tqdm

## For the purpose of prediction
id2label = []
label2id = {}

df = pd.read_csv("/home/ec2-user/collect_iq_dataset_500.csv")
all_label = ['driver_license_front', 'atpi','utility_bill','phone_bill', 'bank_statement', 'paystub', 'other_id','voided_check', 'ach_withdrawal_acknowledgment_form']

#df = df[df['type'].isin(['retail_installment_sales_contract', 'paystub', 'bank_statement', 'credit_approval', 'buyers_order', 'credit_application', 'vehicle_service_contract', 'buy_program', 'contacts', 'cover_sheet', 'odometer_disclosure_statement_retail', 'gap_binder', 'electronic_consent', 'driver_license_front', 'atpi', 'title_application', 'phone_bill', 'insurance_id_card', 'nada_bookout_sheet', 'utility_bill', 'risk_based_pricing_notice'])]

df = df[df['type'].isin(all_label)]
#all_label = ['driver_license_front', 'cover_sheet']

#df = df[df['type'].isin(all_label)]

print("Value counts", df['type'].value_counts())
print("SHAPE", df.shape)
df = df.groupby('type').apply(lambda x:x.sample(200))
#df = df.sample(50)

for i, label  in enumerate(all_label):
    label2id[label]=i
    id2label.append(label)

#print("LABEL2ID", label2id)
#print("IndexLabel", id2label)
curr_class = 0
## Preparing the Dataset
base_directory = '/home/ec2-user/original_images'
dict_of_img_labels = {'img':[], 'label':[]}

images = list(df['file_names'])
df['type_2'] = df['type'].apply(lambda x:label2id[x])

dict_of_img_labels['img'] = images
dict_of_img_labels['label'] = list(df['type_2'])


import pandas as pd
df = pd.DataFrame(dict_of_img_labels)
print(df.head())

from sklearn.model_selection import train_test_split as tts
train_df, valid_df = tts(df, random_state = seed, stratify = df['label'], shuffle = True)

train_df = train_df.reset_index().drop(columns = ['index'], axis = 1)
valid_df = valid_df.reset_index().drop(columns = ['index'], axis = 1)


## Creating the dataset

class RVLCDIPData(Dataset):
    
    def __init__(self, image_list, label_list, target_size, tokenizer, max_len = 256, transform = None):
        
        self.image_list = image_list
        self.label_list = label_list
        self.target_size = target_size
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.transform = transform
        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, idx):
        img_path = self.image_list[idx]
        label = self.label_list[idx]
        
        ## More on this, in the repo mentioned previously
        final_encoding = create_features(
            img_path,
            self.tokenizer,
            add_batch_dim=False,
            target_size=self.target_size,
            max_seq_length=self.max_len,
            path_to_save=None,
            save_to_disk=False,
            apply_mask_for_mlm=False,
            extras_for_debugging=False,
            use_ocr = True
    )
        if self.transform is not None:
            ## Note that, ToTensor is already applied on the image
            final_encoding['resized_scaled_img'] = self.transform(final_encoding['resized_scaled_img'])
        
        
        keys_to_reshape = ['x_features', 'y_features']
        for key in keys_to_reshape:
            final_encoding[key] = final_encoding[key][:self.max_len]
            
        final_encoding['label'] = torch.as_tensor(label).long()
        return final_encoding
    
## Defining the tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

from torchvision import transforms

## Normalization to these mean and std (I have seen some tutorials used this, and also in image reconstruction, so used it)
transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                              
train_ds = RVLCDIPData(train_df['img'].tolist(), train_df['label'].tolist(),
                      target_size, tokenizer, config['max_position_embeddings'], transform)
val_ds = RVLCDIPData(valid_df['img'].tolist(), valid_df['label'].tolist(),
                      target_size, tokenizer,config['max_position_embeddings'],  transform)

#print("TRAIN", next(iter(train_ds)), len(tokenizer))

#for i, trains  in enumerate(train_ds):
            #print("train max input:", torch.max(trains['input_ids']))
            #print("train min input:", torch.min(trains['input_ids']))
            #print("train max label:", torch.max(labels))
            #print("train min label:", torch.min(labels))
            #break

def collate_fn(data_bunch):

    '''
    A function for the dataloader to return a batch dict of given keys

    data_bunch: List of dictionary
    '''

    dict_data_bunch = {}

    for i in data_bunch:
        for (key, value) in i.items():
            if key not in dict_data_bunch:
                dict_data_bunch[key] = []
            #pad_value = config['max_position_embeddings'] - value
            #print("VALUE", value.shape, "PAD VALUE", pad_value.shape)
            #value = value + [0]*pad_value
            #print(value, type(value), value.shape)
            #print("LEN value", len(value), pad_value)
            dict_data_bunch[key].append(value)

    for key in list(dict_data_bunch.keys()):
        dict_data_bunch[key] = torch.stack(dict_data_bunch[key], axis = 0)

    return dict_data_bunch

def collate_fn(data_bunch):

  '''
  A function for the dataloader to return a batch dict of given keys

  data_bunch: List of dictionary
  '''

  dict_data_bunch = {}

  for i in data_bunch:
    for (key, value) in i.items():
      if key not in dict_data_bunch:
        dict_data_bunch[key] = []
      dict_data_bunch[key].append(value)

  for key in list(dict_data_bunch.keys()):
      dict_data_bunch[key] = torch.stack(dict_data_bunch[key], axis = 0)

  return dict_data_bunch


import pytorch_lightning as pl

class DataModule(pl.LightningDataModule):

    def __init__(self, train_dataset, val_dataset,  batch_size = 1):

        super(DataModule, self).__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.batch_size, 
                          collate_fn = collate_fn, shuffle = True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size = self.batch_size,
                                      collate_fn = collate_fn, shuffle = False)


#datamodule = DataModule(train_ds, val_ds)


class DocFormerForClassification(nn.Module):
  
    def __init__(self, config):
      super(DocFormerForClassification, self).__init__()

      self.resnet = ResNetFeatureExtractor(hidden_dim = config['max_position_embeddings'])
      #self.resnet = self.resnet.resize_token_embeddings(len(tokenizer))
      self.embeddings = DocFormerEmbeddings(config)
      self.lang_emb = LanguageFeatureExtractor()
      self.config = config
      self.dropout = nn.Dropout(config['hidden_dropout_prob'])
      self.linear_layer = nn.Linear(in_features = config['hidden_size'], out_features = len(id2label))  ## Number of Classes
      self.encoder = DocFormerEncoder(config)

    def forward(self, batch_dict):

      #print(batch_dict['x_features'], "XFEATRURES",batch_dict['x_features'].shape)
      x_feat = batch_dict['x_features']
      y_feat = batch_dict['y_features']

      token = batch_dict['input_ids']
      img = batch_dict['resized_scaled_img']

      v_bar_s, t_bar_s = self.embeddings(x_feat,y_feat)
      v_bar = self.resnet(img)
      t_bar = self.lang_emb(token)
      out = self.encoder(t_bar,v_bar,t_bar_s,v_bar_s)
      out = self.linear_layer(out)
      out = out[:, 0, :]
      return out

#doc_fomer = DocFormerForClassification()
#print("MODEL", doc_former)

## Defining pytorch lightning model
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torchmetrics



class DocFormer(pl.LightningModule):

  def __init__(self, config , lr = 5e-5):
    super(DocFormer, self).__init__()

    self.save_hyperparameters()
    self.config = config
    self.docformer = DocFormerForClassification(config)

    self.num_classes = len(id2label)
    self.train_accuracy_metric = torchmetrics.Accuracy()
    self.val_accuracy_metric = torchmetrics.Accuracy()
    self.f1_metric = torchmetrics.F1Score(num_classes=self.num_classes)
    self.precision_macro_metric = torchmetrics.Precision(
            average="macro", num_classes=self.num_classes
        )
    self.recall_macro_metric = torchmetrics.Recall(
            average="macro", num_classes=self.num_classes
        )
    self.precision_micro_metric = torchmetrics.Precision(average="micro",  num_classes=self.num_classes)
    self.recall_micro_metric = torchmetrics.Recall(average="micro",  num_classes=self.num_classes)

  def forward(self, batch_dict):
    logits = self.docformer(batch_dict)
    return logits

  def training_step(self, batch, batch_idx):
    logits = self.forward(batch)

    loss = nn.CrossEntropyLoss()(logits, batch['label'])
    preds = torch.argmax(logits, 1)

    ## Calculating the accuracy score
    train_acc = self.train_accuracy_metric(preds, batch["label"])

    ## Logging
    self.log('train/loss', loss,prog_bar = True, on_epoch=True, logger=True, on_step=True)
    self.log('train/acc', train_acc, prog_bar = True, on_epoch=True, logger=True, on_step=True)

    return loss
  
  def validation_step(self, batch, batch_idx):
    logits = self.forward(batch)
    loss = nn.CrossEntropyLoss()(logits, batch['label'])
    preds = torch.argmax(logits, 1)
    
    labels = batch['label']
    # Metrics
    valid_acc = self.val_accuracy_metric(preds, labels)
    precision_macro = self.precision_macro_metric(preds, labels)
    recall_macro = self.recall_macro_metric(preds, labels)
    precision_micro = self.precision_micro_metric(preds, labels)
    recall_micro = self.recall_micro_metric(preds, labels)
    f1 = self.f1_metric(preds, labels)

    # Logging metrics
    self.log("valid/loss", loss, prog_bar=True, on_step=True, logger=True)
    self.log("valid/acc", valid_acc, prog_bar=True, on_epoch=True, logger=True, on_step=True)
    self.log("valid/precision_macro", precision_macro, prog_bar=True, on_epoch=True, logger=True, on_step=True)
    self.log("valid/recall_macro", recall_macro, prog_bar=True, on_epoch=True, logger=True, on_step=True)
    self.log("valid/precision_micro", precision_micro, prog_bar=True, on_epoch=True, logger=True, on_step=True)
    self.log("valid/recall_micro", recall_micro, prog_bar=True, on_epoch=True, logger=True, on_step=True)
    self.log("valid/f1", f1, prog_bar=True, on_epoch=True)
    
    return {"label": batch['label'], "logits": logits}

  def validation_epoch_end(self, outputs):
        labels = torch.cat([x["label"] for x in outputs])
        logits = torch.cat([x["logits"] for x in outputs])
        preds = torch.argmax(logits, 1)

        ##wandb.log({"cm": #wandb.sklearn.plot_confusion_matrix(labels.cpu().numpy(), preds.cpu().numpy())})
        #self.logger.experiment.log(
        #    {"roc": #wandb.plot.roc_curve(labels.cpu().numpy(), logits.cpu().numpy())}
        #)
        
  def configure_optimizers(self):
    return torch.optim.AdamW(self.parameters(), lr = self.hparams['lr'])


from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
#from pytorch_lightning.loggers import #wandbLogger

def main():

    import time
    datamodule = DataModule(train_ds, val_ds)
    print("PASSD DATAMODEULE")
    docformer = DocFormer(config)

    checkpoint_callback = ModelCheckpoint(
        dirpath="./models", monitor="valid/loss", mode="min"
    )
    early_stopping_callback = EarlyStopping(
        monitor="valid/loss", patience=3, verbose=True, mode="min"
    )
    
    ##wandb.init(config=config, project="RVL CDIP with DocFormer New Version")
    ##wandb_logger = #wandbLogger(project="RVL CDIP with DocFormer New Version", entity="iakarshu")
    ## https://www.tutorialexample.com/implement-reproducibility-in-pytorch-lightning-pytorch-lightning-tutorial/
    print("Training STARTED---------------------------------------------------------------------------------------")
    pl.seed_everything(seed, workers=True)
    trainer = pl.Trainer(
        default_root_dir="logs",
        #gpus=(1 if torch.cuda.is_available() else 0),
        #gpus = 0,
        devices='auto',
        accelerator='auto',
        max_epochs=10,
        fast_dev_run=False,
        #logger=#wandb_logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        deterministic=True
    )

    start = time.time()
    trainer.fit(docformer, datamodule)
    end = time.time()

    print("Total training time: ", (end-start))

if __name__ == "__main__":
    main()





