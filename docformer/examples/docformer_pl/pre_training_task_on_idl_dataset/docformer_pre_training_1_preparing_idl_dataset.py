# -*- coding: utf-8 -*-
"""DocFormer Pre-training : 1. Preparing IDL Dataset

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/17G4Io-2BOLx5YwKymgIwO_g14HovwbBb
"""

## Refer here for the dataset: https://github.com/furkanbiten/idl_data 
# (IDL dataset was also used in the pre-training of LaTr), might take time to download the dataset


## The below lines of code download the sample dataset 
# !wget http://datasets.cvc.uab.es/UCSF_IDL/Samples/ocr_imgs_sample.zip
# !unzip /content/ocr_imgs_sample.zip
# !rm /content/ocr_imgs_sample.zip

# Commented out IPython magic to ensure Python compatibility.
# ## Installing the dependencies (might take some time)
# 
# %%capture
# !pip install pytesseract
# !sudo apt install tesseract-ocr
# !pip install transformers
# !pip install pytorch-lightning
# !pip install einops
# !pip install tqdm
# !pip install 'Pillow==7.1.2'
# !pip install PyPDF2

## Getting the JSON File

import json

## For reading the PDFs
from PyPDF2 import PdfReader
import io
from PIL import Image, ImageDraw

## Standard library
import os

pdf_path = "./sample/pdfs"
ocr_path = "./sample/OCR"

## Image property

resize_scale = (500, 500)

from typing import List

def normalize_box(box : List[int], width : int, height : int, size : tuple = resize_scale):
    """
    Takes a bounding box and normalizes it to a thousand pixels. If you notice it is
    just like calculating percentage except takes 1000 instead of 100.
    """
    return [
        int(size[0] * (box[0] / width)),
        int(size[1] * (box[1] / height)),
        int(size[0] * (box[2] / width)),
        int(size[1] * (box[3] / height)),
    ]

## Function to get the images from the PDFs as well as the OCRs for the corresponding images

def get_image_ocrs_from_path(pdf_file_path : str, ocr_file_path : str, resize_scale = resize_scale):

  ## Getting the image list, since the pdfs can contain many image
  reader = PdfReader(pdf_file_path)
  img_list = []
  for i in range(len(reader.pages)):
    page = reader.pages[i]
    for image_file_object in page.images:

      stream = io.BytesIO(image_file_object.data)
      img = Image.open(stream).convert("RGB")
      img_list.append(img)

  json_entry = json.load(open(ocr_file_path))[1]
  json_entry =[x for x in json_entry["Blocks"] if "Text" in x]

  pages = [x["Page"] for x in json_entry]
  ocrs = {pg : [] for pg in set(pages)}

  for entry in json_entry:
    bbox = entry["Geometry"]["BoundingBox"]
    x, y, w, h = bbox['Left'], bbox['Top'], bbox["Width"], bbox["Height"]
    bbox = [x, y, x + w, y + h]
    bbox = normalize_box(bbox, width = 1, height = 1, size = resize_scale)
    ocrs[entry["Page"]].append({"word" : entry["Text"], "bbox" : bbox})

  return img_list, ocrs

# sample_pdf_folder = os.path.join(pdf_path, sorted(os.listdir(pdf_path))[0])
# sample_ocr_folder = os.path.join(ocr_path, sorted(os.listdir(ocr_path))[0])

# sample_pdf = os.path.join(sample_pdf_folder, sample_pdf_folder.split("/")[-1] + ".pdf")
# sample_ocr = os.path.join(sample_ocr_folder, os.listdir(sample_ocr_folder)[0])

# img_list, ocrs = get_image_ocrs_from_path(sample_pdf, sample_ocr)

"""## Preparing the Pytorch Dataset"""

from tqdm.auto import tqdm

img_list = []
ocr_list = []

pdf_files = sorted(os.listdir(pdf_path))[:30] ## Using only 30 since, google session gets crashed
ocr_files = sorted(os.listdir(ocr_path))[:30] 

for pdf, ocr in tqdm(zip(pdf_files, ocr_files), total = len(pdf_files)):
  pdf = os.path.join(pdf_path, pdf, pdf + '.pdf')
  ocr = os.path.join(ocr_path, ocr)
  ocr = os.path.join(ocr, os.listdir(ocr)[0])
  img, ocrs = get_image_ocrs_from_path(pdf, ocr)

  for i in range(len(img)):
    img_list.append(img[i])
    ocr_list.append(ocrs[i+1])  ## Pages are 1, 2, 3 hence 0 + 1, 1 + 1, 2 + 1

"""## Visualizing the OCRs"""

index = 43
curr_img = img_list[index].resize(resize_scale)
curr_ocr = ocr_list[index]

# create rectangle image
draw_on_img = ImageDraw.Draw(curr_img)  

for it in curr_ocr:
  box = it["bbox"]
  draw_on_img.rectangle(box, outline ="violet")

curr_img

