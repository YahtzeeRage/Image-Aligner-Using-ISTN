
import pymira
import torch
from pymira.nets.itn import ITN2D
from pymira.nets.stn import STN2D
from pymira.img.transforms import Resampler
import SimpleITK as sitk
import cv2
import csv
from pymira.img.datasets import ImageSegRegDataset
from pymira.img.transforms import Normalizer
import numpy as np

import pickle
import cv2
import argparse
#initialize istn

def batchCustom(source, target, resampler_img):

  # stuff copied from istn-reg.py code
  if resampler_img:
    source = resampler_img(source)
    target = resampler_img(target)
  source.SetDirection((1, 0, 0, 1))
  target.SetDirection((1, 0, 0, 1))
  source.SetOrigin(np.zeros(len(source.GetOrigin())))
  target.SetOrigin(np.zeros(len(target.GetOrigin())))

  # make images torch friendly
  sample = {'source': torch.from_numpy(sitk.GetArrayFromImage(source)).unsqueeze(0), 
    'target': torch.from_numpy(sitk.GetArrayFromImage(target)).unsqueeze(0)}
  return sample

def warpImg(sample, stn, itn):
  with torch.no_grad():
    # gpu setup
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:" + args.dev if use_cuda else "cpu")

    # put data in torch friendly form (only one image set)
    dataloader_train = torch.utils.data.DataLoader([sample], batch_size=1)

    for batch in dataloader_train:
      # push data to gpu 
      source = batch['source'].to(device)
      target = batch['target'].to(device)

      # get representation for source and target
      source_prime = itn(source)
      target_prime = itn(target)

      # set up STN with ITN representations
      stn(torch.cat((source_prime, target_prime), dim=1))

      # warp image and turn it into a simple itk image
      warped_source = stn.warp_image(source)
      warped_source = sitk.GetImageFromArray(warped_source.cpu().squeeze().numpy())
      warped_source.CopyInformation(sitk.GetImageFromArray(target.cpu().squeeze().numpy()))

      return warped_source

def align(args):

  scansCSVPath = args.scansCSVPath
  itnPath = args.itnPath
  stnPath = args.stnPath
  size = args.size

  # set up the ISTN framework skeleton
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda:" + args.dev if use_cuda else "cpu")
  itn = ITN2D(input_channels=1).to(device)
  stn = STN2D(input_size=size, input_channels=2, device=device).to(device)

  # load ISTN weights to make working model
  itn.load_state_dict(torch.load(itnPath))
  stn.load_state_dict(torch.load(stnPath))

  # put ISTN in evaluation mode
  itn.eval()
  stn.eval()

  # initialize a resampler (whatever that is???)
  resampler_img = Resampler( [1,1], size)

  # this will store the images in alignment
  alignedImages = []

  # open a csv of scans, each line having a source and target file path separated by a comma
  with open(scansCSVPath) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    trg_img = None

    for row in csv_reader:
      # the first image is assumed correct by default
      if (trg_img is None):
        # make it first target to align to
        trg_path = row[0]
        trg_img = sitk.ReadImage(trg_path, sitk.sitkFloat32)
        # we will store a running list of alignedf images
        alignedImages.append(sitk.GetArrayFromImage(trg_img))
        continue
      
      # combine source and target in a way that can be inputted into ISTN
      src_path = row[0]
      src_img = sitk.ReadImage(src_path, sitk.sitkFloat32)
      b = batchCustom(src_img, trg_img, resampler_img)
      
      # input images into ISTN and get a warped image aligned to target
      warpedSrc = warpImg(b, stn, itn)
      
      # I modified pymira
      # this will give me the important part of the affine transformation matrix
      # print(stn.getTheta()) 

      alignedImages.append(sitk.GetArrayFromImage(warpedSrc))

      #set the next target to be the warped image
      trg_img = warpedSrc

  # ways of storing the results
  for indx, image in enumerate(alignedImages):
    cv2.imwrite("image" + str(indx) + ".png", (image))

  pickle.dump(alignedImages, open("alignedImages.dat", "wb"))
    
if __name__ == '__main__':
  # commandline args
  parser = argparse.ArgumentParser(description='ISTN Aligner')
  parser.add_argument('--scansCSVPath', default="", help='path to csv of scans')
  parser.add_argument('--itnPath', default="", help='path to itn')
  parser.add_argument('--stnPath', default="", help='path to stn')
  parser.add_argument('--size', default="", help="size of image ")
  
  # align!
  args = parser.parse_args()
  align(args)