
import pymira
import torch
from pymira.nets.itn import ITN2D, ITN3D
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

def batchFromPaths(src_path, trg_path, src_seg_path, trg_seg_path, resampler_img, resampler_seg):

  source = sitk.ReadImage(src_path, sitk.sitkFloat32)
  target = sitk.ReadImage(trg_path, sitk.sitkFloat32)
  source_seg = sitk.ReadImage(src_seg_path, sitk.sitkFloat32)
  target_seg = sitk.ReadImage(trg_seg_path, sitk.sitkFloat32)

  if resampler_img:
      source = resampler_img(source)
      target = resampler_img(target)

  if resampler_seg:
      source_seg = resampler_seg(source_seg)
      target_seg = resampler_seg(target_seg)

  if len(source.GetSize()) == 3:
      source.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
      target.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
  else:
      source.SetDirection((1, 0, 0, 1))
      target.SetDirection((1, 0, 0, 1))

  source.SetOrigin(np.zeros(len(source.GetOrigin())))
  target.SetOrigin(np.zeros(len(target.GetOrigin())))
  source_seg.CopyInformation(source)
  target_seg.CopyInformation(target)



  sample = {'source': torch.from_numpy(sitk.GetArrayFromImage(source)).unsqueeze(0), 
    'target': torch.from_numpy(sitk.GetArrayFromImage(target)).unsqueeze(0), 
    'source_seg': torch.from_numpy(sitk.GetArrayFromImage(source_seg)).unsqueeze(0), 
    'target_seg': torch.from_numpy(sitk.GetArrayFromImage(target_seg)).unsqueeze(0)}
  return sample

def batchFromArrays(src, trg, src_seg, trg_seg, resampler_img, resampler_seg):
  source = sitk.GetImageFromArray(src)
  target = sitk.GetImageFromArray(trg)
  source_seg = sitk.GetImageFromArray(src_seg)
  target_seg = sitk.GetImageFromArray(trg_seg)

  if resampler_img:
      source = resampler_img(source)
      target = resampler_img(target)

  if resampler_seg:
      source_seg = resampler_seg(source_seg)
      target_seg = resampler_seg(target_seg)

  if len(source.GetSize()) == 3:
      source.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
      target.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
  else:
      source.SetDirection((1, 0, 0, 1))
      target.SetDirection((1, 0, 0, 1))

  source.SetOrigin(np.zeros(len(source.GetOrigin())))
  target.SetOrigin(np.zeros(len(target.GetOrigin())))
  source_seg.CopyInformation(source)
  target_seg.CopyInformation(target)

def batchCustom(src_path, trg, src_seg_path, trg_seg, resampler_img, resampler_seg):
  source = sitk.ReadImage(src_path, sitk.sitkFloat32)
  target = trg
  source_seg = sitk.ReadImage(src_seg_path, sitk.sitkFloat32)
  target_seg = trg_seg

  if resampler_img:
      source = resampler_img(source)
      target = resampler_img(target)

  if resampler_seg:
      source_seg = resampler_seg(source_seg)
      target_seg = resampler_seg(target_seg)

  if len(source.GetSize()) == 3:
      source.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
      target.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
  else: 
      source.SetDirection((1, 0, 0, 1))
      target.SetDirection((1, 0, 0, 1))

  source.SetOrigin(np.zeros(len(source.GetOrigin())))
  target.SetOrigin(np.zeros(len(target.GetOrigin())))
  source_seg.CopyInformation(source)
  target_seg.CopyInformation(target)


  sample = {'source': torch.from_numpy(sitk.GetArrayFromImage(source)).unsqueeze(0), 
    'target': torch.from_numpy(sitk.GetArrayFromImage(target)).unsqueeze(0), 
    'source_seg': torch.from_numpy(sitk.GetArrayFromImage(source_seg)).unsqueeze(0), 
    'target_seg': torch.from_numpy(sitk.GetArrayFromImage(target_seg)).unsqueeze(0)}
  return sample

def warpImg(sample, stn, itn):
  with torch.no_grad():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:" + args.dev if use_cuda else "cpu")

    dataloader_train = torch.utils.data.DataLoader([sample], batch_size=1)

    for batch in dataloader_train:
      source = batch['source'].to(device)
      target = batch['target'].to(device)
      source_seg = batch['source_seg'].to(device)
      target_seg = batch['target_seg'].to(device)

      source_prime = itn(source)
      target_prime = itn(target)

      stn(torch.cat((source_prime, target_prime), dim=1))

      warped_source = stn.warp_image(source)
      warped_source = sitk.GetImageFromArray(warped_source.cpu().squeeze().numpy())
      warped_source.CopyInformation(sitk.GetImageFromArray(target.cpu().squeeze().numpy()))

      warped_source_seg = stn.warp_image(source_seg)
      warped_source_seg = sitk.GetImageFromArray(warped_source_seg.cpu().squeeze().numpy())
      warped_source.CopyInformation(sitk.GetImageFromArray(target.cpu().squeeze().numpy()))
      
      return warped_source, warped_source_seg

def align(args):

  scansCSVPath = args.scansCSVPath
  itnPath = args.itnPath
  stnPath = args.stnPath

  # set up the ISTN framework skeleton
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda:" + args.dev if use_cuda else "cpu")
  itn = ITN2D(input_channels=1).to(device)
  stn = STN2D(input_size=[256, 256], input_channels=2, device=device).to(device)

  # load ISTN weights to make working model
  itn.load_state_dict(torch.load(itnPath))
  stn.load_state_dict(torch.load(stnPath))

  # make ISTN in evaluation mode
  itn.eval()
  stn.eval()

  # initialize a resampler (whatever that is???)
  resampler_img = Resampler( [1,1], [256, 256])
  resampler_seg = Resampler( [1,1], [256, 256])

  # this will store the images in alignment
  alignedImages = []

  # open a csv of scans, each line having a source and target file path separated by a comma
  with open(scansCSVPath) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')

    src_path = None
    src_seg_path = None

    trg_img = None
    trg_seg_img = None

    for row in csv_reader:
      # the first image (and segs) is assumed correct by default so make it the first target to align to
      if (trg_img is None):
        trg_img = sitk.ReadImage(row[0], sitk.sitkFloat32) # first image and segs are first target to align to
        trg_seg_img = sitk.ReadImage(row[1], sitk.sitkFloat32) #read first image and segs directly, no warping
        # alignedImages.append(sitk.GetArrayFromImage(trg_img))
        alignedImages.append(sitk.GetArrayFromImage(trg_img))
        continue
      else:
        # any subsequent image is a source to be aligned to the current target image
        src_path = row[0]
        src_seg_path = row[1]

      # combine images in a way that can be inputted into ISTN
      b = batchCustom(src_path, trg_img, src_seg_path, trg_seg_img, resampler_img, resampler_seg)
      
      # input images into ISTN and get a warped image pair aligned to target
      warpedImg, warpedSeg = warpImg(b, stn, itn)
      
      print(stn.getTheta()) #this will give me the important part of the affine transformation matrix
      
      # alignedImages.append(warpedImg.cpu().squeeze().numpy())
      # alignedImages.append(sitk.GetArrayFromImage(warpedImg))  # it must be numpy array to be used later
      alignedImages.append(sitk.GetArrayFromImage(warpedImg))

      #set the next target to be the warped image and its landmark segmentation
      trg_img = warpedImg
      trg_seg_img = warpedSeg

  for indx, image in enumerate(alignedImages):
    cv2.imwrite("image" + str(indx) + ".png", (image))
    # sitk.WriteImage(image, "image" + str(indx) + ".png")

  # alignedImages = map(lambda x: sitk.GetArrayFromImage(x), alignedImages)
  pickle.dump(alignedImages, open("alignedImages.dat", "wb"))
    

        

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='ISTN Aligner')
  parser.add_argument('--scansCSVPath', default="", help='path to csv of scans')
  parser.add_argument('--itnPath', default="", help='path to itn')
  parser.add_argument('--stnPath', default="", help='path to stn')
  
  args = parser.parse_args()
  align(args)

  







