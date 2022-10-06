# -*- coding: utf-8 -*-
import yaml
import csv
from pathlib import Path
from collections import OrderedDict
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pytorch_lightning as pl

from pycocotools import mask
from pycocotools.coco import COCO

RANDOM_SEED = 30

class Reproj_Dataset(Dataset):

    def __init__(self,
                dataset_file,
                subset,
                resize_prop,
                sequencing,
                class_labels=[],
                samplesNum=[None, None],
                depth_rel_path='',
                ):

      self.dataset_name = Path(dataset_file).stem
      self._root_dir = Path(dataset_file).parent.parent / self.dataset_name


      self.coco_mask = mask
      self.class_labels = class_labels
      self.resize_prop = resize_prop

      self.sequencing = sequencing
      self.subset = subset

      assert self.subset == 'train' or self.subset == 'valid' or self.subset == 'eval' or self.subset == 'infer'

      if self.subset =='train':
          self.samplesNum = samplesNum[0]
      else:
          self.samplesNum = samplesNum[1]

      
      # Directories and root addresses
      self.annotation_files_dir = self._root_dir / (self.dataset_name + '.json')
      #  Image Sets lists (train, valid, eval)
      self.dataset_config_list_dir = self._root_dir / (self.dataset_name + '.yaml')
      with open(self.dataset_config_list_dir) as fp:
          self.dataset_config = yaml.load(fp, Loader=yaml.FullLoader)

      self.image_sets = self.dataset_config["image_sets"]
      self.stds = self.dataset_config["img_mu"]
      self.means = self.dataset_config["img_std"]

      # initialize COCO api for instance annotations
      self.coco = COCO(self.annotation_files_dir)

      # getting the IDs of images inside the directory
      self.ids = self.coco.getImgIds()
      # Get Categories of loaded Dataset
      self.cat_ids = self.coco.getCatIds()

      # display COCO categories and supercategories
      self.classes = self.coco.loadCats(self.cat_ids)
      self.coco_labels = [cat['name'] for cat in self.classes]

      if self.subset == 'infer':
        self.img_set_ids = self.image_sets['eval']
      else:
        self.img_set_ids = self.image_sets[self.subset]

      self.img_path_to_ids = {}
      for md in self.coco.loadImgs(self.img_set_ids):
        im_path = self._root_dir / self.dataset_rel_path(md['path'])
        self.img_path_to_ids[im_path] = md['id']

      if self.subset == 'infer':
        self.img_set_ids = [str(p) for p in sorted(Path(self.inference_path).iterdir()) if p.suffix in ['.tiff','.png']]

      self.depth_rel_path = depth_rel_path

      # image sequencing options
      if self.sequencing is not None:
        if "random_frame_skips" in self.sequencing and self.sequencing['random_frame_skips']:
          if subset=='eval' or subset=='infer':
            self.used_frames_seqs = self.genRandomFrameSkips(num_sequences=len(self.img_set_ids), seed=RANDOM_SEED)
          else:
            self.used_frames_seqs = self.genRandomFrameSkips(num_sequences=len(self.img_set_ids))
        else: 
          self.used_frames_seqs = [self.sequencing["used_frames"]]

        if 'odometry_rel_file_path' in sequencing.keys():
          self.odom_file_path = sequencing['odometry_rel_file_path']
        else:
          self.odom_file_path = 'odometry.csv'

        if 'robot_mask' in sequencing.keys() and sequencing['robot_mask']['enable']:
            if'robot_mask_path' in sequencing['robot_mask'].keys():
              self.robot_mask_path = sequencing['robot_mask_path']
            else:
              self.robot_mask_path = 'robot_mask.png'

      if resize_prop is not None:
        self.resize_prop = resize_prop

        self.tf_resize_img = transforms.Resize((self.resize_prop["height"], self.resize_prop["width"]), transforms.InterpolationMode('bicubic'))
        self.tf_resize_depth = transforms.Resize((self.resize_prop["height"], self.resize_prop["width"]), transforms.InterpolationMode('nearest'))
        self.tf_resize_lbl = transforms.Resize((self.resize_prop["height"], self.resize_prop["width"]), transforms.InterpolationMode('nearest'))

      # transformations for tensors
      self.tensorize_img = self.tensorize_depth = transforms.ToTensor()

      print(f"*** {subset} with {self.__len__()} samples ***")

    ##
    ## get item
    ##
    ######################################################################################
    def __getitem__(self, index):

      if self.sequencing is not None:
        direction = self.sequencing["direction_of_travel"]
        
        if len(self.used_frames_seqs) == 1:
          self.used_frames = self.used_frames_seqs[0]
        else:
          self.used_frames = self.used_frames_seqs[index]

        data = [{} for _ in range(len(self.used_frames))]

        imgList, labelList, depthList, odomList, camParams, fileNameList = self.getReprojSequence(index, direction)

        if self.resize_prop is not None:
          camParams = self.matchCamParamsScale(camParams, imgList)
        
        for i in range(len(data)): data[i]['intrinsics'] = camParams['intrinsics']
        for i in range(len(data)): data[i]['extrinsics'] = camParams['extrinsics']

        # pass odometry to the next frame
        for i, o in enumerate(odomList): data[i]['odom'] = o

        if hasattr(self, 'robot_mask_path'):
          robot_mask = self.getRobotMask(index)
          robotMaskList = [robot_mask for _ in range(len(imgList))]
          robot_mask_tensors = self.prepareTensors(robotMaskList, 'mask')
          for i, mask in enumerate(robot_mask_tensors): data[i]["robot_mask"] = mask

      else:
        data = [{}]

        img, mask, fileName = self.getImgLabelPairFromIdx(index)
        imgList = [img]
        labelList = [mask]
        fileNameList = [fileName]
        depthList = [self.getDepthFromIdx(index)]


      # Tensorize image/imgae-list and its mask/mask-list
      imgs_tensors = self.prepareTensors(imgList, 'rgb')
      for i, im in enumerate(imgs_tensors): data[i]["rgb"] = im

      depth_tensors = self.prepareTensors(depthList, 'depth')
      for i, d in enumerate(depth_tensors): data[i]["depth"] = d 

      if len(labelList) == 0:
        print('no mask')
        import sys; exit(1);
      # prepare labels and flags for RNN loss computation
      mask_tensors = self.prepareTensors(labelList, 'label')
      for i, m in enumerate(mask_tensors): 
        data[i]["labels"] = m

      if sum(['labels' in d.keys() for d in data]) == 0:
        print(['labels' in d.keys() for d in data])
        img_metadata = self.coco.loadImgs(self.getTragetID(index))[0]
        print(f'image path: {img_metadata["path"]}')
        print('Too many labels!, exiting...')
        import sys; exit(1);

      for i, file_name in enumerate(fileNameList): data[i]["file_names"] = file_name

      return data


    ##
    ## Dataloader utils
    ##
    ######################################################################################
    def genRandomFrameSkips(self,num_sequences=1,seed=None):
      if seed is not None:
        np.random.default_rng(seed)
      # Get sequence frame skips parameters from config dict
      min_skips = max(1, self.sequencing['min_skip'] if 'min_skip' in self.sequencing else 1)
      max_skips = max(1, self.sequencing['max_skip'] if 'max_skip' in self.sequencing else 1)
      num_frames = max(1, self.sequencing['num_frames'] if 'num_frames' in self.sequencing else 1)
      # Generate requested frame sequences
      return [[0, *np.cumsum(np.random.randint(min_skips, max_skips+1, num_frames-1))] for _ in range(num_sequences)]
      
    def matchCamParamsScale(self, camParams, origImage):
      if type(origImage) == list:
        origImage = origImage[0]
      assert self.resize_prop is not None
      h_scale = self.resize_prop["height"] / origImage.size[1]
      w_scale = self.resize_prop["width"] / origImage.size[0]
      # scale x parameters row
      camParams['intrinsics'][0] *= w_scale
      # scale y parameters row
      camParams['intrinsics'][1] *= h_scale
      return camParams

    def prepareTensors(self, imgList, imgType):
      result = []
      img = None
      reveseType = False
      if not type(imgList) == list:
        imgList = [imgList]
        reveseType = True

      for i in imgList:
        # tensorize image i
        if i is None:
          img = torch.Tensor([])
        elif imgType == 'rgb':

          # Resize the img and it label
          if self.resize_prop is not None:
            img = self.tf_resize_img(i)
          else:
            img = i.copy()

          img = self.tensorize_img(img.copy()).float()

        elif imgType == 'label':

          # Resize the img and it label
          if self.resize_prop is not None:
            img = self.tf_resize_lbl(i)
          else:
            img = i.copy()

          img = torch.from_numpy(np.array(img)).long()

        elif imgType == 'depth' or imgType == 'mask':

          # Resize the img and it label
          if self.resize_prop is not None:
            img = self.tf_resize_depth(i)
          else:
            img = i.copy()

          img = self.tensorize_depth(img)

        # append image i to list of images !
        if not reveseType:
          result.append(img)
        else:
          result = img

      return result

    def getRobotMask(self, index):
      if not hasattr(self, 'robot_mask_path'):
        raise ValueError('Robot mask was not enabled in the dataloader config file. Add sequencing:robot_mask:enable:True to you .yaml config file')
      img_path = self.getImgPathFromIdx(index)
      return Image.open(img_path.parent.parent / self.robot_mask_path).convert('L')

    def getReprojSequence(self, index, direction):
      return self.getNRealImgLabelPairs(index, direction, return_reproj_data=True)

    def getImageMetadataFromIdx(self,index):
      return self.coco.loadImgs(self.getTragetID(index))[0]

    def getDepthFromIdx(self,index):
      md = self.getImageMetadataFromIdx(index)
      ds_img_path = self.dataset_rel_path(md['path'])
      img_path = self._root_dir / ds_img_path
      return Image.open(img_path.parent / self.depth_rel_path / img_path.name)

    def csv_odom_to_transforms(self, path):
      odom_tfs = {}
      with open(path, mode='r') as f:
        reader = csv.reader(f)
        # get header and change timestamp label name
        header = next(reader)
        header[0] = 'ts'
        # Convert string odometry to numpy transfor matrices
        for row in reader:
          odom = {l: row[i] for i, l in enumerate(header)}
          # Translarion and rotation quaternion as numpy arrays 
          trans = np.array([float(odom[l]) for l in ['tx', 'ty', 'tz']])
          rot = Rotation.from_quat([float(odom[l]) for l in ['qx', 'qy', 'qz', 'qw']]).as_matrix()
          # Build numpy transform matrix
          odom_tf = torch.eye(4)
          odom_tf[0:3, 3] = torch.from_numpy(trans)
          odom_tf[0:3, 0:3] = torch.from_numpy(rot)
          # Add transform to timestamp indexed dictionary
          odom_tfs[odom['ts']] = odom_tf
      
      return odom_tfs

    def getNRealImgLabelPairs(self, index, direction, return_reproj_data=False):
      img_id = self.getTragetID(index)
      # get sorted image paths of all images in the dataset sequence
      img_metadata = self.coco.loadImgs(img_id)[0]
      im_ds_path = self.dataset_rel_path(img_metadata['path'])
      img_path = str(self._root_dir / im_ds_path)

      return self.getNRealImgLabelPairsFromImgPath(direction, img_path, return_reproj_data) 

    def getNRealImgLabelPairsFromImgPath(self, direction, img_path, return_reproj_data=False):
      img_path = Path(img_path)
      img_parent_path = Path(img_path).parent
      img_seq_paths = [p for p in sorted(img_parent_path.iterdir()) if p.suffix == img_path.suffix]
      # index of the central image of the sequence to extract
      seq_idx = img_seq_paths.index(img_path)

      if return_reproj_data:
        # read odometry as a disctionary indexed by [us] timestamp as a string
        odom_from_ts = self.csv_odom_to_transforms(str(img_parent_path / self.odom_file_path))
        
      # Extract images for the sequence relative to the central one,
      # considering the direction of travel
      # The last frame is always idx=0 (aka. central frame)
      imgList = []
      labelList = []
      depthList = []
      odomList = []
      fileNameList = []

      frame_deltas = reversed(sorted(self.used_frames))
      
      img_idxs = [min(len(img_seq_paths)-1,max(0,int(seq_idx - d * direction))) for d in frame_deltas]
      img_paths = [img_seq_paths[idx] for idx in img_idxs]

      for i, path in enumerate(img_paths):
        img, mask, file_name= self.getImgLabelPairFromPath(path)
        imgList.append(img)
        labelList.append(mask)
        fileNameList.append(file_name)

        if return_reproj_data:
          #TODO: add a dept path parameter on the config file
          # get depth
          depthList.append(Image.open(path.parent / self.depth_rel_path / path.name))
          
          if i == len(img_paths)-1:
            # Identity odometry for the last frame in the sequence
            odomList.append(torch.eye(4))
          else:
            # get odometry according to timestamps (aka. filenames)
            t_src = str(path.name).split('.')[0]
            t_dst = str(img_paths[i+1].name).split('.')[0]
            dtf = torch.matmul(torch.inverse(odom_from_ts[t_src]), odom_from_ts[t_dst])
            odomList.append(dtf)
      
      if return_reproj_data:
        # load camera intrinsics and extrinsincs from yaml file
        with open(str(img_parent_path / 'params.yaml'), mode='r') as yml:
          camParams = yaml.load(yml, Loader=yaml.FullLoader)
          camParams = {k:torch.Tensor(v) for k,v in camParams.items()}
        return imgList, labelList, depthList, odomList, camParams, fileNameList

      return imgList, labelList, fileNameList

    def getTragetID(self, index):
      img_set_ids = self.img_set_ids[index]
      img_list_idx = next((index for (index, d) in enumerate(self.coco.dataset["images"]) if d["id"] == img_set_ids), None)
      image_id = self.coco.dataset["images"][img_list_idx]["id"]

      return image_id

    def getImgLabelPairFromPath(self, path):
      # Retrieve annotated image if in the dataset
      if path in self.img_path_to_ids.keys():
        return self.getImgLabelPairFromId(self.img_path_to_ids[path])
      # If image is not annotated, return it and an empty mask
      img = Image.open(path).convert('RGB')
      label = Image.fromarray(np.ones(img.size[::-1]) *  -1)
      return img, label, Path(path).name

    def getImgLabelPairFromIdx(self, index):
      img_id = self.getTragetID(index)
      return self.getImgLabelPairFromId(img_id)

    def getImgLabelPairFromId(self, img_id):
      # Get meta data of called image
      img_metadata = self.coco.loadImgs(img_id)[0]
      im_ds_path = self.dataset_rel_path(img_metadata['path'])
      #  Load BGR image and convert it to RGB
      img = Image.open(self._root_dir / im_ds_path).convert('RGB')
      #  Load Annotation if Image with ID
      cocoTarget = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
      # Creat the mask with loaded annotations with same size as RGB image
      label = Image.fromarray(self.generateMask(img_metadata))
      # Creat the mask with loaded annotations with same size as RGB image
      label = Image.fromarray(self.generateMask(img_metadata))
      return img, label, Path(im_ds_path).name

    def generateMask(self, img_metadata):
      LUT = OrderedDict()
      cat_ids = set()
      # Get cat ids from dataset with names or supercategories sepecified in te config file
      for id, c in self.coco.cats.items():
        if c['supercategory'] in self.class_labels:
          LUT[id] = self.class_labels.index(c['supercategory'])
          cat_ids.add(id)
        elif c['name'] in self.class_labels:
          LUT[id] = self.class_labels.index(c['name'])
          cat_ids.add(id)

      anns_ids = self.coco.getAnnIds(imgIds=img_metadata['id'], catIds=cat_ids, iscrowd=None)
      anns = self.coco.loadAnns(anns_ids)

      mask = np.zeros((img_metadata['height'], img_metadata['width'])).astype(np.uint8)
      for ann in anns:
        if not ann['segmentation']:
          continue
        ann_mask = self.coco.annToMask(ann)
        mask *= not(ann_mask).all()
        mask += ann_mask * LUT[ann["category_id"]]
        mask = np.clip(mask, 0, max(LUT.values()))
      return mask

    def dataset_rel_path(self, path=''):
      path_parts = Path(path).parts
      if len(path_parts) < 4:
        raise ValueError('Invalid dataset path, it only has 2 or less subpaths')
      return str(Path(*path_parts[3:]))

    def __len__(self):
      return len(self.img_set_ids)

##
## Data module
##
#############################################################################
class Parser(pl.LightningDataModule):
  def __init__(self, cfg):
    super(Parser, self).__init__()
    self.cfg = cfg
  
    dataset = self.cfg['dataset']
    self.dataset_file = dataset["yaml_path"]
    self.class_labels = dataset["class_labels"]
    self.depth_rel_path = dataset["depth_rel_path"] if 'depth_rel_path' in dataset else ''

    dataloader = self.cfg['dataloader']
    self.workers = dataloader["workers_num"]
    self.batch_size = dataloader["batch_size"]
    self.resize_prop = dataloader["resize_prop"] if 'resize_prop' in dataloader else None
    self.sequencing = dataloader["sequencing"] if 'sequencing' in dataloader else None

  def train_dataloader(self):
    # Data loading code
    trainSet = Reproj_Dataset(dataset_file=self.dataset_file,
                          subset='train',
                          resize_prop=self.resize_prop,
                          class_labels=self.class_labels,
                          sequencing = self.sequencing,
                          depth_rel_path=self.depth_rel_path,
                          )

    trainloader = torch.utils.data.DataLoader(trainSet,
                                batch_size=self.batch_size,
                                shuffle=True,
                                num_workers=self.workers,
                                pin_memory=True,
                                drop_last=True)
    return trainloader

  def val_dataloader(self):
    validSet = Reproj_Dataset(dataset_file=self.dataset_file,
                          subset='valid',
                          resize_prop=self.resize_prop,
                          class_labels=self.class_labels,
                          sequencing = self.sequencing,
                          depth_rel_path=self.depth_rel_path,
                          )

    validloader = torch.utils.data.DataLoader(validSet,
                                batch_size=self.batch_size,
                                shuffle=False,
                                num_workers=self.workers,
                                pin_memory=True,
                                drop_last=False)
    return validloader
     
  def test_dataloader(self):
    evalSet = Reproj_Dataset(dataset_file=self.dataset_file,
                          subset='eval',
                          resize_prop=self.resize_prop,
                          class_labels=self.class_labels,
                          sequencing=self.sequencing,
                          depth_rel_path=self.depth_rel_path,
                          )

    evalloader = torch.utils.data.DataLoader(evalSet,
                                batch_size=self.batch_size,
                                shuffle=False,
                                num_workers=self.workers,
                                pin_memory=True,
                                drop_last=False)

    return evalloader