import warnings

import torch
import torch.nn.functional as F


from contextlib import contextmanager
@contextmanager
def set_default_tensor_type(tensor_type):
    if torch.tensor(0).is_cuda:
        old_tensor_type = torch.cuda.FloatTensor
    else:
        old_tensor_type = torch.FloatTensor

    torch.set_default_tensor_type(tensor_type)
    yield
    torch.set_default_tensor_type(old_tensor_type)

def crop_and_concat(upsampled, bypass, crop=False):
    """
    This layer crop the layer from contraction block and concat it with expansive block vector
    """
    if crop:
      c = (bypass.size()[2] - upsampled.size()[2]) // 2
      bypass = F.pad(bypass, (-c, -c, -c, -c))

    return torch.cat((upsampled, bypass), 1)

def reprojectMaskFeedback(mask, 
                          depth=None,
                          odometry=None,
                          intrinsics=None,
                          extrinsics=None,
                          mask_reproj_model=None):

    if odometry is not None:
      for batch in range(mask.shape[0]):
        with set_default_tensor_type(torch.cuda.FloatTensor):
          if intrinsics is not None:
            mask_reproj_model.set_cam_model(intrinsics[batch])
          if extrinsics is not None:
            mask_reproj_model.set_cam_extrinsics(extrinsics[batch])
          mask[batch,0] = mask_reproj_model(mask[batch,0],
                                            depth_img=depth[batch,0]*0.001,
                                            odometry=odometry[batch])

    return mask

def reprojectTensorFeedback(mask, 
                            depth=None,
                            odometry=None,
                            intrinsics=None,
                            extrinsics=None,
                            tensor_reproj_model=None):
  
  if odometry is not None:
    # Reproject mask batch-wise
    # TODO: add batch processiong support to reprojectorch.py in agrobot-utils
    batch_reproj = lambda mask, depth, odometry, intrinsics, extrinsics: \
      tensor_reproj_model(mask[0], depth_img=depth[0],
                          odometry=odometry,
                          cam_model=intrinsics,
                          cam_extrinsics=extrinsics,
                          return_depth=True)
    with set_default_tensor_type(torch.cuda.FloatTensor):
      # unbind batches to reproject batch-wise
      batch_tensors = map(lambda x:torch.unbind(x,0), (mask, depth/1000, odometry, intrinsics, extrinsics))
      # reproject masks with odometry and reprojector and get pixel-wise shifts
      feat_shifts = list(map(batch_reproj, *batch_tensors))
      feat_shifts = torch.stack(feat_shifts, 0)
      feat_shifts = torch.movedim(feat_shifts,-1,1)

  return feat_shifts

def round_kernel_size(kernel_size):
  if kernel_size < 3:
    kernel_size = 3
    warnings.warn(f'kernel size is lower than 3, using size 3x3 instead')
  elif kernel_size % 2 == 0:
    kernel_size += 1
    warnings.warn(f'kernel size is even, using size {kernel_size}x{kernel_size} instead')
  return kernel_size

def matchSize(inputFeatures, refFeatures, mode='bilinear'):
  
  outFeatures = F.interpolate(inputFeatures,
                              size=(refFeatures.shape[2], refFeatures.shape[3]),
                              mode=mode, #F.InterpolationMode(mode),
                              align_corners=False)


  return outFeatures


def interpolateShiftMat(shiftMat, height, width, mode='bilinear'):
  # Compute pixel shift scaling 
  scale_h = shiftMat.shape[2] / height
  scale_w = shiftMat.shape[3] / width
  # Interpolate shifts
  dst_shifts = F.interpolate(shiftMat,
                              size=(height, width),
                              mode=mode,
                              align_corners=False)
  # Rescale shifts ot interpolated size
  dst_shifts[:,0] /= scale_h
  dst_shifts[:,1] /= scale_w
  return dst_shifts

def matchShiftMatSize(shiftMat, dst_feature, mode='bilinear'):
  return interpolateShiftMat(shiftMat,
                             height=dst_feature.shape[2],
                             width=dst_feature.shape[3],
                             mode=mode)


def shiftTensor(features, shiftMat, mask, default_value=0.1):
    ### 1) Shifts downsampling and build image outliers mask
    ################################################################
    # downsample shift matrix and rescale pixel shifts
    # [batch,[dy,dx,depth],y,x]
    dst_shifts = matchShiftMatSize(shiftMat, features)
    # compute destination coordiantes and zero out mask for points outside of the image
    # [batch,[y_dst,x_dst,depth],y,x]
    dst_shifts[:,0] += torch.arange(dst_shifts.shape[2]).reshape(-1,1).to(torch.device(dst_shifts.device.index))
    dst_shifts[:,1] += torch.arange(dst_shifts.shape[3]).to(torch.device(dst_shifts.device.index))
    dst_coords = dst_shifts
    # Create a mask for the oixels that have a dst coordinate inside of the image
    h_valid_mask = torch.logical_and(dst_coords[:,0,:,:] > 0 ,dst_coords[:,0,:,:] < dst_shifts.shape[2])
    w_valid_mask = torch.logical_and(dst_coords[:,1,:,:] > 0 ,dst_coords[:,1,:,:] < dst_shifts.shape[3])
    hw_valid_mask = torch.logical_and(h_valid_mask, w_valid_mask)

    ### 2) Mask downsampling and filter image outliers
    ################################################################
    # downsample mask
    mask = F.interpolate(mask,
                         size=tuple(features.shape[2:]),
                         mode='bilinear',
                         align_corners=False)
    # Encode batch coordinate in masks
    for batch in range(mask.shape[0]):
      mask[batch,mask[batch]>0] += batch
    # remove points from mask with invalid dst coordinates
    mask = torch.where(hw_valid_mask.unsqueeze(1), mask, torch.zeros_like(mask))
    # remove points with invalid depth
    mask = torch.where(dst_coords[:,2].unsqueeze(1) > 0, mask, torch.zeros_like(mask))

    ### 3) Cat augmented featuremap with reprojection information
    ################################################################
    feat_cat = torch.cat((mask, dst_coords,features),dim=1)
    # Change dimention order to simplify channels spatial shift below
    # [batch,ch,y,x] -> [batch,y,x,ch]
    feat_cat = torch.movedim(feat_cat,1,-1)
    # Fill the output thensor with the default value
    feat_shifted = torch.ones_like(feat_cat[:,:,:,4:]) * default_value
    # Keep only feature maps with masks != 0 and flatten
    # [batch,h,w,ch] -> [n_valid_pts, [batch+1, y_coord, x_coord, depth, ch]]
    feat_flat = feat_cat[feat_cat[:,:,:,0]>0]
    # Sort according to decending depth, this makes that only the closesp point
    # to the camera gets shifted
    # [batch,h,w,ch] -> [n_valid_pts, [batch+1, y_coord, x_coord, depth, ch]]
    feat_flat = feat_flat[torch.argsort(feat_flat[:,3], descending=True)]
    # decrease mask vaues to use as index
    # [n_valid_pts, [batch+1, y_coord, x_coord, ch]] -> [n_valid_pts, [batch, y_coord, x_coord, depth, ch]]
    feat_flat[:,0] -= 1

    ### 4) Shift feature map and return with original size
    ################################################################
    # Shift features using the batch and destination coordinates
    feat_shifted[tuple((feat_flat[:,:3].T.type(torch.int)).detach().cpu().numpy())] = feat_flat[:,4:]
    # Move channels dimention back to pytorch's default and return
    return torch.movedim(feat_shifted,-1,1)

###
# image debugging
###

import cv2
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.cm as cm

def blend_img_and_tensor(img,*tensors, window_name='blend'):
    class show_updater():
      def __init__(self, img, tensors, blend=50, t_num=0, channel=0, b_num=0):
        self.img = img.moveaxis(1,0)[[2,1,0]].moveaxis(0,-1).detach().cpu().numpy()
        self.tensors = tensors

        self.blend = blend
        self.t_ch = channel
        self.t_num = t_num
        self.b_num = b_num


      def next_channel(self):
        self.t_ch = min(self.t_ch+1, self.tensors[self.t_num].shape[1]-1)
        self.update()
      def prev_channel(self):
        self.t_ch = max(self.t_ch-1, 0)
        self.update()
      def next_tensor(self):
        self.t_num = min(self.t_num+1, len(self.tensors)-1)
        self.update()
      def prev_tensor(self):
        self.t_num = max(self.t_num-1, 0)
        self.update()

      
      def blend_change(self, val):
        self.blend = val
        self.update()
      def channel_change(self, val):
        self.t_ch = val
        self.update()
      def batch_change(self, val):
        self.b_num = val
        self.update()
      def tensor_change(self, val):
        self.t_num = val
        self.update()

      def update(self):      
        tensor = self.tensors[self.t_num]
        img = self.img[self.b_num]
        if len(tensor.shape) == 3:
          tensor = torch.unsqueeze(tensor,1)

        img_resized = cv2.resize(img,tensor.shape[2:][::-1]) 
        tensor = tensor[self.b_num].detach().cpu().numpy()   
    
        alpha = self.blend/100
        beta = (1.0 - alpha)

        clamped_ch = min(self.t_ch, tensor.shape[0]-1)
        print(f'Image blended with tensor #{self.t_num} batch:{self.b_num} ch:{clamped_ch}')
        cv2.setTrackbarMax('tensor_ch', window_name, tensor.shape[0])
        cv2.setTrackbarPos('tensor_ch', window_name, self.t_ch)
        cv2.setTrackbarPos('tensor', window_name, self.t_num)

        # activation = np.dstack(3*[tensor[t_channel]])
        activation = tensor[clamped_ch]
        activation = cm.ScalarMappable(norm=mcolors.Normalize(vmin=0.5, vmax=1.15), cmap=cm.cool).to_rgba(activation)[:,:,0:3].astype(img_resized.dtype)
        activation [tensor[clamped_ch] > 4] = np.array([0,0,0])
        activation = cv2.cvtColor(activation, cv2.COLOR_RGB2BGR)
        result = cv2.addWeighted(img_resized, alpha, activation, beta, 0.0)
        cv2.imshow(window_name, result)

    plotter = show_updater(img,tensors)


    cv2.namedWindow(window_name,cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1000,1000)

    plotter.update()
    cv2.createTrackbar('img_blend', window_name, 30, 100, plotter.blend_change)

    if tensors[plotter.t_num].shape[0] > 1:
      cv2.createTrackbar('batch', window_name, 0, tensors[plotter.t_num].shape[0]-1, plotter.batch_change)
    
    cv2.createTrackbar('tensor_ch', window_name, 0, tensors[plotter.t_num].shape[1], plotter.channel_change)
    if isinstance(tensors, (list,tuple)) and len(tensors) > 1:
      cv2.createTrackbar('tensor', window_name, 0, len(tensors)-1, plotter.tensor_change)

    key = 0
    while True:
      k = cv2.waitKey(0)
      if k == ord('s'):    # S next annotated image
        break

      if k == ord('a'):    # A previous channel in tensor
        plotter.prev_channel()
      if k == ord('d'):    # D next channel in tensor
        plotter.next_channel()

      if k == ord('q'):    # Q previous tensor in arguments
        plotter.prev_tensor()
      if k == ord('e'):    # E next tensor in arguments
        plotter.next_tensor()
        

      if k ==27:    # Esc key to stop
        import sys; sys.exit(1)
    cv2.destroyAllWindows()



def show_imgs(imgList):
      import cv2
      cv2.namedWindow('img',cv2.WINDOW_NORMAL)
      imgStack = np.hstack([np.array(im[0][[2,1,0]].permute(1,2,0).detach().cpu()) for im in imgList])
      cv2.imshow('img', imgStack)
      cv2.waitKey(0)

def show_labels(labelList):
      import cv2
      cv2.namedWindow('label',cv2.WINDOW_NORMAL)
      labelStack = np.hstack([(np.array(im[0].detach().cpu()) > 0).astype(float) for im in labelList])
      cv2.imshow('label', labelStack)
      cv2.waitKey(0)