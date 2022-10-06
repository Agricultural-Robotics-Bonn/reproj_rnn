# -*- coding: utf-8 -*-
# @Author: Claus Smitt
# @Date:   2020-09-10 14:55:33
# @Last Modified by:   lvisroot
# @Last Modified time: 2020-09-24 14:31:55
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division

from enum import Enum
from pycocotools import coco

import torch
from torch import nn

class PxCoordOrder(Enum):
  uv = 1
  vu = 2

def reproject(points, depth_img, cam_transform,
              cam_model_src=None, cam_model_dst=None,
              ret_valid_src=False,
              ret_depth=False,
              depth_vector=None):
  """Summary
  reproject sets of pixels between cameras
  Args:
      points (torch.array [2,N]): array on 2D point in the src image coordinates
      depth_img (torch.array(torch.float32) [h,w]): [h,w] depth image in meters
      cam_transform (torch.array [4x4]): Homogeneous transformation between cameras
      cam_model_src (None, optional): Src camera linear intrinsics matrix
      cam_model_dst (None, optional): Dst camera linear intrinsics matrix

  Returns:
      torch.array [2,M]: M valid reprojected points
  """
  ret_size = 2
  if ret_depth:
    ret_size = 3
  zero_ret = torch.Tensor(ret_size,0)

  if points.nelement() == 0:
    if ret_valid_src:
      return zero_ret, zero_ret
    return zero_ret

  if cam_model_src is None:
    cam_model_src = torch.eye(3)
    cam_model_src[:-1,2] = torch.Tensor([depth_img.shape[1], depth_img.shape[0]])
  if cam_model_dst is None:
    cam_model_dst = cam_model_src

  # 1) Undistort points if using cam_model_src if available
  p_src = torch.ones_like(points).type(torch.float32) #[2xN]
  p_src[0,:] = (points[0,:] - cam_model_src[0,2]) / cam_model_src[0,0]
  p_src[1,:] = (points[1,:] - cam_model_src[1,2]) / cam_model_src[1,1]

  # 2) Unproject undist-points using depth_img
  # P_src = torch.vstack( (p_src, torch.ones(p_src.shape[1])) ) * depth_img[tuple( points[[1,0],:])] #[3xN]
  if depth_vector is not None:
    P_src = torch.cat( (p_src, torch.ones( (1,p_src.shape[1]) )), dim=0 ) * depth_vector #[3xN]
  else:
    P_src = torch.cat( (p_src, torch.ones( (1,p_src.shape[1]) )), dim=0 ) * depth_img[tuple( points[[1,0],:])] #[3xN]
  # remove invalid depth points
  q_valid_src = points[:,P_src[2,:] != 0.0] #[2xN]
  P_src = P_src[:,P_src[2,:] != 0.0] #[3xN]

  if P_src.nelement() == 0:
    if ret_valid_src:
      return zero_ret, zero_ret
    return zero_ret

  # 3) Transform from src to dst camera
  P_dst = (torch.matmul(cam_transform[:3,:3], P_src).T + cam_transform[:3,3]).T #[3xN]

  # 4) Project to dst camera
  q_dst = torch.ones((2,P_dst.shape[1])).type(torch.float32) #[2xN]
  q_dst[0,:] = (P_dst[0,:]/P_dst[2,:]) * cam_model_dst[0,0] + cam_model_dst[0,2]
  q_dst[1,:] = (P_dst[1,:]/P_dst[2,:]) * cam_model_dst[1,1] + cam_model_dst[1,2]

  if ret_depth:
    q_dst = torch.vstack((q_dst, P_dst[2,:]))
    q_valid_src = torch.vstack((q_valid_src, P_src[2,:]))

  if q_dst.nelement() == 1:
    q_dst = q_dst.reshape(ret_size,1)

  if ret_valid_src:
   return q_dst, q_valid_src
  return q_dst

def compute_clip_mask(points, width, height):
    pts_clip_mask_u = torch.logical_and(points[0,:] >= 0,
                                    points[0,:] < width)
    pts_clip_mask_v = torch.logical_and(points[1,:] >= 0,
                                    points[1,:] < height)
    return torch.logical_and(pts_clip_mask_v,pts_clip_mask_u)

def clip_points(points, width, height):
    # Return empty mask or array if no points were reprojected
    if points.nelement() == 0:
      return points
    # remove points outside of FOV
    pts_clip_mask = compute_clip_mask(points, width, height)
    return points[:,pts_clip_mask]

def compute_shift_matrix(pts_src, pts_dst, img_size, clip=False):
    shifts_mat = torch.zeros(*img_size, pts_src.shape[0])

    if pts_src.nelement() == 0 or pts_dst.nelement() == 0 or pts_src.shape != pts_dst.shape:
      return shifts_mat

    if clip:
      clip_mask = compute_clip_mask(pts_dst, img_size[1], img_size[0])
      pts_src = pts_src[:, clip_mask]
      pts_dst = pts_dst[:, clip_mask]

    if pts_src.nelement() == 0 or pts_dst.nelement() == 0:
      return shifts_mat
    # u,v shift for each pixel
    shift_pts = pts_dst[0:2] - pts_src[0:2]
    # Copy dst depth to shift matrix 3 dimension for convenience
    if pts_src.shape[0] == 3:
      shift_pts = torch.vstack((shift_pts, pts_dst[2]))
    # copy shift to original mask pixel coordinates
    shifts_mat[tuple(pts_src.type(torch.long)[0:2,:][[1,0]])] = shift_pts.T

    return shifts_mat

def reproject_and_clip(points, depth_img, cam_transform, cam_model_src=None, cam_model_dst=None, dst_img_size=None, depth_vector=None):

    rep_pts = reproject(points=points,
                        depth_img=depth_img,
                        cam_transform=cam_transform,
                        cam_model_src=cam_model_src,
                        cam_model_dst=cam_model_dst,
                        depth_vector=None)
    # Clip reprojected points outside of the dst FOV
    img_size = dst_img_size or depth_img.shape
    return clip_points(rep_pts, width=img_size[1], height=img_size[0])

def mask_to_pts(mask):
  pts = torch.nonzero(mask)
  return pts.T[[1,0]] if pts.nelement() else pts


def pts_to_mask(pts, img_size):
  rep_mask = torch.zeros(tuple(img_size))
  # Create reprojected mask from points
  if pts.nelement() != 0 :
    pts = clip_points(pts, width=img_size[1], height=img_size[0])
    rep_mask[tuple(pts[[1,0],:].type(torch.long))] = 1
  return rep_mask

def pts_to_depth_image(pts, img_size):
  depth_img = torch.zeros(tuple(img_size))
  # Create reprojected mask from points
  if pts.nelement() != 0 and pts.shape[1] > 2:
    pts = clip_points(pts, width=img_size[1], height=img_size[0])
    depth_img[tuple(pts[[1,0],:].type(torch.long))] = pts[2] 
  return depth_img

class Reprojectorch(nn.Module):
  """docstring for Reprojector
  """
  def __init__(self, cam_model= None, cam_model_src=None, cam_model_dst=None,
               cam_extrinsics=None,
               static_transform=False,
               dst_image_size=None,
               coord_order=PxCoordOrder.uv):
    """Summary
    reproject sets of pixels between cameras
    Args:
        cam_model_src (torch.Tensor [3,3]) (None, optional):
          Src camera linear intrinsics matrix
        cam_model_dst (torch.Tensor [3,3]):
          Dst camera linear intrinsics matrix
        cam_extrinsincs (torch.Tensor [4,4]):
          Odom to cam extrinsics transform
    Returns:
        torch.Tensor [2,M]: M valid reprojected points
    """
    if not isinstance(coord_order,PxCoordOrder):
      raise ValueError("Coordinate order argument must be of type PxCoordOrder")
    self.coord_order = coord_order

    # init pytroch module
    super(Reprojectorch, self).__init__()

    if cam_model is None:
      if cam_model_src is None:
        cam_model = torch.eye(3)
      else:
        cam_model = cam_model_src
    self.register_buffer('cam_model_src', cam_model)

    if cam_model_dst is None:
      cam_model_dst = self.cam_model_src
    self.register_buffer('cam_model_dst', cam_model_dst)


    if dst_image_size is None:
      dst_image_size = torch.Tensor([])
    self.register_buffer('dst_img_size', dst_image_size)# [height, width]


    if cam_extrinsics is None:
      cam_extrinsics = torch.eye(4)
    self.register_buffer('extrinsics', cam_extrinsics)
    self.register_buffer('i_extrinsics', torch.inverse(self.extrinsics))

    self.static_transform = static_transform


  def _compute_cam_transform(self,odometry):
    if self.static_transform:
      return self.extrinsincs
    else:
      # return the same mask if transform isn't static and odometry is 0
      if torch.equal(odometry, torch.eye(4)) or odometry is None:
        return None
      # Compute camera transform
      return torch.matmul(self.i_extrinsics, torch.matmul(torch.inverse(odometry), self.extrinsics))

  def reproject_points(self, points, depth_img, odometry=None,
                       clip_invalid_pts=True,
                       return_mask=False,
                       return_shift_matrix=False,
                       return_depth=False,
                       depth_vector=None,
                       dst_img_size=None,
                       **Kargs):
    # prepear return dict and image size
    ret={}
    
    if dst_img_size is not None:
      img_size = dst_img_size
    else:
      img_size = self.dst_img_size if self.dst_img_size.nelement() != 0 else depth_img.shape
    
    ret_size = 3 if return_depth else 2

    cam_transform = self._compute_cam_transform(odometry)
    # Return the same mask if no transform is to be applied
    if cam_transform is None:
      if return_shift_matrix:
        ret['shifts'] = compute_shift_matrix(torch.Tensor(ret_size,0),
                                             torch.Tensor(ret_size,0),
                                             img_size)
      if return_depth or return_mask:
        pts_mask = pts_to_mask(points, img_size)
      if return_depth:
        ret['depth'] = torch.where(pts_mask>0, depth_img, torch.zeros_like(depth_img))
      if return_mask:
        ret['mask'] = pts_mask
      
      if self.coord_order == PxCoordOrder.vu:
        points[[0,1]] = points[[1,0]]
      if ret:
        ret['points'] = points
      else:
        ret = points
      return ret
    # Reproject mask
    pts_rep = reproject(points=points,
                        depth_img=depth_img,
                        cam_transform=cam_transform,
                        cam_model_src=self.cam_model_src,
                        cam_model_dst=self.cam_model_dst,
                        ret_valid_src=return_shift_matrix,
                        ret_depth=return_depth,
                        depth_vector=depth_vector)
    if return_shift_matrix:
      # unpack reprojected and original points
      pts_rep, pts_src = pts_rep
      ret['shifts'] = compute_shift_matrix(pts_src=pts_src,
                                           pts_dst=pts_rep,
                                           img_size=img_size,
                                           clip=clip_invalid_pts)
      if self.coord_order == PxCoordOrder.vu:
        ret['shifts'][:,:,[0,1]] = ret['shifts'][:,:,[1,0]]

    if clip_invalid_pts:
      pts_rep = clip_points(pts_rep,
                            width=img_size[1],
                            height=img_size[0])
    if return_mask:
      ret['mask'] = pts_to_mask(pts_rep, img_size)
    if return_depth:
      ret['depth'] = pts_to_depth_image(pts_rep, img_size)

    if self.coord_order == PxCoordOrder.vu:
      pts_rep[[0,1]] = pts_rep[[1,0]]
    # Add points to return dictionary if many outputs are required
    if ret:
      ret['points'] = pts_rep
    else:
      ret = pts_rep
    return ret

  ## Specific wrappers computation
  ##################################
  def reproject_mask(self, mask, depth_img, odometry=None,
                     clip_invalid_pts=True,
                     return_points=False,
                     return_shift_matrix=False,
                     return_depth=False,
                     **Kargs):

    ret = self.reproject_points(mask_to_pts(mask), depth_img,
                                odometry=odometry,
                                clip_invalid_pts=clip_invalid_pts,
                                return_mask=True,
                                return_shift_matrix=return_shift_matrix,
                                return_depth=return_depth)

    if not return_shift_matrix and not return_points:
      return ret['mask']
    return ret

  def shift_matrix_from_mask(self, mask, depth_img, odometry=None,
                             clip_invalid_pts=True,
                             return_depth=False,
                             **Kargs):

    return self.reproject_mask(mask, depth_img, odometry=odometry,
                               clip_invalid_pts=clip_invalid_pts,
                               return_shift_matrix=True,
                                 return_depth=return_depth)['shifts']

  def depth_shift_matrix_from_mask(self, mask, depth_img, odometry=None,
                                   clip_invalid_pts=True,
                                   return_depth=False,
                                   **Kargs):
    
    ret = self.reproject_mask(mask, depth_img, odometry=odometry,
                               clip_invalid_pts=clip_invalid_pts,
                               return_shift_matrix=True,
                               return_depth=True)

    return torch.cat((ret['shifts'], ret['depth'].unsqueeze(-1)), -1)

  def shift_matrix_from_points(self, points, depth_img, odometry=None,
                               clip_invalid_pts=True,
                               return_depth=False,
                               **Kargs):

    return self.reproject_points(points, depth_img, odometry=odometry,
                                 clip_invalid_pts=clip_invalid_pts,
                                 return_shift_matrix=True,
                                 return_depth=return_depth)['shifts']

  def reproject_mask_to_points(self, mask, depth_img, odometry=None,
                               clip_invalid_pts=True,
                               return_depth=False,
                               **Kargs):

    return self.reproject_mask(mask, depth_img, odometry=odometry,
                               clip_invalid_pts=clip_invalid_pts,
                               return_points=True,
                                 return_depth=return_depth)['points']

  def reproject_points_to_mask(self, points, depth_img, odometry=None,
                               clip_invalid_pts=True,
                               return_depth=False,
                               **Kargs):

    return self.reproject_points(points, depth_img, odometry=odometry,
                                 clip_invalid_pts=clip_invalid_pts,
                                 return_mask=True,
                                 return_depth=return_depth)['mask']

  def reproject_points_with_depth(self, points, dst_img_size, odometry=None,
                                  clip_invalid_pts=True,
                                  return_depth=False,
                                  **Kargs):
    """Reprojects points with depth in the 3rd coordinate of the points array

    Args:
        points (torch.Tensor[3,N]): Points in the camera plane with depth on the 3rd coordinate
        dst_img_size (torch.Tensor[2]): Destination image size used to clip invalid points
        odometry (torch.Tensor[4,4], optional): Odometry between frames to reproject. Defaults to None.
        clip_invalid_pts (bool, optional): Wether to clip points reprojected outside of the destination frame. Defaults to True.
        return_depth (bool, optional): Weather to return depth in the 3rd coordinate of the output. Defaults to False.

    Returns:
        torch.Tensor[3,N]: Reprojected points
    """
    if points.shape[0] != 3 or len(points.shape) != 2:
      raise(f'Points are expected to be of size [3,N] but got {points.shape}')

    return self.reproject_points(points=points[:2],
                                 depth_img=None,
                                 odometry=odometry,
                                 clip_invalid_pts=clip_invalid_pts,
                                 return_depth=return_depth,
                                 dst_img_size=dst_img_size,
                                 depth_vector=points[2])

    depth_vector = points[2] if points[2].shape[0] == 3 else None
  

  def set_cam_model(self, cam_model=None, **Kargs):
    if cam_model is not None:
      self.cam_model_src =  cam_model.to(self.cam_model_src.device)
      self.cam_model_dst = cam_model.to(self.cam_model_dst.device)
  def set_cam_model_src(self, cam_model_src=None, **Kargs):
    if cam_model_src is not None:
      self.cam_model_src = cam_model_src.to(self.cam_model_src.device)
  def set_cam_model_dst(self, cam_model_dst=None, **Kargs):
    if cam_model_dst is not None:
      self.cam_model_dst = cam_model_dst.to(self.cam_model_dst.device)
  def set_cam_extrinsics(self, cam_extrinsics=None, **Kargs):
    if cam_extrinsics is not None:
      self.extrinsics = cam_extrinsics.to(self.extrinsics.device)
      self.i_extrinsics = torch.inverse(cam_extrinsics).to(self.i_extrinsics.device)
  
  def set_cam_params(self, **Kargs):
    self.set_cam_model(**Kargs)
    self.set_cam_model_src(**Kargs)
    self.set_cam_model_dst(**Kargs)
    self.set_cam_extrinsics(**Kargs)

class ReprojPointsModel(Reprojectorch):
  def forward(self, *args, **Kargs):
    self.set_cam_params(**Kargs)
    return self.reproject_points(*args, **Kargs)
class ReprojMaskModel(Reprojectorch):
  def forward(self, *args, **Kargs):
    self.set_cam_params(**Kargs)
    return self.reproject_mask(*args, **Kargs)
class ReprojMaskToShiftMatrix(Reprojectorch):
  def forward(self, *args, **Kargs):
    self.set_cam_params(**Kargs)
    return self.shift_matrix_from_mask(*args, **Kargs)
class ReprojMaskToDepthShiftMatrix(Reprojectorch):
  def forward(self, *args, **Kargs):
    self.set_cam_params(**Kargs)
    return self.depth_shift_matrix_from_mask(*args, **Kargs)
class ReprojPointsToShiftMatrix(Reprojectorch):
  def forward(self, *args, **Kargs):
    self.set_cam_params(**Kargs)
    return self.shift_matrix_from_points(*args, **Kargs)
class ReprojMaskToPoints(Reprojectorch):
  def forward(self, *args, **Kargs):
    self.set_cam_params(**Kargs)
    return self.reproject_mask_to_points(*args, **Kargs)
class ReprojPointsToShiftMatrix(Reprojectorch):
  def forward(self, *args, **Kargs):
    self.set_cam_params(**Kargs)
    return self.reproject_points_to_mask(*args, **Kargs)
class ReprojPointsWithDepth(Reprojectorch):
  def forward(self, *args, **Kargs):
    """Reprojects points with depth in the 3rd coordinate of the points array

    Args:
        points (torch.Tensor[3,N]): Points in the camera plane with depth on the 3rd coordinate
        dst_img_size (torch.Tensor[2]): Destination image size used to clip invalid points
        odometry (torch.Tensor[4,4], optional): Odometry between frames to reproject. Defaults to None.
        clip_invalid_pts (bool, optional): Wether to clip points reprojected outside of the destination frame. Defaults to True.
        return_depth (bool, optional): Weather to return depth in the 3rd coordinate of the output. Defaults to False.

    Returns:
        torch.Tensor[3,N]: Reprojected points
    """
    self.set_cam_params(**Kargs)
    return self.reproject_points_with_depth(*args, **Kargs)
