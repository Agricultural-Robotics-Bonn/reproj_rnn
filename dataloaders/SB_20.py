from .reproj_rnn_dl import Parser as BaseParser


class Parser(BaseParser):
  def __init__(self, cfg):
    super().__init__(cfg)

    if self.sequencing is not None:
      if self.sequencing['odom_source'] == 'rgbd':
        self.sequencing['odometry_rel_file_path'] = '../rgbd_odom.csv'
      else:
        self.sequencing['odometry_rel_file_path'] = '../poseDict.csv'
    self.depth_rel_path='../depth'