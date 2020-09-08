import importlib
import apply_mask
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

from tools import validate_inputs

grp = "examples/groups/group1.jpg"
sf = 'examples/faces/sideface.jpg'
frm = 'from'
mm = 'from/Marilyn_Monroe_photo_pose_Seven_Year_Itch.jpg'
settings = {"input":frm, 'mask':'examples/virusmasks', 'output':'to', 'unique':False,'pose':True}
settings_obj = dotdict(settings)
input_paths, masks, output_paths, apply_unique_masks, is_video,use_pose_detection  =  validate_inputs.run(settings_obj)


apply_mask.apply_masks(input_paths, masks, output_paths, apply_unique_masks, is_video,use_pose_detection)


