import numpy as np
import os
import json

from isaacgymenvs.tasks.dexanytwist.get_rot_direction import get_joint_axis, get_xy_axis


def load_base_twist_config(cfg_dir_path="config/twist_anything/shampoo_bottle.json", with_points_cloud=False):
    twist_config_path = os.path.join(os.path.dirname(__file__), cfg_dir_path)
    with open(twist_config_path, 'r') as f:
        twist_config_data = json.load(f)
        
        twist_config = {}  
        for k, v in twist_config_data.items():
            twist_config[k] = {}
            twist_config[k]['pose'] = [v['dx'], v['dy']+0.03, v['dz']]
            twist_config[k]['urdf_path'] = v['urdf_path']
            twist_config[k]['kp_bais'] = np.array(v['kp_bais']) + np.array(v['kp_center'])
            
            if with_points_cloud:
                feature_root = '../assets'

                twist_config[k]['points_feature'] = np.load(f'{feature_root}/{v["points_feature"]}')[0]

            asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../assets')
            twist_config[k]['axis'] = get_joint_axis(os.path.join(asset_root, v['urdf_path']), 'joint1')[2]
            
            twist_config[k]['xy_axis_limit'] = get_xy_axis(os.path.join(asset_root, v['urdf_path']), 'joint1_rx')

        num_twist_cls = len(twist_config)
        twist_cls_list = list(twist_config.keys())
        
    return twist_config, num_twist_cls, twist_cls_list


def load_list_twist_config(cfg_dir_path_list=["config/twist_anything/shampoo_bottle.json"], with_points_cloud=False):
    twist_config_list = {}
    num_twist_cls_list = 0
    twist_cls_list_list = []
    for cfg_dir_path in cfg_dir_path_list:
        twist_config, num_twist_cls, twist_cls_list = load_base_twist_config(cfg_dir_path, with_points_cloud)
        twist_config_list.update(twist_config)
        num_twist_cls_list += num_twist_cls
        twist_cls_list_list += twist_cls_list
    return twist_config_list, num_twist_cls_list, twist_cls_list_list


def get_twist_config(dataset='all', with_points_cloud=False):
    preset_cls = [
        'bottle', 'bulb', 'cosmetic', 'nut', 'rotation_switch', 
        'screwdriver', 'shampoo_bottle', 'valve', 'liquor', 'others', 
    ]

    if dataset == 'twist_all':
        cls_list  = preset_cls
    elif dataset == 'twist_no_others':
        cls_list = [
        'bottle', 'bulb', 'cosmetic', 'nut', 'rotation_switch', 
        'screwdriver', 'shampoo_bottle', 'valve', 'liquor'
    ]
    elif dataset == 'twist_100':
        cls_list = preset_cls[5:]
    elif dataset == 'twist_single':
        cls_list = ['twist_single']
    elif dataset in preset_cls:
        cls_list = [dataset]
    else:
        raise ValueError(f"dataset {dataset} is not supported")

    twist_config_list, num_twist_cls_list, twist_cls_list_list = load_list_twist_config(
        cfg_dir_path_list=["config/twist_anything/{}.json".format(cls_name) for cls_name in cls_list],
        with_points_cloud=with_points_cloud)
    return twist_config_list, num_twist_cls_list, twist_cls_list_list

def counter_cls_from_ids(ids, twist_all_cls_list):
    
    preset_cls = [
        'bottle', 'bulb', 'cosmetic', 'nut', 'rotation_switch', 
        'screwdriver', 'shampoo_bottle', 'valve', 'liquor', 'others', 
    ]
    rst = {}
    for cls in preset_cls:
        _, _, twist_cls_list = get_twist_config(
            dataset=cls, with_points_cloud=False)
        rst[cls] = twist_cls_list

    counts = {}
    for id in ids:
        cls_name = twist_all_cls_list[id]
        for k in rst:
            if cls_name in rst[k]:
                counts[k] = counts.get(k, 0) + 1
                break
    return counts


if __name__ == '__main__':
    
    twist_config_list, num_twist_cls_list, twist_cls_list_list = get_twist_config(
        dataset='twist_all',
        with_points_cloud=False,
    )
    
    print(num_twist_cls_list)
