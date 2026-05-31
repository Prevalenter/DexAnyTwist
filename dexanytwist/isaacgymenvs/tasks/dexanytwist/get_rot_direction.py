import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List

def get_joint_axis(urdf_path: str, joint_name: str) -> List[float]:
    """
    从 URDF 文件中提取指定关节的 axis（xyz 向量）。
    
    Args:
        urdf_path: URDF 文件路径
        joint_name: 关节名字，比如 "joint1"
        
    Returns:
        长度为 3 的 float 列表 [x, y, z]
    """
    urdf_path = Path(urdf_path)
    if not urdf_path.is_file():
        raise FileNotFoundError(f"URDF file not found: {urdf_path}")

    tree = ET.parse(urdf_path)
    root = tree.getroot()

    for joint in root.findall("joint"):
        if joint.get("name") == joint_name:
            axis_elem = joint.find("axis")
            if axis_elem is None:
                raise ValueError(f"Joint '{joint_name}' has no <axis> element.")
            xyz_str = axis_elem.get("xyz")
            if xyz_str is None:
                raise ValueError(f"Joint '{joint_name}' <axis> has no 'xyz' attribute.")

            axis = [float(v) for v in xyz_str.split()]
            if len(axis) != 3:
                raise ValueError(f"Joint '{joint_name}' axis is not length 3: {xyz_str}")
            return axis

    raise ValueError(f"Joint '{joint_name}' not found in URDF.")



def get_xy_axis(urdf_path: str, joint_name: str = "joint1_rx") -> List[float]:
    """
    从 URDF 文件中提取指定关节的 axis（xyz 向量）。
    
    Args:
        urdf_path: URDF 文件路径
        joint_name: 关节名字，比如 "joint1"
        
    Returns:
        长度为 3 的 float 列表 [x, y, z]
    """
    urdf_path = Path(urdf_path)
    if not urdf_path.is_file():
        raise FileNotFoundError(f"URDF file not found: {urdf_path}")

    tree = ET.parse(urdf_path)
    root = tree.getroot()

    for joint in root.findall("joint"):
        if joint.get("name") == joint_name:
            axis_elem = joint.find("limit")
            if axis_elem is None:
                raise ValueError(f"Joint '{joint_name}' has no <limit> element.")
            upper_str = axis_elem.get("upper")
            if upper_str is None:
                raise ValueError(f"Joint '{joint_name}' <axis> has no 'upper' attribute.")

            return float(upper_str)
