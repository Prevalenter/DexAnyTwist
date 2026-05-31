# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.

import numpy as np

if not hasattr(np, "float"):
    np.float = float

import isaacgym

from .dexanytwist.DexAnyTwist import DexAnyTwist


isaacgym_task_map = {
    "DexAnyTwist": DexAnyTwist,
}
