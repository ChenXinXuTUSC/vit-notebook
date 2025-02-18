import numpy as np

def R_mat(axis: np.ndarray, angle: float, radian: bool=True):
    """
    generate the 3x3 rotation matrix with the specified axis and angle

    Params
    -
        - axis: np.ndarray, [3, 1] or [3]
        - angle: float, rotation angle, in radian
        - radian: bool, angle in radian or degree
    
    Return
    -
        - R: np.ndarray, [3, 3] rotation matrix
    """
    # angle to radian
    theta = angle
    if not radian:
        theta = np.radians(angle)

    # normalize rotation axis
    if not isinstance(axis, np.ndarray):
        axis = np.array(axis)
    axis = axis / np.linalg.norm(axis)

    # 构造叉积矩阵
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])

    # 使用罗德里格斯公式计算旋转矩阵
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)

    return R

def T_vec(values: list):
    return np.array(values).reshape((-1, 1))

def A2B_R(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    '''
    Rotation Matrix to transform direction A to B (DeepSeek generated)
    
    Params
    -
        - A: np.ndarray, [1, 3]
        - B: np.ndarray, [1, 3]
    
    Return
    -
        - R: np.ndarray, [3, 3]
    '''
    assert isinstance(A, np.ndarray) or isinstance(A, list)
    assert isinstance(B, np.ndarray) or isinstance(B, list)
    if not isinstance(A, np.ndarray):
        A = np.array(A)
    if not isinstance(B, np.ndarray):
        B = np.array(B)
    if len(A.shape) > 1:
        A = A.flatten()
    if len(B.shape) > 1:
        B = B.flatten()
    assert len(A) == 3
    assert len(B) == 3

    # 归一化向量
    A = A / np.linalg.norm(A)
    B = B / np.linalg.norm(B)
    
    # 计算旋转轴
    k = np.cross(A, B)
    k_norm = np.linalg.norm(k)
    
    # 如果A和B平行或反平行，返回单位矩阵或180度旋转矩阵
    if k_norm < 1e-6:
        if np.dot(A, B) > 0:
            return np.eye(3)
        else:
            # 选择任意正交的旋转轴，例如单位z轴
            # 但需要确保A和选择的轴不平行
            axis = np.array([1, 0, 0])
            R = np.eye(3) - 2 * np.outer(axis, axis)
            return R
    
    # 归一化旋转轴
    k = k / k_norm
    
    # 计算旋转角度
    theta = np.arccos(np.clip(np.dot(A, B), -1.0, 1.0))

    return R_mat(k, theta)
