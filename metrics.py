import numpy as np

def mpjpe(pred, true):
    '''
    Assume both inputs are (24,2) numpy arrays
    Compute the mean of Euclidean distance for all joints 
    '''
    return sum(np.linalg.norm(pred - true, axis=1)) / 24

def pck(pred, true, thereshold=20):
    '''
    Compute the percentage of correctly idenfied joints
    Correcly identified = predicted value within true value +- thereshold (in pixels)
    '''
    count = 0
    for i in range(24):
        pred_joint = pred[i,:]
        true_joint = true[i,:]
        if abs(pred_joint[0] - true_joint[0]) <= thereshold and abs(pred_joint[1] - true_joint[1]) <= thereshold:
            count += 1
    return count / 24

def pcp(pred, true, thereshold=0.5):
    '''
    Percentage of correct parts
    thereshold = 0.5 * limb length by default
    If start point and end point are a both within thereshold, the set of joints are classified as correct
    '''
    bones = [[0,1], [1,2], [2,3], [3,4], [4,5], [1,6], [6,7], [7,8], [8,9],
             [2,10], [10,11], [11,12], [6,13], [13,14], [14,15],
             [16,17], [17,18], [18,19], [12,17], [12,18],
             [20,21], [21,22], [22,23], [15,21], [15,22]]
    count = 0
    for bone in bones:
        start_idx, end_idx = bone
        true_start, true_end = true[start_idx, :], true[end_idx, :]
        pred_start, pred_end = pred[start_idx, :], pred[end_idx, :]
        true_limb_length = np.linalg.norm(true_end - true_start)
        start_joint_error = np.linalg.norm(pred_start - true_start)
        end_joint_error = np.linalg.norm(pred_end - true_end)
        if start_joint_error <= thereshold * true_limb_length and end_joint_error <= thereshold * true_limb_length:
            count += 1
    return count / len(bones)