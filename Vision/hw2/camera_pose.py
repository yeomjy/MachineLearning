import numpy as np

from feature import EstimateE_RANSAC


def GetCameraPoseFromE(E):
    """
    Find four conﬁgurations of rotation and camera center from E

    Parameters
    ----------
    E : ndarray of shape (3, 3)
        Essential matrix

    Returns
    -------
    R_set : ndarray of shape (4, 3, 3)
        The set of four rotation matrices
    C_set : ndarray of shape (4, 3)
        The set of four camera centers
    """
    
    U, S, Vh = np.linalg.svd(E)
    t = U[:, 2]
    tt = -t

    Z = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    R_set = np.zeros((4, 3, 3))
    C_set = np.zeros((4, 3))

    R_set[0] = U @ W @ Vh
    C_set[0] = -R_set[0].T @ t

    R_set[1] = U @ W @ Vh
    C_set[1] = -R_set[1].T @ tt

    R_set[2] = U @ W.T @ Vh
    C_set[2] = -R_set[2].T @ t

    R_set[3] = U @ W.T @ Vh
    C_set[3] = -R_set[3].T @ tt
    return R_set, C_set



def Triangulation(P1, P2, track1, track2):
    """
    Use the linear triangulation method to triangulation the point

    Parameters
    ----------
    P1 : ndarray of shape (3, 4)
        Camera projection matrix 1
    P2 : ndarray of shape (3, 4)
        Camera projection matrix 2
    track1 : ndarray of shape (n, 2)
        Point correspondences from pose 1
    track2 : ndarray of shape (n, 2)
        Point correspondences from pose 2

    Returns
    -------
    X : ndarray of shape (n, 3)
        The set of 3D points
    """
    
    # TODO Your code goes here
    n = track1.shape[0]
    X = -np.ones((n, 3))

    for i in range(n):
        x1 = track1[i]
        x2 = track2[i]
        if np.all(x1 == [-1, -1]) or np.all(x2 == [-1, -1]):
            continue

        A = np.zeros((4, 4))

        A[0] = P1[2] * x1[0] - P1[0]
        A[1] = P1[2] * x1[1] - P1[1]
        A[2] = P2[2] * x2[0] - P2[0]
        A[3] = P2[2] * x2[1] - P2[1]

        U, S, Vh = np.linalg.svd(A)
        xx = Vh[-1]
        xx = xx/xx[3]

        X[i] = xx[0:3]



    return X



def EvaluateCheirality(P1, P2, X):
    """
    Evaluate the cheirality condition for the 3D points

    Parameters
    ----------
    P1 : ndarray of shape (3, 4)
        Camera projection matrix 1
    P2 : ndarray of shape (3, 4)
        Camera projection matrix 2
    X : ndarray of shape (n, 3)
        Set of 3D points

    Returns
    -------
    valid_index : ndarray of shape (n,)
        The binary vector indicating the cheirality condition, i.e., the entry 
        is 1 if the point is in front of both cameras, and 0 otherwise
    """
    
    # TODO Your code goes here
    n = X.shape[0]
    valid_index = np.zeros((n,))
    for i in range(n):
        x = X[i]
        if np.all(x == [-1, -1, -1]):
            continue

        xx = np.array([x[0], x[1], x[2], 1])
        x1 = P1 @ xx
        x2 = P2 @ xx

        if x1[2] > 0 and x2[2] > 0:
            valid_index[i] = 1



    return valid_index



def EstimateCameraPose(track1, track2):
    """
    Return the best pose conﬁguration

    Parameters
    ----------
    track1 : ndarray of shape (n, 2)
        Point correspondences from pose 1
    track2 : ndarray of shape (n, 2)
        Point correspondences from pose 2

    Returns
    -------
    R : ndarray of shape (3, 3)
        The rotation matrix
    C : ndarray of shape (3,)
        The camera center
    X : ndarray of shape (F, 3)
        The set of reconstructed 3D points
    """
    
    # TODO Your code goes here
    ransac_n_iter = 200
    ransac_thr = 0.01
    E, _ = EstimateE_RANSAC(track1, track2, ransac_n_iter, ransac_thr)
    R_set, C_set = GetCameraPoseFromE(E)

    max_idx = -1
    max_num = -1
    points = None

    for i in range(4):
        R_i, C_i = R_set[i], C_set[i]
        I = np.concatenate([np.eye(3), np.zeros((3, 1))], axis=1)
        Rt = np.concatenate([R_i, (-R_i @ C_i).reshape((3, 1))], axis=1)
        triangulated = Triangulation(I, Rt, track1, track2)
        cheirality = EvaluateCheirality(I, Rt, triangulated)

        if len(cheirality) > max_num:
            max_idx = i
            max_num = len(cheirality)
            points = triangulated

    R = R_set[max_idx]
    C = C_set[max_idx]
    X = points

    return R, C, X