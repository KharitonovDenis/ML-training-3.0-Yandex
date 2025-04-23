import numpy as np
# do not change the code in the block below
# __________start of block__________
class DummyMatch:
    def __init__(self, queryIdx, trainIdx, distance):
        self.queryIdx = queryIdx  # index in des1
        self.trainIdx = trainIdx  # index in des2
        self.distance = distance
# __________end of block__________


def match_key_points_numpy(des1: np.ndarray, des2: np.ndarray) -> list:
    """
    Match descriptors using brute-force matching with cross-check.

    Args:
        des1 (np.ndarray): Descriptors from image 1, shape (N1, D)
        des2 (np.ndarray): Descriptors from image 2, shape (N2, D)

    Returns:
        List[DummyMatch]: Sorted list of mutual best matches.
    """
    # YOUR CODE HERE
    matches = []

    if des1.size == 0 or des2.size == 0:
        return matches
    
    distances = np.sqrt(np.sum((des1[:, np.newaxis] - des2) ** 2, axis=2))
    
    best_1_to_2 = np.argmin(distances, axis=1)
    best_2_to_1 = np.argmin(distances, axis=0)
    
    for i1, i2 in enumerate(best_1_to_2):
        if best_2_to_1[i2] == i1:
            distance = distances[i1, i2]
            matches.append(DummyMatch(i1, i2, distance))
    
    matches.sort(key=lambda m: m.distance)    

    return matches