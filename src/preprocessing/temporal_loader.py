def create_temporal_pairs(frames):
    """
    Create consecutive frame pairs for temporal feature extraction.

    Input:
        frames -> tensor [T, 3, 224, 224]

    Output:
        list of tuples:
        [(frame1, frame2), (frame2, frame3), ...]
    """

    temporal_pairs = []

    for i in range(len(frames) - 1):
        temporal_pairs.append(
            (frames[i], frames[i + 1])
        )

    return temporal_pairs