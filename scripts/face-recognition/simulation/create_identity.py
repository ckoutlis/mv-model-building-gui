from src.face_recognition.simulation import *


def embedding_computation(fp):
    """
    Comoputes the support embedding of a new identity based on images in fp

    Arguments:
        fp: list of filepaths containing images of one new identity

    Returns:
        embedding: support embedding of the new identity
        k: number of support images
    """
    model = feature_extractor()  # load VGGFace feature extraction model
    faces = get_faces(fp)
    embedding = get_embedding(model, faces)
    k = len(fp)
    return embedding, k


def face_verification(fp, support, k):
    model = feature_extractor()  # load VGGFace feature extraction model
    thresholds = celeba_distance_thresholds()
    face = get_faces([fp])
    query = get_embedding(model, face)
    distance = euclidean_distance(support, query)
    threshold = thresholds[k if k <= 20 else 20]
    if distance < threshold:
        return True
    else:
        return False
