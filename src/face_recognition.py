import numpy as np
import os
import pickle
from PIL import Image
import matplotlib.pyplot as plt
from keras_vggface.vggface import VGGFace
from tensorflow.keras.models import Model
from keras_vggface.utils import preprocess_input


def feature_extractor():
    """
    Loads VGGFace model and keeps all layers but the classification one.

    Returns:
         model: keras model
    """

    model = VGGFace(model='resnet50')
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    return model


def get_faces(fp):
    """
    Read, stack and preprocess input images.

    Arguments:
        fp: list of images filepaths

    Returns:
        faces: preprocessed support images of input list
    """

    faces = np.stack(
        [np.asarray(Image.fromarray(plt.imread(x)).resize((224, 224))).astype('float32') for x in fp],
        axis=0
    )
    faces = preprocess_input(faces, version=2)
    return faces


def get_embedding(model, faces):
    """
    Computes face embeddings using VGGFace.

    Arguments:
        model: the feature extractor
        faces: a 4D numpy array containing the images

    Returns:
        embedding: support set or query embedding
    """

    k = faces.shape[0]
    embeddings = model.predict(faces)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1).reshape(-1, 1)
    embedding = np.mean(embeddings, axis=0) if k > 1 else embeddings[0, :]
    return embedding


def euclidean_distance(x, y):
    """
    Euclidean distance
    """
    return np.linalg.norm(x - y)


def celeba_distance_thresholds():
    """
    CelebA computed Euclidean distance thresholds.
    """
    basepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # mv-model-building-gui absolute path
    with open(os.path.join(basepath, 'results/face-recognition/CelebA_threshold_distance.pickle'), 'rb') as h:
        thresholds = pickle.load(h)
    return thresholds
