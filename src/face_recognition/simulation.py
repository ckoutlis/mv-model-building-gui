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
    basepath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # mv-model-building-gui absolute path
    with open(os.path.join(basepath, 'results/face-recognition/eval/CelebA_threshold_distance.pickle'), 'rb') as h:
        thresholds = pickle.load(h)
    return thresholds


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


def verification(fp):
    """
    Verifies if the face in fp belongs to any identity of the database.

    Arguments:
        fp: filepath of the image with the face to be verified

    Returns:
        response: dictionary with keys the database's identity names and boolean values
    """

    model = feature_extractor()  # load VGGFace feature extraction model
    thresholds = celeba_distance_thresholds()
    face = get_faces([fp])
    query = get_embedding(model, face)

    basepath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # mv-model-building-gui absolute path
    database = f'{basepath}/data/face-recognition/identities.pickle'
    with open(database, 'rb') as h:
        identities = pickle.load(h)

    response = {}
    for identity in identities:
        support = identities[identity]['embedding']
        k = identities[identity]['k']
        distance = euclidean_distance(support, query)
        threshold = thresholds[k if k <= 20 else 20]
        if distance < threshold:
            response[identity] = True
        else:
            response[identity] = False
    return response


def update_database(identities):
    """
    Updates the user's identities database with new identities contained in identities argument.

    Arguments:
        identities: new identity names and corresponding filepaths, dictionary
        e.g.
        identities = {
            'name1': ['fp1_1.jpg', 'fp1_2.jpg', ...],
            'name2': ['fp2_1.jpg', 'fp2_2.jpg', ...],
         ...
        }
    """
    basepath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # mv-model-building-gui absolute path
    database = f'{basepath}/data/face-recognition/identities.pickle'
    for identity in identities:
        embedding, k = embedding_computation(identities[identity])

        if os.path.exists(database):
            with open(database, 'rb') as h:
                identities_database = pickle.load(h)
        else:
            identities_database = {}

        identities_database[identity] = {'embedding': embedding, 'k': k}

        with open(database, 'wb') as h:
            pickle.dump(identities_database, h, protocol=pickle.HIGHEST_PROTOCOL)
