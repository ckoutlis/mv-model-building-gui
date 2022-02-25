import numpy as np
import os
import pickle
from PIL import Image
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN
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


def extract_faces(detector, images, scale=0.3, output_size=(224, 224)):
    _, H, W, _ = images.shape
    faces = []
    for image in images:
        results = detector.detect_faces(image)
        for r in results:
            xmin, ymin, w, h = r['box']
            x1, y1 = max(0, int(xmin - w * scale)), max(0, int(ymin - h * scale))
            x2, y2 = min(W, int(x1 + w * (1 + 2 * scale))), min(H, int(y1 + h * (1 + 2 * scale)))
            face = image[y1:y2, x1:x2]
            faces.append(np.asarray(Image.fromarray(face.astype(np.uint8)).resize(output_size)))
    return np.array(faces).astype('float32')


def get_faces(fp, detector, mode):
    """
    Extract, stack and preprocess faces existing in input filepaths.

    Arguments:
        fp: list of images filepaths
        detector: mtcnn face detector
        mode: mode, either 'update' or 'verify'

    Returns:
        faces: faces extracted from support images of input list
    """

    images = np.stack(
        [np.asarray(Image.fromarray(plt.imread(x)).resize((224, 224))).astype('float32') for x in fp],
        axis=0
    )
    faces = extract_faces(detector, images)
    faces = preprocess_input(faces, version=2)

    if mode == 'update':
        if faces.shape[0] != images.shape[0]:
            raise Exception(f'{faces.shape[0]} faces in {images.shape[0]} images where found. '
                            f'When updating the database one face per image is allowed.')
    elif mode != 'verify':
        raise Exception('"update" and "verify" are the only allowed values for mode argument.')
    return faces


def get_embedding(model, faces, mode):
    """
    Computes face embeddings using VGGFace.

    Arguments:
        model: the feature extractor
        faces: a 4D numpy array containing the images
        mode: mode, either 'update' or 'verify'

    Returns:
        embedding: support set or query embedding
    """

    k = faces.shape[0]
    embeddings = model.predict(faces)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1).reshape(-1, 1)

    if mode == 'update':
        embedding = np.mean(embeddings, axis=0) if k > 1 else embeddings[0, :]
        return embedding
    elif mode == 'verify':
        return embeddings
    else:
        raise Exception('"update" and "verify" are the only allowed values for mode argument.')


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


def verification(fp):
    """
    Verifies if the faces in fp belong to any identity of the database.

    Arguments:
        fp: filepath of the image with the face to be verified

    Returns:
        response: dictionary with keys the database's identity names and boolean values
    """

    model = feature_extractor()  # load VGGFace feature extraction model
    detector = MTCNN()
    thresholds = celeba_distance_thresholds()
    face = get_faces([fp], detector, 'verify')
    queries = get_embedding(model, face, 'verify')

    basepath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # mv-model-building-gui absolute path
    database = f'{basepath}/data/face-recognition/identities.pickle'
    with open(database, 'rb') as h:
        identities = pickle.load(h)

    response = {}
    for query in queries:
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
    Updates the user's identities database with new identities.

    Arguments:
        identities: new identity names and corresponding filepaths, dictionary
        e.g.
        identities = {
            'name1': ['fp1_1.jpg', 'fp1_2.jpg', ...],
            'name2': ['fp2_1.jpg', 'fp2_2.jpg', ...],
         ...
        }
    """

    model = feature_extractor()  # load VGGFace feature extraction model
    detector = MTCNN()

    basepath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # mv-model-building-gui absolute path
    database = f'{basepath}/data/face-recognition/identities.pickle'
    for identity in identities:
        fp = identities[identity]
        faces = get_faces(fp, detector, 'update')
        embedding = get_embedding(model, faces, 'update')
        k = len(fp)

        if os.path.exists(database):
            with open(database, 'rb') as h:
                identities_database = pickle.load(h)
        else:
            identities_database = {}

        identities_database[identity] = {'embedding': embedding, 'k': k}

        with open(database, 'wb') as h:
            pickle.dump(identities_database, h, protocol=pickle.HIGHEST_PROTOCOL)
