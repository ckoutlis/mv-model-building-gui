import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from keras_vggface.vggface import VGGFace
from tensorflow.keras.models import Model
import os
from keras_vggface.utils import preprocess_input


def generate_random_image_tuples(identities, k, n, same_identity):
    num_random_identities = 1 if same_identity else 2
    threshold = k + 1 if same_identity else k
    tuples = []
    while True:
        identity = np.random.choice(list(identities.keys()), num_random_identities, replace=False)
        identity_0_size = len(identities[identity[0]])
        if identity_0_size >= threshold:
            tuples.append(
                generate_image_tuple(
                    identities,
                    identity,
                    threshold,
                    same_identity
                )
            )
        if len(tuples) == n:
            break
    return tuples


def generate_image_tuple(identities, identity, threshold, same_identity):
    if same_identity:
        return np.random.choice(identities[identity[0]], threshold, replace=False)
    else:
        return np.concatenate((np.random.choice(identities[identity[0]], threshold, replace=False),
                               np.random.choice(identities[identity[1]], 1, replace=False)))


def feature_extractor():
    model = VGGFace(model='resnet50')
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    return model


def id2fp(df):
    identities = {}
    for index, row in df.iterrows():
        image_fp = row[0]
        identity = row[1]
        if identity in identities:
            identities[identity].append(image_fp)
        else:
            identities[identity] = [image_fp]
    return identities


def get_faces(directory, tuples):
    support = [os.path.join(directory, x) for x in tuples[:-1]]
    query = os.path.join(directory, tuples[-1])
    faces = np.stack(
        [np.asarray(Image.fromarray(plt.imread(x)).resize((224, 224))).astype('float32') for x in support] +
        [np.asarray(Image.fromarray(plt.imread(query)).resize((224, 224))).astype('float32')],
        axis=0)
    faces = preprocess_input(faces, version=2)
    return faces


def get_embeddings(model, faces, k):
    embeddings = model.predict(faces)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1).reshape(-1, 1)
    support_same = np.mean(embeddings[:k, :], axis=0) if k > 1 else embeddings[0, :]
    query_same = embeddings[k, :]
    support_diff = np.mean(embeddings[k + 1:-1, :], axis=0) if k > 1 else embeddings[2, :]
    query_diff = embeddings[-1, :]
    return support_same, query_same, support_diff, query_diff


def euclidean_distance(x, y):
    return np.linalg.norm(x - y)


def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
