import numpy as np
from keras_vggface.vggface import VGGFace
from tensorflow.keras.models import Model


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