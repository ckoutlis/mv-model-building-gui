import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from keras_vggface.vggface import VGGFace
from tensorflow.keras.models import Model
import os
from keras_vggface.utils import preprocess_input


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


def evaluation(thresholds, metric_same, metric_different, mode, k, showtxt=True):
    accuracy = []
    for threshold in thresholds:
        if mode == 'distance':
            same = (np.array(metric_same) < threshold).astype(int)
            diff = (np.array(metric_different) < threshold).astype(int)
        elif mode == 'similarity':
            same = (np.array(metric_same) > threshold).astype(int)
            diff = (np.array(metric_different) > threshold).astype(int)
        tp = np.sum(same == 1)
        tn = np.sum(diff == 0)
        fp = np.sum(diff == 1)
        fn = np.sum(same == 0)
        accuracy.append((tp + tn) / (tp + tn + fp + fn))
    index = np.argmax(accuracy)
    if showtxt:
        if mode == 'distance':
            print(f'Euclidean distance[k={k}]: best accuracy {accuracy[index] * 100:1.1f}% '
                  f'for threshold {thresholds[index]:1.3f}')
        elif mode == 'similarity':
            print(f'Cosine similarity[k={k}]: best accuracy {accuracy[index] * 100:1.1f}% '
                  f'for threshold {thresholds[index]:1.3f}\n')
    return accuracy[index], thresholds[index]


def plot_histograms(optimal_accuracy_distance,
                    optimal_threshold_distance,
                    optimal_accuracy_similarity,
                    optimal_threshold_similarity,
                    distance_same,
                    distance_different,
                    similarity_same,
                    similarity_different,
                    k,
                    n):

    plt.figure(figsize=(11, 5))

    plt.subplot(1, 2, 1)
    plt.title(f'Euclidean distance - acc={optimal_accuracy_distance * 100:1.1f}%, '
              f'thresh={optimal_threshold_distance:1.3f}\n k={k}, n={n}')
    y1, _, _ = plt.hist(distance_same, alpha=0.8, label='same identity')
    y2, _, _ = plt.hist(distance_different, alpha=0.8, label='diff. identity')
    plt.plot([optimal_threshold_distance, optimal_threshold_distance],
             [0, np.max(np.concatenate((y1, y2)))],
             '-r',
             linewidth=1.5,
             label='optimal threshold')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title(f'Cosine similarity - acc={optimal_accuracy_similarity * 100:1.1f}%, '
              f'thresh={optimal_threshold_similarity:1.3f}\n k={k}, n={n}')
    y1, _, _ = plt.hist(similarity_same, alpha=0.8, label='same identity')
    y2, _, _ = plt.hist(similarity_different, alpha=0.8, label='diff. identity')
    plt.plot([optimal_threshold_similarity, optimal_threshold_similarity],
             [0, np.max(np.concatenate((y1, y2)))],
             '-r',
             linewidth=1.5,
             label='optimal threshold')
    plt.legend()

    basepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    plt.savefig(os.path.join(basepath, f'results/face-recognition/figs/fig_k{k}.jpg'))


def plot_evaluation(accuracy_distance, accuracy_similarity, threshold_distance, threshold_similarity):
    plt.figure()
    plt.plot(np.arange(1, 11), accuracy_distance, '.-', label='accuracy (euc.dist.)')
    plt.plot(np.arange(1, 11), accuracy_similarity, '.-', label='accuracy (cos.sim.)')
    plt.plot(np.arange(1, 11), threshold_distance, '.-', label='threshold (euc.dist.)')
    plt.plot(np.arange(1, 11), threshold_similarity, '.-', label='threshold (cos.sim.)')
    plt.xlabel('k (# support samples)')
    plt.grid()
    plt.legend()
    basepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    plt.savefig(os.path.join(basepath, f'results/face-recognition/figs/fig_evaluation.jpg'))
