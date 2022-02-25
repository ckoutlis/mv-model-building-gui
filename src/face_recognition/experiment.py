import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from keras_vggface.vggface import VGGFace
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


def id2fp(df):
    """
    Generates a dictionary with identity names as keys mapping to lists of corresponding image filepaths

    Arguments:
        df: a pandas DataFrame containing the identities

    Returns:
         identities: dictionary of identities and image filepaths
    """

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
    """
    Randomly samples n (k+1)-tuples that contain k+1 instances of the same identity if same_identity=True.
    If same_identity=False the tuples contain k instances of one identity and one identity of a different identity.

    Arguments:
        identities: dictionary with identity names as keys mapping to lists of corresponding image filepaths
        k: number of shots, int
        n: number of examples, int
        same_identity: whether to generate tuple for one identity or two, boolean

    Returns:
        tuples: list of tuples
    """

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
    """
    Samples a tuple of images either belonging to the same or different identitites.

    Arguments:
        identities: dictionary with identity names as keys mapping to lists of corresponding image filepaths
        identity: identity name
        threshold: minimum number of instances for identity
        same_identity: whether to generate tuple for one identity or two, boolean

    Returns:
        (k+1)-tuple
    """

    if same_identity:
        return np.random.choice(identities[identity[0]], threshold, replace=False)
    else:
        return np.concatenate((np.random.choice(identities[identity[0]], threshold, replace=False),
                               np.random.choice(identities[identity[1]], 1, replace=False)))


def get_faces(directory, tuple):
    """
    Read, stack and preprocess images of an example tuple.

    Arguments:
        directory: the images directory
        tuple: tuple of images filepaths

    Returns:
        faces: preprocessed support and query images of input tuple
    """

    support = [os.path.join(directory, x) for x in tuple[:-1]]
    query = os.path.join(directory, tuple[-1])
    faces = np.stack(
        [np.asarray(Image.fromarray(plt.imread(x)).resize((224, 224))).astype('float32') for x in support] +
        [np.asarray(Image.fromarray(plt.imread(query)).resize((224, 224))).astype('float32')],
        axis=0)
    faces = preprocess_input(faces, version=2)
    return faces


def get_embeddings(model, faces, k):
    """
    Computes face embeddings using VGGFace.

    Arguments:
        model: the feature extractor
        faces: a 4D numpy array containing the images
        k: number of shots

    Returns:
        support_same: support set embedding for same identity example
        query_same: query embedding for same identity example
        support_diff: support set embedding for different identities example
        query_diff: query embedding for different identities example
    """

    embeddings = model.predict(faces)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1).reshape(-1, 1)
    support_same = np.mean(embeddings[:k, :], axis=0) if k > 1 else embeddings[0, :]
    query_same = embeddings[k, :]
    support_diff = np.mean(embeddings[k + 1:-1, :], axis=0) if k > 1 else embeddings[2, :]
    query_diff = embeddings[-1, :]
    return support_same, query_same, support_diff, query_diff


def euclidean_distance(x, y):
    """
    Euclidean distance
    """

    return np.linalg.norm(x - y)


def cosine_similarity(x, y):
    """
    Cosine similarity
    """

    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def evaluation(thresholds, metric_same, metric_different, mode, k, showtxt=True):
    """
    Computes the optimal metric threshold and accuracy for classification between different identities.

    Arguments:
        thresholds: array of candidate thresholds
        metric_same: metric values for the same identity
        metric_different: metric values for different identities
        mode: the metric, either distance or similarity
        k: number of shots
        showtxt: boolean

    Returns:
        optimal_accuracy: best accuracy score across thresholds
        optimal_threshold: threshold that results in best accuracy
    """

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
    optimal_accuracy = accuracy[index]
    optimal_threshold = thresholds[index]
    return optimal_accuracy, optimal_threshold


def plot_statistics(fp, identities, identities_df, savfig):
    """
    Plots statistics of CelebA dataset.

    Arguments:
        fp: figure filepath
        identities: dict mapping between identities and image filepaths
        identities_df: identities data frame
        savfig: whether to save the figure
    """

    nimgV = [len(identities[x]) for x in identities]
    plt.hist(nimgV)
    plt.title('Number of images per identity')
    plt.xlabel('# images')
    plt.ylabel('# identities')
    print(f'Number of identities: {len(nimgV)}')
    print(f'Number of images: {identities_df.shape[0]}')
    if savfig:
        plt.savefig(fp)


def plot_histograms(fp,
                    optimal_accuracy_distance,
                    optimal_threshold_distance,
                    optimal_accuracy_similarity,
                    optimal_threshold_similarity,
                    distance_same,
                    distance_different,
                    similarity_same,
                    similarity_different,
                    k,
                    n,
                    savfig):
    """
    Plots two histograms for Euclidean distance and two histograms for cosine similarity.
    The two histograms for each case illustrate the distribution of the corresponding metric in cases
    with the same identity and cases with different identities.

    Arguments:
        fp: figure filepath
        optimal_accuracy_distance: best accuracy based on Euclidean distance
        optimal_threshold_distance: optimal threshold based on Euclidean distance
        optimal_accuracy_similarity: best accuracy based on cosine similarity
        optimal_threshold_similarity: optimal threshold based on cosine similarity
        distance_same: distances between instances of the same identity
        distance_different: distances between instances of different identities
        similarity_same: similarities between instances of the same identity
        similarity_different: similarities between instances of different identities
        k: maximum number of shots
        n: number of episodes for same and for different identities scenarios
        savfig: whether to save the figure
    """

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

    if savfig:
        plt.savefig(fp)


def plot_evaluation(fp, accuracy_distance, accuracy_similarity, threshold_distance, threshold_similarity, savfig):
    """
    Plots accuracy and thresholds for Euclidean distance and cosine similarity.

    Arguments:
        fp: figure filepath
        accuracy_distance: accuracy for each k based on Euclidean distance
        accuracy_similarity: accuracy for each k based on cosine similarity
        threshold_distance: threshold for each k based on Euclidean distance
        threshold_similarity: threshold for each k based on cosine similarity
        savfig: whether to save the figure
    """

    k = len(accuracy_distance)
    plt.figure()
    plt.plot(np.arange(1, k + 1), accuracy_distance, '.-', label='accuracy (euc.dist.)')
    plt.plot(np.arange(1, k + 1), accuracy_similarity, '.-', label='accuracy (cos.sim.)')
    plt.plot(np.arange(1, k + 1), threshold_distance, '.-', label='threshold (euc.dist.)')
    plt.plot(np.arange(1, k + 1), threshold_similarity, '.-', label='threshold (cos.sim.)')
    plt.xlabel('k (# support samples)')
    plt.grid()
    plt.legend()

    if savfig:
        plt.savefig(fp)
