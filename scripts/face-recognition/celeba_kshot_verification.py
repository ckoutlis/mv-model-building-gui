import pandas as pd
from src.face_recognition import *

np.random.seed(0)  # set random seed

drive = '/home/ckoutlis/disk_2_ubuntu/home/ckoutlis/'  # my second drive where data are stored
imgdir = f'{drive}DataStorage/CelebA/Img/img_align_celeba/'  # images directory
identities_fp = f'{drive}DataStorage/CelebA/Anno/identity_CelebA.txt'  # identities .txt filepath
identities_df = pd.read_csv(identities_fp, header=None, sep=' ')  # identities data frame
n = 1000  # number of episodes for same and for different identities scenarios
threshold_distance_range = np.arange(0.0, 2.005, 0.005)  # candidate distance thresholds for same vs. different identity
threshold_similarity_range = np.arange(-1.0, 1.005, 0.005)  # candidate similarity thresholds for same vs. different identity

print(f'Number of episodes n={n}')

model = feature_extractor()  # load VGGFace feature extraction model
identities = id2fp(identities_df)  # load a dict mapping between identities and image filepaths

# optimal accuracy and threshold for each k (using distance and similarity metrics respectively)
accuracy_distance, accuracy_similarity, threshold_distance, threshold_similarity = [], [], [], []
for k in range(1, 11):  # k: number of support set samples - k-shot verification
    image_tuples_same_id = generate_random_image_tuples(identities, k, n, same_identity=True)
    image_tuples_diff_id = generate_random_image_tuples(identities, k, n, same_identity=False)

    distance_same, distance_different, similarity_same, similarity_different = [], [], [], []
    for i in range(n):
        faces = np.concatenate((get_faces(imgdir, image_tuples_same_id[i]),
                                get_faces(imgdir, image_tuples_diff_id[i])),
                               axis=0)
        support_same, query_same, support_diff, query_diff = get_embeddings(model, faces, k)

        distance_same.append(euclidean_distance(support_same, query_same))
        distance_different.append(euclidean_distance(support_diff, query_diff))
        similarity_same.append(cosine_similarity(support_same, query_same))
        similarity_different.append(cosine_similarity(support_diff, query_diff))

    optimal_accuracy_distance, optimal_threshold_distance = evaluation(
        thresholds=threshold_distance_range,
        metric_same=distance_same,
        metric_different=distance_different,
        mode='distance',
        k=k,
        showtxt=True
    )
    accuracy_distance.append(optimal_accuracy_distance)
    threshold_distance.append(optimal_threshold_distance)

    optimal_accuracy_similarity, optimal_threshold_similarity = evaluation(
        thresholds=threshold_similarity_range,
        metric_same=similarity_same,
        metric_different=similarity_different,
        mode='similarity',
        k=k,
        showtxt=True
    )
    accuracy_similarity.append(optimal_accuracy_similarity)
    threshold_similarity.append(optimal_threshold_similarity)

    plot_histograms(
        optimal_accuracy_distance,
        optimal_threshold_distance,
        optimal_accuracy_similarity,
        optimal_threshold_similarity,
        distance_same,
        distance_different,
        similarity_same,
        similarity_different,
        k,
        n
    )

plot_evaluation(
    accuracy_distance,
    accuracy_similarity,
    threshold_distance,
    threshold_similarity
)
