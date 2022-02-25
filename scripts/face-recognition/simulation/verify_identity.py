from src.face_recognition.simulation import verification
from src.face_recognition.experiment import id2fp
import pandas as pd
import numpy as np

np.random.seed(0)

# sample two identities from CelebA and use their first 5 images
drive = '/home/ckoutlis/disk_2_ubuntu/home/ckoutlis/'  # my second drive where data are stored
imgdir = f'{drive}DataStorage/CelebA/Img/img_align_celeba/'  # images directory
identities_fp = f'{drive}DataStorage/CelebA/Anno/identity_CelebA.txt'  # identities .txt filepath
identities_df = pd.read_csv(identities_fp, header=None, sep=' ')  # identities data frame
identities = id2fp(identities_df)  # load a dict mapping between identities and image filepaths
names_in = np.random.choice(list(identities.keys()), 2)  # identity names that exist in database
names_out = np.random.choice(list(identities.keys()), 2)  # identity names that do not exist in database

# generate input dictionary and update the database
fps_in = {n: [f'{imgdir}{x}' for x in identities[n]][6] for n in names_in}  # pick an instance not considered when creating the identity's embedding
fps_out = {n: [f'{imgdir}{x}' for x in identities[n]][0] for n in names_out}  # pick the first instance

for name in fps_in:
    response = verification(fps_in[name])
    print(f'Verification for identity {name}: {response}')

for name in fps_out:
    response = verification(fps_out[name])
    print(f'Verification for identity {name}: {response}')
