from src.face_recognition.simulation import update_database
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
names = np.random.choice(list(identities.keys()), 2)

# generate input dictionary and update the database
new_identities = {n: [f'{imgdir}{x}' for x in identities[n]][:5] for n in names}
update_database(new_identities)
