import pandas as pd
import matplotlib.pyplot as plt

drive = '/home/ckoutlis/disk_2_ubuntu/home/ckoutlis/'
imgdir = f'{drive}DataStorage/CelebA/Img/img_align_celeba'
identities_fp = f'{drive}DataStorage/CelebA/Anno/identity_CelebA.txt'
identities_df = pd.read_csv(identities_fp, header=None, sep=' ')
identities = {}
for index, row in identities_df.iterrows():
    image_fp = row[0]
    identity = row[1]
    if identity in identities:
        identities[identity].append(image_fp)
    else:
        identities[identity] = [image_fp]

nimgV = [len(identities[x]) for x in identities]

plt.hist(nimgV)
plt.title('Number of images per identity')
plt.xlabel('# images')
plt.ylabel('# identities')
print(f'Number of identities: {len(nimgV)}')
print(f'Number of images: {identities_df.shape[0]}')
plt.show()
