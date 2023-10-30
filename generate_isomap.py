# import packages
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import pandas as pd
import h5py
from os.path import join as pjoin
from tqdm import tqdm
from nilearn import datasets
from nilearn import surface
from sklearn import manifold
from PIL import Image
from scipy.interpolate import LinearNDInterpolator
import os
import itertools

class NSD_Subject():
    isomap_fitted_lh = None
    isomap_fitted_rh = None
    def __init__(self, subj_id, basedir=None, maskdir=None, designdir=None):
        self.subj_id = subj_id
        self.basedir = f'/Volumes/side_project/nsd/fsaverage/{self.subj_id}'
        self.maskdir = f'/Volumes/side_project/nsd/fsaverage/roi_masks'
        self.designdir = f'/Volumes/side_project/nsd/fsaverage/{self.subj_id}/nsd_design'
        self.beta_session = 40
        self.beta_sub_session = 2 # avoid memory error
        self.session_num = 37
        self.run_num_less = 12
        self.run_num_more = 14


    def get_betas(self, hemi="lh"):
        ### It will return the beta values for each voxel in the fsaverage space in LEFT and RIGHT HEMISPHERE
        # get a list of files in the directory
        files = os.listdir(self.basedir)
        # filter the list to include only files starting with "lh.betas_session"
        betas_files = [f for f in files if f.startswith(f'{hemi}.betas_session')]
        beta_voxel_pd = pd.DataFrame()
        # get the voxel index for each subject

        ### add a tqdm progress bar
        for i in tqdm(betas_files[:self.beta_sub_session]):
            fsaverage_betas = nib.load(pjoin(self.basedir, i))
            # write fsaverage_betas into a dataframe
            fsaverage_betas_pd = pd.DataFrame(fsaverage_betas.get_fdata()[:,0,0,:])
            # concatenate the dataframe vertically
            beta_voxel_pd = pd.concat([beta_voxel_pd, fsaverage_betas_pd], axis=1)
        return beta_voxel_pd
    
    def apply_masks(self, hemi="lh"):
        # import mask

        # load npy file
        # lh_visualrois = np.load(pjoin(fmri_dir, 'lh.all-vertices_fsaverage_space.npy'))
        # rh_visualrois = np.load(pjoin(fmri_dir, 'rh.all-vertices_fsaverage_space.npy'))

        # lh_visualrois = np.load("/Users/yilewang/workspaces/diff_fMRI/roi_masks/lh.prf-visualrois_fsaverage_space.npy")
        # rh_visualrois = np.load("/Users/yilewang/workspaces/diff_fMRI/roi_masks/rh.prf-visualrois_fsaverage_space.npy")

        visualrois = np.load(self.maskdir + f"/{hemi}.streams_fsaverage_space.npy")
        return visualrois


    def get_3d_coordinates(self, visualrois, surf_mesh = "infl", hemi="left"):
        ## get the 3d coordinates of the fsaverage mesh surface
        fsaverage = datasets.fetch_surf_fsaverage(mesh="fsaverage")
        hemi_mesh = surface.load_surf_mesh(fsaverage[f"{surf_mesh}_{hemi}"])
        # get the 3d coordinates of the visual cortex
        visual_3d = hemi_mesh.coordinates[np.where(visualrois)[0]]
        return visual_3d

    def isomap_fitting(self, visual_3d, n_components=2, n_neighbors=12, p=2):
        # generate isomap
        isomap = manifold.Isomap(n_neighbors=n_neighbors, n_components=n_components, p=p)
        S_isomap_fitted = isomap.fit_transform(visual_3d)
        return S_isomap_fitted

    def concat_all_designs(self):
        # create an empty array
        _1d_list = []
        ## create strings to read data
        for i in range(self.session_num):
            for z in range(self.run_num_more):
                try:
                    _filename = pjoin(self.designdir, 'design_session' + str(i+1).zfill(2)+'_run'+str(z+1).zfill(2)+'.tsv')
                    ## read tsv data
                    _data = pd.read_csv(_filename, sep='\t', header=None).values
                    # get num larger than 0
                    _data_nonzero = _data[_data > 0]
                    _1d_list.append(_data_nonzero)
                except:
                    continue
        return np.array(list(itertools.chain(*_1d_list)))

    def generate_isomap_imgs(self, S_isomap_fitted, S_betas, visual_rois, img_id, hemi, x_range1=80,x_range2=-80, y_range1=45, y_range2=-45, show_or_save="save"):
        S_betas_values = np.array(S_betas)[np.where(visual_rois)[0]]
        x = S_isomap_fitted[:,0]
        y = S_isomap_fitted[:,1]
        X = np.linspace(x_range1, x_range2, num=int(x_range1-x_range2))
        Y = np.linspace(y_range1, y_range2, num=int(y_range1-y_range2))
        X, Y = np.meshgrid(X, Y)  # 2D grid for interpolation
        interp = LinearNDInterpolator(list(zip(x, y)), S_betas_values)
        Z = interp(X, Y)
        if show_or_save == "save":
            # export the isomap as images
            im = Image.fromarray(Z.astype(np.uint8))
            # if path does not exist, create it
            if not os.path.exists(f"/Volumes/side_project/nsd/fsaverage/{self.subj_id}/isomaps"):
                os.makedirs(f"/Volumes/side_project/nsd/fsaverage/{self.subj_id}/isomaps")
            im.save(f"/Volumes/side_project/nsd/fsaverage/{self.subj_id}/isomaps/{img_id}_{hemi}.png")

    def main(self):
        """
        The general workflow:
            1. get the beta values for each voxel in the fsaverage space in LEFT and RIGHT HEMISPHERE. Based on the design files, 
        we can locate the img indexes. The generated isomap in step 5 will be corresponding to the img indexes.
            2. apply the visual cortex mask to the beta values
            3. get the 3d coordinates of the visual cortex
            4. generate isomap
            5. generate isomap images
        """

        #step 1: get betas
        betas_lh = self.get_betas(hemi="lh")*300
        betas_rh = self.get_betas(hemi="rh")*300
        print("step1 done")
        #step 2: get masks
        visual_rois_lh = self.apply_masks(hemi="lh")
        visual_rois_rh = self.apply_masks(hemi="rh")
        print("step2 done")
        #step 3: get 3d coordinates of the visual cortex
        visual_3d_lh = self.get_3d_coordinates(visual_rois_lh, surf_mesh = "infl", hemi="left")
        visual_3d_rh = self.get_3d_coordinates(visual_rois_rh, surf_mesh = "infl", hemi="right")
        print("step3 done")
        #step 4: generate isomap fitting
        if self.isomap_fitted_lh is None and self.isomap_fitted_rh is None:
            NSD_Subject.isomap_fitted_lh = self.isomap_fitting(visual_3d_lh, n_components=2, n_neighbors=12, p=2)
            NSD_Subject.isomap_fitted_rh = self.isomap_fitting(visual_3d_rh, n_components=2, n_neighbors=12, p=2)
        else:
            pass
        print("step4 done")
        #step 5: generate img indexes that correspond to the isomap
        designs = self.concat_all_designs()
        print("step5 done")
        #step 6: generate isomap images
        for index, i in enumerate(designs[:self.beta_sub_session*750]):
            self.generate_isomap_imgs(self.isomap_fitted_lh, betas_lh.iloc[:,index], visual_rois_lh, i, hemi="lh", x_range1=80, x_range2=-80, y_range1=53, y_range2=-47, show_or_save="save")
            self.generate_isomap_imgs(self.isomap_fitted_rh, betas_rh.iloc[:,index], visual_rois_rh, i, hemi="rh", x_range1=85, x_range2=-75, y_range1=53, y_range2=-47, show_or_save="save")
        print("step6 done")

if __name__ == "__main__":
    subj1 = NSD_Subject("subj01")
    subj1.main()
    subj2 = NSD_Subject("subj02")
    print(subj2.isomap_fitted_lh)