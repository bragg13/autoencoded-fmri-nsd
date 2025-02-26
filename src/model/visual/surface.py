import os
import numpy as np
import jax.numpy as jnp
from nilearn import datasets, plotting
from matplotlib import pyplot as plt
from nilearn.surface import load_surf_mesh
from data.roi import ROI_TO_CLASS

def get_fmri_vector(data_dir: str = './dataset/nsd_data/', subject: int = 3):
    """
    Load fMRI training data (both left and right hemisphere) for a given subject.

    Args:
        data_dir: Base directory where NSD data are stored.
        subject: Subject number.

    Returns:
        Tuple of NumPy arrays: (lh_fmri, rh_fmri)
    """
    fmri_dir = os.path.join(data_dir, f"subj0{subject}", "training_split", "training_fmri")
    lh_fmri = np.load(os.path.join(fmri_dir, "lh_training_fmri.npy"))
    rh_fmri = np.load(os.path.join(fmri_dir, "rh_training_fmri.npy"))
    print("LH training fMRI data shape:", lh_fmri.shape)
    print("RH training fMRI data shape:", rh_fmri.shape)
    print(f"Sample of LH fMRI data: {lh_fmri[0, :5]}")
    print(f"Sample of RH fMRI data: {rh_fmri[0, :5]}")
    return lh_fmri, rh_fmri

def map_fmri_on_brain_surface(challenge_roi, fsaverage_roi, hemisphere: str, lh_fmri, rh_fmri, img_index: int = 0):
    """
    Map the fMRI data onto the fsaverage space using the ROI mask.

    Returns:
        A 1D array with fMRI values mapped to the fsaverage space.
    """
    print("Challenge ROI shape:", challenge_roi.shape) # this is (19004, )  just the viscor
    print("fsaverage ROI shape:", fsaverage_roi.shape) # this is (163842, ) the whole brain

    # tutto a zero per ora - lunghezza di fsaverage perche e' dove mappero'
    fsaverage_response = np.zeros(len(fsaverage_roi))

    # qui metto i valori di fMRI dove c'e' la ROI
    if hemisphere == 'lh':
        fsaverage_response[np.where(fsaverage_roi)[0]] = lh_fmri[img_index, np.where(challenge_roi)[0]]
    elif hemisphere == 'rh':
        fsaverage_response[np.where(fsaverage_roi)[0]] = rh_fmri[img_index, np.where(challenge_roi)[0]]
    print("fsaverage response shape:", fsaverage_response.shape)
    return fsaverage_response

def get_surface_mesh(fsaverage_response, hemisphere: str):
    """
    Create a filtered mesh from the fsaverage *flat* surface using the fMRI response.

    Args:
        fsaverage_response: 1D array of fMRI values in fsaverage space.
        hemisphere: 'left' or 'right'.

    Returns:
        Tuple of (filtered_coords, masked_response, filtered_faces):
            - filtered_coords: 2D coordinates of vertices in ROI (flat, so only 2 columns).
            - masked_response: fMRI values at the ROI vertices.
            - filtered_faces: Face indices adjusted to the new vertex ordering.
    """
    print(fsaverage_response.shape)

    fsaverage = datasets.fetch_surf_fsaverage("fsaverage")
    coords, faces = load_surf_mesh(fsaverage["flat_left" if hemisphere == 'lh' else 'flat_right'])
    print("Coords shape:", coords.shape)
    print("Faces shape:", faces.shape)

    # we want to mask out the parts of the mesh where there is no response - keep only the vertices with a response
    response_mask = np.where(fsaverage_response)[0] # indices where there is a response
    filtered_coords = coords[response_mask][:, :2]  # use only first 2 dimensions for flat visualization

    # now, my mesh faces are defined with indices referring to the full set of vertices
    # I need to create a new array with the same length as original vertices, but
    # with -1 for the vertices that are not in the response, and a new index for the others
    index_mapping = np.full(np.max(faces) + 1, -1)
    index_mapping[response_mask] = np.arange(response_mask.size)

    # remove the faces that have -1 in a vertex, as they are not valid
    filtered_faces = index_mapping[faces]
    valid_faces_mask = np.all(filtered_faces != -1, axis=1)
    filtered_faces = filtered_faces[valid_faces_mask]

    filtered_response = fsaverage_response[response_mask]
    print("Filtered coords shape:", filtered_coords.shape)
    print("Filtered faces shape:", filtered_faces.shape)
    print("Filtered response shape:", filtered_response.shape)

    return filtered_coords, filtered_response, filtered_faces

def plot_roi(coords, response, title="ROI Scatter plot"):
    plt.figure(figsize=(6, 5))
    plt.scatter(coords[:, 0], coords[:, 1], c=response, cmap="cold_hot", s=5)
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.colorbar(label="BOLD Response")
    plt.show()
