# %%
import numpy as np
import nibabel as nib
import nanslice as ns
from scipy.ndimage import gaussian_filter
from scipy.stats import t
from scipy.ndimage import zoom
from nanslice.jupyter import three_plane
from scipy.stats import t

# ------------------------------
# Load T1w background and resize to match t-/p-maps
# ------------------------------
bg_img_path = "/scratch/hackathon-2025/team-1/blsa_open_neuroimaging/derivatives/template/tpl-blsa_T1w.nii.gz"
bg_img = nib.load(bg_img_path)

affine = bg_img.affine
target_shape = (24, 29, 24)
data = bg_img.get_fdata()

# Compute zoom factors for each dimension
factors = [t / s for t, s in zip(target_shape, data.shape)]

# Resize using cubic interpolation
resized_data = zoom(data, factors, order=3)  # order=3 = cubic interpolation
print("Resized shape:", resized_data.shape)

# Save the resized image, keep the original affine (optional: adjust if you care about voxel size)
resized_img = nib.Nifti1Image(resized_data, affine=affine)
print('Base IMG size',resized_img.shape)
shape = resized_img.shape

# ------------------------------
# Simulate beta map
# ------------------------------
rng = np.random.default_rng(seed=42)
beta_vals = rng.normal(loc=0.5, scale=0.2, size=shape)
beta_img = nib.Nifti1Image(beta_vals, affine)

# ------------------------------
# Load p-value map
# ------------------------------
p_val_img_path = "/home/baej3/Hackathon/neurovoxel/tests/test_data/pib_dvr_mimap_contrast-age_stat-logp_max_t_map.nii.gz"
logp_img = nib.load(p_val_img_path)
logp_data = logp_img.get_fdata()
print("logp stats:")
print("min:", np.nanmin(logp_data), "max:", np.nanmax(logp_data))

# Convert back to p-values
p_data = np.power(10.0, -logp_data)
print("p-value stats:")
print("min:", np.nanmin(p_data), "max:", np.nanmax(p_data))

sig_mask = p_data <= 0.05
print("Number of voxels with p ≤ 0.05:", np.sum(sig_mask))
# This shows only 1 voxel being significant!

# For AlphaData
one_minus_p = 1 - p_data
one_minus_p = np.clip(one_minus_p, 0, 1)
print("1-p stats:")
print("min:", np.nanmin(one_minus_p), "max:", np.nanmax(one_minus_p))
one_min_p_img = nib.Nifti1Image(one_minus_p, affine=logp_img.affine, header=logp_img.header)


# ------------------------------
# Load t-value map
# ------------------------------
t_val_img_path = "/home/baej3/Hackathon/neurovoxel/tests/test_data/pib_dvr_mimap_contrast-age_stat-t_map.nii.gz"
t_val_img = nib.load(t_val_img_path)

# Manually computing p-value 
t_data = t_val_img.get_fdata()
df = 91 # Based on pib_dvr_mimap observations

# two-tailed p-values
p_data = 2 * (1 - t.cdf(np.abs(t_data), df=df))
print("Uncorrected p-value stats:")
print("min:", np.nanmin(p_data), "max:", np.nanmax(p_data))
sig_mask = p_data <= 0.05
print("Number of voxels with Uncorrected p ≤ 0.05:", np.sum(sig_mask))

one_minus_p = 1 - p_data
one_minus_p = np.clip(one_minus_p, 0, 1)
print("1-Uncorrected_p stats:")
print("min:", np.nanmin(one_minus_p), "max:", np.nanmax(one_minus_p))

one_min_p_img = nib.Nifti1Image(one_minus_p, affine=t_val_img.affine, header=t_val_img.header)


base_layer = ns.Layer(
    resized_img,   # replace with your structural or reference image
    cmap="gray"
)

dual_layer = ns.Layer(
    beta_img,
    cmap="RdYlBu_r",
    clim=(-1, 1),
    scale=1,
    label="Beta coefficients",
    alpha=one_min_p_img,
    alpha_lim=(0.5, 1.0),  # only significant voxels are semi-transparent
    alpha_label="1-p"
)


#Display overlay with contours from alpha
three_plane(
    [base_layer, dual_layer],
    cbar=1,
    contour=0.95,   # contours voxels with alpha >= 0.95 (p <= 0.05)
    title="Beta overlay with alpha=1-p and contours"
)

# # get the voxel values

# t_vals = one_min_p_img.get_fdata().ravel()

# # remove NaNs or 0s (if background is 0)
# t_vals = t_vals[np.isfinite(t_vals)]
# t_vals = t_vals[t_vals != 0]

# plt.hist(t_vals, bins=100, color="steelblue", edgecolor="black")
# plt.xlabel("t-values")
# plt.ylabel("Voxel count")
# plt.title("Distribution of t-values")
# plt.show()


# %%

# import numpy as np
# from nilearn.glm import OLSModel

# # ------------------------------
# # Simulate data
# # ------------------------------
# n_samples = 50       # number of observations
# n_predictors = 3     # number of regressors
# n_voxels = 1000      # number of voxels

# rng = np.random.default_rng(seed=42)

# # Design matrix with intercept + two predictors
# x_mat = rng.normal(size=(n_samples, n_predictors))

# # Simulate beta coefficients
# true_betas = np.array([0.5, -0.3, 0.8]).reshape(-1, 1)  # shape (n_predictors, 1)
# # Simulate voxel-wise data with noise
# y_mat = x_mat @ true_betas + rng.normal(0, 0.1, size=(n_samples, n_voxels))

# # ------------------------------
# # Fit OLSModel
# # ------------------------------
# ols_model = OLSModel(x_mat)
# ols_results = ols_model.fit(y_mat)

# # ------------------------------
# # Extract beta coefficients
# # ------------------------------
# beta_coef = ols_results.theta  # shape (n_predictors, n_voxels)
# print("Beta coefficients shape:", beta_coef.shape)

# # ------------------------------
# # Extract t-statistics predictor by predictor
# # ------------------------------
# t_stats = np.zeros_like(beta_coef)
# for i in range(n_predictors):
#     t_stats[i, :] = ols_results.t(column=i)

# print("T-statistics shape:", t_stats.shape)
# print("T-statistics (first predictor, first 10 voxels):", t_stats[0, :10])







