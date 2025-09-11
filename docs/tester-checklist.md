# Tester Checklist

1. Enter the BIDS root directory path (provided by the dev team).
2. Enter the tabular data file path (participants.tsv or biomarker CSV).
3. Provide brain template image (.nii.gz) and mask file (.nii.gz).
4. Adjust parameters (smoothing, voxel size, permutations).
5. Run queries such as:
   - "Is tau PET associated with plasma p-tau181, adjusting for age?"
   - "Is hippocampal volume associated with memory score?"
6. Verify:
   - Map loads correctly
   - Text summary is clear
   - Variables match what you asked
