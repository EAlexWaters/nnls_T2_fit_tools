# nnls_T2_fit_tools

This is code in development for performing pixelwise fits of an MRI T2 mapping dataset using nonnegative least squares. It is implemented in Python 3. It takes in a raw dataset with dimensions and echo spacing as described in the script "fit_masked_t2_with_NNLS", and outputs a 4 dimensional array where 3 dimensions correspond to spatial localization and the 4th is the array of t2 components that best fit the measured data. The script "process_nnls_t2_arrays" takes a directory of those arrays, collapses each 4d array into a 1D array containing a distribution of t2 components across all pixels, and creates a dataframe containing the 4D array, the pooled array, and metrics calculated on the distribution of t2 values in the pooled array.
