# ===============================================================
#                   SuStaIn Modeling Full Pipeline
#            Preprocessing → Normalization → SuStaIn Model
#                       → MCMC Trace → Subtype Output
# ===============================================================

# 1) Import dependencies -----------------------------------------
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sustain import SuStaIn
import warnings
warnings.filterwarnings("ignore")


# 2) Load dataset -------------------------------------------------
# Input file contains GMV features from 15 ROIs
# The first columns correspond to demographic/subject information
data = pd.read_excel('Biomarkers_GMV_Data_15ROI.xlsx')

# Extract biomarker columns (starting from column index 6)
biomarkers = data.columns[6:]


# 3) Basic data inspection ----------------------------------------
print(data.head())                   # preview first samples
print(data.Diagnosis.value_counts()) # diagnosis group counts


# 4) Extract phenotype labels & feature matrix ---------------------
diagnosis = data['Diagnosis']        # group labels (CN/EMCI/LMCI/AD)
X = data[biomarkers].values          # convert features to NumPy array


# 5) Normalize features using Median & IQR -------------------------
# Robust scaling is recommended for biological features with skew/outliers
median = np.median(X, axis=0)
iqr = np.percentile(X, 75, axis=0) - np.percentile(X, 25, axis=0)
X_norm = (X - median) / iqr          # standardization for SuStaIn input


# 6) Initialize SuStaIn model -------------------------------------
# n_subtypes = number of disease subgroups (should be validated by CV)
# n_stages   = disease progression stages
# n_samples  = MCMC sampling steps (higher → more stable but slower)
model = SuStaIn(n_subtypes=3, n_stages=40, n_samples=50000)


# 7) Train the model (MCMC inference) ------------------------------
model.fit(X_norm)


# 8) Output model results ------------------------------------------
subtype_prob = model.get_subject_probabilities()   # P(subject → subtype)
ll_trace = model.get_likelihood_trace()             # MCMC convergence trace

print("Subtype Probability Matrix Shape:", subtype_prob.shape)
print("Length of Likelihood Trace:", len(ll_trace))


# 9) Plot MCMC likelihood trace -----------------------------------
plt.plot(ll_trace)
plt.title("MCMC Log-Likelihood Trace")
plt.xlabel("Sampling Iterations")
plt.ylabel("Log-Likelihood")
plt.show()


# 10) Determine most probable subtype for each subject -------------
predicted_subtype = np.argmax(subtype_prob, axis=1)

unique, counts = np.unique(predicted_subtype, return_counts=True)
print("Subtype Distribution:", dict(zip(unique, counts)))
