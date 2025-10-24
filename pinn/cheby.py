# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
# ---

# %%
import torch


# %%
def generate_chebyshev_features(x, chebyshev_freq_min, chebyshev_freq_max):
    theta = torch.pi * x[:, 0]
    cos_theta = torch.cos(theta).unsqueeze(1)

    if chebyshev_freq_min > chebyshev_freq_max:
        return torch.empty((x.shape[0], 0), device=x.device)

    # cheby poly 2nd kind Uₖ(cos(πx))
    # U0 = 1
    U_k_minus_2 = torch.ones_like(cos_theta)
    # U₁ = 2 * x
    U_k_minus_1 = 2 * cos_theta
    # all cheby features
    chebyshev_features = []
    # degree k loop
    for k in range(chebyshev_freq_max + 1):
        if k == 0:
            U_k = U_k_minus_2
        elif k == 1:
            U_k = U_k_minus_1
        else:
            U_k = 2 * cos_theta * U_k_minus_1 - U_k_minus_2
            U_k_minus_2 = U_k_minus_1
            U_k_minus_1 = U_k

        if k >= chebyshev_freq_min:
            chebyshev_features.append(U_k)

    return torch.cat(chebyshev_features, dim=1)
