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
def chebyshev_transformed_features(x, chebyshev_freq_min, chebyshev_freq_max):
    chebyshev_features = []
    theta = torch.pi * x[:, 0]
    cos_theta = torch.cos(theta)

    chebyshev_features.append(torch.ones_like(cos_theta))
    chebyshev_features.append(2 * cos_theta)

    for degree in range(2, chebyshev_freq_max):
        u = 2 * cos_theta * chebyshev_features[degree - 1] - chebyshev_features[degree - 2]
        chebyshev_features.append(u)
    x = torch.stack(chebyshev_features).T

    return x[:, chebyshev_freq_min-1:]

def generate_chebyshev_features(x, chebyshev_freq_min, chebyshev_freq_max):
    theta = torch.pi * x[:, 0]
    cos_theta = torch.cos(theta).unsqueeze(1)
    
    if chebyshev_freq_min > chebyshev_freq_max:
        return torch.empty((x.shape[0], 0), device=x.device)

    U_k_minus_2 = torch.ones_like(cos_theta)
    U_k_minus_1 = 2 * cos_theta
    
    for k_current_degree in range(2, chebyshev_freq_min + 1): 
        U_k = 2 * cos_theta * U_k_minus_1 - U_k_minus_2
        U_k_minus_2 = U_k_minus_1
        U_k_minus_1 = U_k

    if chebyshev_freq_min == 0:
        u_k_min = U_k_minus_2
        u_k_min_plus_1 = U_k_minus_1
    elif chebyshev_freq_min == 1:
        u_k_min = U_k_minus_1
        u_k_min_plus_1 = 2 * cos_theta * U_k_minus_1 - U_k_minus_2
    else:
        u_k_min = U_k_minus_1
        u_k_min_plus_1 = 2 * cos_theta * U_k_minus_1 - U_k_minus_2

    chebyshev_features = []
    
    if chebyshev_freq_min <= chebyshev_freq_max:
        chebyshev_features.append(u_k_min)
        
        if chebyshev_freq_min + 1 <= chebyshev_freq_max:
            chebyshev_features.append(u_k_min_plus_1)
            
            u_k_minus_2 = u_k_min
            u_k_minus_1 = u_k_min_plus_1

            for k_current_degree in range(chebyshev_freq_min + 2, chebyshev_freq_max + 1):
                current_chebyshev_u = 2 * cos_theta * u_k_minus_1 - u_k_minus_2
                
                u_k_minus_2 = u_k_minus_1
                u_k_minus_1 = current_chebyshev_u
                
                chebyshev_features.append(current_chebyshev_u)

    return torch.cat(chebyshev_features, dim=1)
