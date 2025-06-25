import torch


def chebyshev_transformed_features(x, chebyshev_freq_min, chebyshev_freq_max):
    chebyshev_features = []
    theta = torch.pi * x[:, 0]
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    left_end = torch.abs(theta) < 1e-8
    right_end = torch.abs(theta - torch.pi) < 1e-8

    u_k_minus_2 = torch.sin((chebyshev_freq_min) * theta) / sin_theta
    u_k_minus_2[left_end] = float(chebyshev_freq_min)
    u_k_minus_2[right_end] = float((chebyshev_freq_min) * (-1)**(chebyshev_freq_min - 1))

    u_k_minus_1 = torch.sin((chebyshev_freq_min + 1) * theta) / sin_theta
    u_k_minus_1[left_end] = float(chebyshev_freq_min + 1)
    u_k_minus_1[right_end] = float((chebyshev_freq_min + 1) * (-1)**(chebyshev_freq_min))

    for k_current_degree in range(chebyshev_freq_min, chebyshev_freq_max + 1):
        if k_current_degree == chebyshev_freq_min:
            current_chebyshev_u = u_k_minus_2
        elif k_current_degree == chebyshev_freq_min + 1:
            current_chebyshev_u = u_k_minus_1
        else:
            current_chebyshev_u = 2 * cos_theta * u_k_minus_1 - u_k_minus_2
            u_k_minus_2 = u_k_minus_1
            u_k_minus_1 = current_chebyshev_u
        chebyshev_features.append(current_chebyshev_u.unsqueeze(1))

    return torch.cat(chebyshev_features, dim=1)


def chebyshev_transformed_features2(x, chebyshev_freq_min, chebyshev_freq_max):
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
