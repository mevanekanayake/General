import torch


# RANDOM MASK
def apply_random_mask(kspace, accelerations, seed):
    torch.manual_seed(seed) if seed else None
    num_rows, num_cols = kspace.shape[0], kspace.shape[1]
    choice = torch.randint(0, len(accelerations), (1,))
    acceleration = accelerations[choice]
    acc_cent_table = [float('inf'), 0.3200, 0.1600, 0.1067, 0.0800, 0.0640, 0.0533, 0.0457, 0.0400, 0.0356, 0.0320, 0.0291, 0.0267]
    center_fraction = acc_cent_table[int(acceleration)]

    num_low_freqs = int(round(num_cols * center_fraction))
    prob = (num_cols / acceleration - num_low_freqs) / (num_cols - num_low_freqs)
    mask = torch.rand(num_cols) < prob
    pad = (num_cols - num_low_freqs + 1) // 2
    mask[pad: pad + num_low_freqs] = True
    mask = mask.repeat(num_rows, 1).long()
    masked_kspace = kspace * mask.unsqueeze(-1)

    return masked_kspace, mask

# TO DO
# EQUISPACED MASK
# def apply_equispaced_mask(kspace, accelerations, seed):
#     return masked_kspace, mask
