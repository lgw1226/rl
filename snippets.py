import torch


def tensor_shape_test():
    t_a = torch.tensor(1)
    t_b = torch.tensor([1])

    print(f"t_a: {t_a}\nt_b: {t_b}")
    print(f"t_a.shape: {t_a.shape}\nt_b.shape: {t_b.shape}")


if __name__ == "__main__":
    tensor_shape_test()