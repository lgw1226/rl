import torch


def tensor_shape_test():
    t_a = torch.tensor(1)
    t_b = torch.tensor([1])
    t_c = torch.tensor([1, 2, 3, 4])

    print(f"t_a: {t_a}\nt_b: {t_b}\nt_c: {t_c}")
    print(f"t_a.shape: {t_a.shape}\nt_b.shape: {t_b.shape}\nt_c.shape: {t_c.shape}")

def tensor_indexing_test():
    t = torch.tensor([[1, 2, 3],
                      [4, 5, 6]])
    
    val, idx = torch.max(t, dim=1)

    t_idx = t.gather(1, idx.view(-1, 1)).view(-1)
    print(t_idx)

    print(torch.randint(0, 3, (1,)))


if __name__ == "__main__":
    tensor_indexing_test()