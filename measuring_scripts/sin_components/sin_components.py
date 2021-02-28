import sys

import numpy as np
import torch
import tqdm

device = 'cuda'

delta_t = 0.1
times = np.arange(0, 2, delta_t)
A = 5
fi = np.pi / 4  # 45 degree
omega = 5

noise_amplitude = 0

input_size = 3

# y
y_t = A * np.sin(omega * times + fi)

# noise
noise = noise_amplitude * np.random.random(len(times))
y_t = y_t + noise

# the y inputs to the model
y_i = np.array([y_t[i - input_size:i] for i in range(input_size, len(y_t))])

if input_size == 3:
    # analytic result:
    w = np.array([1, -2 * np.cos(omega * delta_t), 1])
    w = w / np.linalg.norm(w)

    # analytical result check
    P_y = np.matmul(y_i, w)
    print(f'For analytical solution: {w} the P abs sum is: {np.sum(np.abs(P_y))}')

# calculate it with gradient method
y_i_torch = torch.as_tensor(y_i, dtype=torch.float32, device=torch.device(device))
w_torch = torch.nn.Parameter(torch.rand([input_size, 1], dtype=torch.float32, device=torch.device(device)),
                             requires_grad=True)
lr = 0.001
optimizer = torch.optim.Adam([w_torch], lr)
epochs = 5000

pbar = tqdm.tqdm(range(epochs), file=sys.stdout)
for epoch in pbar:
    w_torch_norm = w_torch / torch.norm(w_torch.detach())
    P_y = torch.matmul(y_i_torch, w_torch_norm)
    P_y_sum = torch.sum(P_y ** 2)

    optimizer.zero_grad()
    P_y_sum.backward()
    optimizer.step()

    print_str = f'P_y_sum in epoch: {epoch:05d} is: {float(P_y_sum.detach()):.5f}'
    pbar.set_description(print_str, refresh=False)

print(f'For optimization solution: {list(w_torch_norm.detach().cpu())} '
      f'the P abs sum is: {np.sum(np.abs(P_y.detach().cpu().numpy()))}')
