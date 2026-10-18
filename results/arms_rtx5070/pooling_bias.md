# Pooling-bias diagnostic (thesis Table F.15)

Measured on the RTX 5070 Laptop GPU on 14 August 2026, without retraining. Each value is the ratio of what the converted spiking network passes through a max-pooling layer to what its QCFS source network passes through the same layer (1.000 means no bias). The converted network max-pools binary spike tensors while the source network max-pools analogue activations, and because max is convex, the mean over timesteps of the max is at least the max of the mean.

| Condition | T | conv1 pooling | conv2 pooling |
|---|---|---|---|
| Network held fixed (L = 8 source) | 2 | 1.174 | 1.110 |
| Network held fixed (L = 8 source) | 4 | 1.308 | 1.240 |
| Network held fixed (L = 8 source) | 8 | 1.186 | 1.280 |
| Network held fixed (L = 8 source) | 16 | 1.145 | 1.330 |
| Primary grid (L = T) | 2 | 1.151 | 1.114 |
| Primary grid (L = T) | 8 | 1.186 | 1.280 |

Source: the project run report of 14 August 2026 (Addendum 3, "The pooling bias, measured"). No separate result file for this measurement survives, so this table is the record.
