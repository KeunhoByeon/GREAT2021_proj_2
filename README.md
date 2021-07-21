# OnlineHD extension - adversarial attack defense
Original code: https://gitlab.com/biaslab/onlinehd

## How to run

### Train
```
python3 train.py
```

### Test
```
python3 test.py
```

### Retrain
```
python3 adversarial_attack_defense.py
```

## Added or changed codes in ./onlinehd
Added
```python
# ./onlinehd/onlinehd.py
def probabilities_raw(self, x: torch.Tensor, encoded: bool = False)
def decode(self, h: torch.Tensor)

# ./onlinehd/onlinehd.py
def decode(self, h: torch.Tensor)

# ./onlinehd/spatial.py
def inverse_cos_cdist(cdist: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor, eps: float = 1e-8)
```
Changed
```python
# ./onlinehd/onlinehd.py
def __call__(self, x: torch.Tensor):
    ...
    # we need batches to remove memory usage
    for i in range(0, n, bsize):
        torch.matmul(x[i:i + bsize], self.basis.T, out=temp)
        torch.add(temp, self.base, out=h[i:i + bsize])
        # h[i:i+bsize].cos_().mul_(temp.sin_())
        # h[i:i+bsize].mul_(temp)
    
```

## TODO
from ./onlinehd/spatial.py
```python
def inverse_cos_cdist(cdist: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor, eps: float = 1e-8):
    # TODO: The error value is large in this part, so it has to be replaced
    eps = torch.tensor(eps, device=x1.device)
    norms1 = x1.norm(dim=1).unsqueeze_(1).max(eps)
    norms2 = x2.norm(dim=1).unsqueeze_(0).max(eps)

    cdist = cdist.mul(norms2).mul(norms1)
    new_x1 = cdist @ x2.T.pinverse()

    return new_x1
```

## About Debug Image
Filename
```
i.png (i is the index of test data)
```

Image title
```
img_x_test: Original image
img_inverse_x_test: Decoded image
img_noise: Noise image
img_noise_x_test: Original + Noise image
```

Info
```
img_x_test
- Ground truth: Ground truth of the original image
- 1st highest pred: First highest probability prediction label of the original image
- 2nd highest pred: Second highest probability prediction label of the original image

img_inverse_x_test
- Inverse image pred: First highest probability prediction label of the decoded image

img_noise_x_test
- Noise image pred: First highest probability prediction label of the original+noise image

Original image probabiliy: Predicted probabilities of the original image
Noise image probabiliy: Predicted probabilities of the original+noise image
```

