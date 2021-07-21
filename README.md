# OnlineHD extension - adversarial attack defense
Original code: https://gitlab.com/biaslab/onlinehd

## How to run

### Step 1. Train
```
python3 train.py
```

### Step 2. Test
```
python3 test.py
```

### Step 3. Retrain
```
python3 adversarial_attack_defense.py
```

## Added or changed codes in ./onlinehd
Added
```python
# ./onlinehd/onlinehd.py
class OnlineHD(object):
    def probabilities_raw(self, x: torch.Tensor, encoded: bool = False)
    def decode(self, h: torch.Tensor)

# ./onlinehd/encoder.py
class Encoder(object):
    def decode(self, h: torch.Tensor)

# ./onlinehd/spatial.py
def reverse_cos_cdist(cdist: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor, eps: float = 1e-8)
```
Changed
```python
# ./onlinehd/onlinehd.py
class OnlineHD(object):
    def __call__(self, x: torch.Tensor):
        ...
        # we need batches to remove memory usage
        for i in range(0, n, bsize):
            torch.matmul(x[i:i + bsize], self.basis.T, out=temp)
            torch.add(temp, self.base, out=h[i:i + bsize])
            # h[i:i+bsize].cos_().mul_(temp.sin_())
            # h[i:i+bsize].mul_(temp)
    
```

## Progress Note
- I think it seems to works because the more I retrained the model, the more noise I needed to attack the model.
```
1. Getting noise probability(score) for adversarial attacks
1-1. I've tried several times, but
      noise_prob = first_highest_prob - second_highest_prob
      doesn't seem to work well, so I used
      highest_error_indices = np.where(gt == first_highest_indices, second_highest_indices, first_highest_indices)
      noise_prob = 1.0 - raw_prob[range(raw_prob.shape[0]), highest_error_indices]

2. Reversing cos_cdist
2-1. I think when reversing x from distance, there seems to be a relatively large error due to pinverse() in cdist @ x2.T.pinverse() operation.

3. Decoding data
3-1. Removed cos and sin operations.
3-2. Since the mul operation in the encoding stage requires solving a quadratic function during decoding, two cases occur during decoding.
      So I also removed the mul operation during encoding.
      (I am not sure if I can handle it like this)
(3-1 & 3-2: Because of this removal, the validation accuracy in the MNIST dataset decreased from about 93 to about 83.)
3-3. There was an error due to very small decimal data, so it was rounded to 6 decimal places.

4. Retrain model with x + noise data (iteratively)
```

## TODO
The error value is relatively large at this part, so I hope this part to be changed.
```python
# ./onlinehd/spatial.py
def reverse_cos_cdist(cdist: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor, eps: float = 1e-8):
    # TODO: The error value is relatively large at this part, so I hope this part to be changed.
    eps = torch.tensor(eps, device=x1.device)
    norms1 = x1.norm(dim=1).unsqueeze_(1).max(eps)
    norms2 = x2.norm(dim=1).unsqueeze_(0).max(eps)

    cdist = cdist.mul(norms2).mul(norms1)
    reverse_x1 = cdist @ x2.T.pinverse()

    return reverse_x1
```

## About Debug Image
Filename
```
i.png (i is the index of test data)
```

Image title
```
img_x_test: Original image
img_reverse_x_test: Decoded image
img_noise: Noise image
img_noise_x_test: Original + Noise image
```

Info
```
img_x_test
- Ground truth: Ground truth of the original image
- 1st highest pred: First highest probability prediction label of the original image
- 2nd highest pred: Second highest probability prediction label of the original image

img_reverse_x_test
- reverse image pred: First highest probability prediction label of the decoded image

img_noise_x_test
- Noise image pred: First highest probability prediction label of the original+noise image

Original image probabiliy: Predicted probabilities of the original image
Noise image probabiliy: Predicted probabilities of the original+noise image
```

