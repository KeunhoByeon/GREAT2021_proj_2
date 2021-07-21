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
