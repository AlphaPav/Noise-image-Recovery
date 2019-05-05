
# Noise-image-Recovery
CNN encoder and decoder for noise image recovery in Pytorch
- model 
  ![modelimage](./modelimage.png)

- `generate noise`

  ```python
  def gen_noise_img(self,image,noise_percent):
      img = copy.deepcopy(image)
      height, width = img.shape[:2]
      real_num= int(width*(1-noise_percent)) # true pixel number
      for i in range(0,height): # every row
          for k in range(0, 3):  # every channel 
              mask = np.zeros(width)
              mask[:real_num] = 1
              random.shuffle(mask)
              for j in range(0,width): # col
                      img[i][j][k] *=mask[j]
  ```
## Results


### CIFAR10：

Noise Image - Recover Image  - Original Image （epoch40）：

- 80% noise

![epoch10_step2500_noise](./output/gray80/epoch10_step2500_noise.png)![epoch10_step2500_output](./output/gray80/epoch10_step2500_output.png)![epoch10_step2500_true](./output/gray80/epoch10_step2500_true.png)

![epoch10_step2400_noise](./output/gray80/epoch10_step2400_noise.png)![epoch10_step2400_output](./output/gray80/epoch10_step2400_output.png)![epoch10_step2400_true](./output/gray80/epoch10_step2400_true.png)

![epoch10_step2300_noise](./output/gray80/epoch10_step2300_noise.png)![epoch10_step2300_output](./output/gray80/epoch10_step2300_output.png)![epoch10_step2300_true](./output/gray80/epoch10_step2300_true.png)

- 40% noise

![epoch40_step2400_noise](./output/color40/epoch40_step2400_noise.png)![epoch40_step2400_output](./output/color40/epoch40_step2400_output.png)![epoch40_step2400_true](./output/color40/epoch40_step2400_true.png)

![epoch40_step2500_noise](./output/color40/epoch40_step2500_noise.png)![epoch40_step2500_output](./output/color40/epoch40_step2500_output.png)![epoch40_step2500_true](./output/color40/epoch40_step2500_true.png)

![epoch40_step2300_noise](./output/color40/epoch40_step2300_noise.png)![epoch40_step2300_output](./output/color40/epoch40_step2300_output.png)![epoch40_step2300_true](./output/color40/epoch40_step2300_true.png)

- 60% noise

![epoch40_step2500_noise](./output/color60/epoch40_step2500_noise.png)![epoch40_step2500_output](./output/color60/epoch40_step2500_output.png)![epoch40_step2500_true](./output/color60/epoch40_step2500_true.png)

![epoch40_step2400_noise](./output/color60/epoch40_step2400_noise.png)![epoch40_step2400_output](./output/color60/epoch40_step2400_output.png)![epoch40_step2400_true](./output/color60/epoch40_step2400_true.png)

![epoch40_step2300_noise](./output/color60/epoch40_step2300_noise.png)![epoch40_step2300_output](./output/color60/epoch40_step2300_output.png)![epoch40_step2300_true](./output/color60/epoch40_step2300_true.png)

### A（80% noise）：

<figure class="half">
  <img src="A.png" >
  <img src="./output/gray80/test_39_output.png" >
</figure>

- Change process：

<figure class="half">
<img src="./output/gray80/test_0_output.png" width="200"><img src="./output/gray80/test_5_output.png" width="200">
</figure>

Epoch = 1 -> Epoch = 6

<figure class="half">
<img src="./output/gray80/test_10_output.png" width="200"><img src="./output/gray80/test_39_output.png" width="200">
</figure>

Epoch = 11 -> Epoch = 40


### B（40% noise）：

<figure class="half">
<img src="B.png" width="200"><img src="./output/color40/test_output.png" width="200">
</figure>

### C（60% noise）：

<figure class="half">
<img src="C.png" width="200"><img src="./output/color60/test_output.png" width="200">
</figure>

