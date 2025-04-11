# 🧠 CIFAR-10 Classification with LeNet-5 and ResBlock Enhanced LeNet 
### (HW for DL course @ NEUQ - by Chiashu Wang 王家澍)

This is the FIRST homework of Deep Learning course @ NEUQ (Northeastern University of Qinhuangdao)

The project implements and compares two convolutional neural network architectures on the CIFAR-10 dataset:

- **LeNet-5** (adapted for RGB images)
- **LeNet-5 with Residual Blocks (ResBlock)**

About the architectures, plz go see the comment section in "./models/resblock.py", "./models/lenet_cifar10.py",
"./models/reslenet.py"

The models are trained and evaluated using PyTorch, with performance metrics plotted for comparison.

## 📈 RESULT
![Model Comparison](./comparison_lenet_resblock.png)

---

## ⚡ Quick Start 

This project is tested and runs smoothly in the [**d2l-zh**](https://github.com/d2l-ai/d2l-zh) environment.

Once you're inside the **d2l-zh** environment, simply run:

```bash
python main.py
```
This will start the training of the two independent neural networks.

Output will be two **.pth** files for the *best epoch weights* of the two & *a line chart* showing how the metrics about 
the two models change after epoches.

## 📚 References

- **LeNet-5**  
  LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998).  
  *Gradient-based learning applied to document recognition.* Proceedings of the IEEE.  
  [Paper link](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)

- **ResNet**  
  He, K., Zhang, X., Ren, S., & Sun, J. (2016).  
  *Deep Residual Learning for Image Recognition.* In CVPR.  
  [Paper link](https://arxiv.org/abs/1512.03385)

- **D2L (Dive into Deep Learning)**  
  Aston Zhang, Zachary C. Lipton, Mu Li, and Alexander J. Smola.  
  *Dive into Deep Learning.*  
  [GitHub Repo (Chinese version)](https://github.com/d2l-ai/d2l-zh)

