# Hardware
MacBook Air  M1 💻

# Overview
This project uses Mediapipe and PyTorch to build a hand gesture recognition model. It includes code for both model training and real-time gesture prediction.

<img width="474" alt="스크린샷 2024-11-04 오후 5 19 25" src="https://github.com/user-attachments/assets/a4424ad7-b59e-46dd-8af3-eb2467164f72">



# model architecture

```python
class CNNHandGestureModel(nn.Module):
    def __init__(self):
        super(CNNHandGestureModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        self.fc1 = nn.Linear(64 * 5 * 2, 128)
        self.fc2 = nn.Linear(128, 6)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 64 * 5 * 2)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```


install Anaconda: (recommended)

Download Anaconda from the link above.
Follow the installation instructions.
Once installed, open the terminal (or Anaconda Prompt on Windows) to create a new environment:
```
conda create -n hand-gesture-env python=3.9
conda activate hand-gesture-env
```
Note: If you have not installed Anaconda, you can ignore any steps that include conda.

## Setup
Clone the Repository
First, open a terminal. To navigate to your desired directory, type:

```
git clone https://github.com/100-heon/finger_detection_draw_mediapipe.git
```

## move to directory
```
cd finger_detection_draw_mediapipe
```


## Install Required Packages
```
pip install -r requirements.txt
```

## run
```
python detect.py
```

Done 👍🏻

enjoy
