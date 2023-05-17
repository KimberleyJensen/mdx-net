# KUIELab-MDX-Net

- This is a modified KUIELab-MDX-Net used to train the models for the aicrowd music demixing challenge.
- 
## 0. Environment

- I used Ubuntu 18.04
- cuda-able GPU (>= 2080ti) (I used a Nvidia T4 from aws which has 16gb vram)
- wandb for logging

## 1. Installation

```bash
conda env create -f conda_env_gpu.yaml -n mdx-net
conda activate mdx-net
pip install -r requirements.txt

sudo apt-get install soundstretch
```

## 2. Preparing data and files

Each model was trained using three files wav files, the mixture, the target source and the mixture minus target source. For the vocals, bass and drums model, the mixture was called mixture.wav, vocals bass or drums.wav and other.wav. for the other model I used mixture.wav, other.wav and notother.wav

Edit the source files at these two places
https://github.com/KimberleyJensen/mdx-net/blob/f331b185db1fb63bc87b80b22849934577c95b78/configs/datamodule/musdb18_hq.yaml#L14-L16
https://github.com/KimberleyJensen/mdx-net/blob/f331b185db1fb63bc87b80b22849934577c95b78/src/datamodules/datasets/musdb.py#L43

Change the target source in this line if you are not using vocals
https://github.com/KimberleyJensen/mdx-net/blob/29f3828d423f5cfadac4cac8e17cfadb2c930151/src/datamodules/datasets/musdb.py#L86
Inside the data folder there is a folder called train, put your training track folders and validation track folders inside the train folder.

Sign up to https://wandb.ai/site . once you are logged in go to https://wandb.ai/settings and copy your API key.

Open the .env.example file and after wandb_api_key= paste your API key then after data_dir= enter your data folder path (not train folder path)

Windows it will look something like this data_dir=F:\mdx-net\data

Ubuntu it will look something like this data_dir=/home/ubuntu/mdx-net/data

Then save the file but rename it to .env

Open configs/datamodule/musdb18_hq.yaml
 
Under validation_set: is where you put the names of the folders of your validation set (delete the ones that are already there)

## 3. Training
If you are training a bass, drums or other model you need to change BatchNorm2d with GroupNorm at all points in these files
https://github.com/KimberleyJensen/mdx-net/blob/Leaderboard_B/src/models/mdxnet.py
https://github.com/KimberleyJensen/mdx-net/blob/Leaderboard_B/src/models/modules.py

The bass model was traned with num_groups=4 while the drums and other model was trained with num_groups=2

Make sure you are in the mdx-net directory in the command line and you have run conda activate mdx-net if you are using ubuntu

Copy the command below to start training, change multigpu_vocals and ConvTDFNet_vocals to multigpu_bass and ConvTDFNet_bass if you want to train a bass model and the same changes for drums and other.

python run.py experiment=multigpu_vocals model=ConvTDFNet_vocals

