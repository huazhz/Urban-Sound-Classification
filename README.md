# Urban Sound classification
Classification of UrbanSound8K dataset using transfer learning

In this project, I'll show how I used a pre-trained model to classify audio samples into 10 different categories.

The main idea is to make a spectrogram from each WAV file and to address the problem as an image classification problem.

Based on Inception, a pre-trained CNN which was trained to classify 1,000 classes of the Imagenet dataset, I trained my model to classify spectrograms into 10 different sound sources.

# Dataset
The UrbanSound8k dataset used for model training, can be downloaded from the following [[link]](https://urbansounddataset.weebly.com/urbansound8k.html)

# Main code

My retrain.py is based on tensorflow script which cab ne downloaded using:
curl -O https://raw.githubusercontent.com/tensorflow/tensorflow/r1.1/tensorflow/examples/image_retraining/retrain.py
