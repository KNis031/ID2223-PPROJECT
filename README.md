# ID2223-PROJECT
*This project is a part of the course ID2223 Scalable machine learning, given at KTH, fall 2023*

## Report
### Introduction
A common problem discussed in *Music information retrival* is that of auto tagging sounds. For this problem, one aims to predict tag information of a sound based on acoustic aspects. Examples include sound event recognition, music genre classification and musical instrument classification [1]. For this project, there was no limitation to a specific audio domain, but the problem of finding relevant tags for sounds was considered in the general case. Posed as a machine learning problem it could be considered a multiclass, multi-label classification task, where a number of string-labels are suggested based on an analysis of audio data. The dataset used is the [dataset used in COALA](https://zenodo.org/records/3887261), a dataset of 189896 mel-spectrograms computed on <10 seconds audio clips from [freesound.org](https://freesound.org/) [1]. For the classification task, the model I employed follows a simple convolutional neural network (CNN) architecture with fully connected layers on top. After training the model was tested on a test set. Beyond training and testing, I developed two apps which can be used for generating tags for uploads to freesound.org. Also, as a part of the apps, I developed a simple framework which can be used for further development.

Tags: Music information retrival, Hopsworks, modal, Hugging Face spaces, freesound.org API

### Background
In the general audio tagging classification problem, an audio clip can be given one or multiple tags by a (machine learning) classifier. Considering the multi-label case, we have a feature-vector *x* and a corresponding tag-vector *y*. We then aim to find a function *f* that maps the features to a prediction vector y<sub>pred</sub>:

f (x) = y<sub>pred</sub>

such that y<sub>pred</sub> is similar to y given some metric.

There are at least three common ways to do audio tagging. First is the “old-school” way, where pre-computed features such MFCCs are used together with non-deep models such as k-nearest neighbors [2]. This approach has fallen out of fashion, and is generally outperformed by deep approaches. Of these there exists two, either computing mel-spectrograms and using these as input, or using the waveform directly as input. Both of these approaches generally use CNNs, but the CNN architectures varies to some extent between and within the two approaches. Considering the mel-spectrogram approach, common architectures employ 3 to 6 convolutional layers followed by 1 to 3 fully connected layers. Still, different takes on kernel size. pooling stratergy etc. exists. Other approaches include replacing the fully connected layers with recurrent neural networks (RNNs). These differences in architecture are in part motivated by the shape of the input spectrogram. One may for example input the mel-spectrogram of the full duration of the sound clip or, only a segment of it.

### Method
#### Dataset and features
As mentioned, the model is trained with the [dataset used in COALA](https://zenodo.org/records/3887261), a dataset of 189896 mel-spectrograms computed over <10 second audio clips from [freesound.org](https://freesound.org/). The dataset is uploaded in a training and validation split. For the project, I ended up splitting it further into training, validation and testing sets. Of the 170793 entries in the original 'training' set, 70% were used for training and the remaining 30% were used for validation. The set referenced as 'validation' in the dataset description consisting of 19103 entries, I used for testing. The mel-spectrograms are computed from all clips with a duration of less than 10 seconds, with the shorter clips being padded with noise to 10 seconds. Sounds are resampled to 22 kHz. The spectrograms are computed with 96 mel-bins, a window size of 1024 samples, a hopsize of 512 samples, and a Hamming windowing function. Finally, a patch of size 96x96 with the highest energy is collected from each mel-spectrogram, the patch is batch-normalized and is used as features input to the model. This last step is motivated by that events in the highest energy frame are assumed to contain most discriminitive power. For targets, the dataset uses multi-hot encoded tags from a vocabulary of 1000 user entered tags. This vocabulary was created by removing stopwords, making plural forms of nouns to singular, removing less informative words, and taking the remaining 1000 tags of the sounds of freesound.org [1].

![melspec](https://github.com/KNis031/ID2223-PPROJECT/blob/main/melspec.png))

*a mel-spectrogram*

#### Architecture
With model architecture, I took inspiration from a number of papers, including one of the models explored by Favory et.al [1] which the dataset were created for. After some trial and error, I landed in using a Convolutional Neural Network architecture of four convolutional layers and two linear layers.

#### Training
Initially, it was planned to train the model for 101 epochs. However, after inspecting the loss curves after half the training time (51 epochs), it seemed as the model had reached the best validation performance after just 11 epochs. Therefore no more training was done, and the best model was kept. The model was trained with google Colab for ~7 hours, utilizing a T4 GPU.

![loss](https://github.com/KNis031/ID2223-PPROJECT/blob/main/trainingloss.png)

*loss curves: blue - validation, red - training*

### Testing
Evaluation metrics used were Area Under Receiver Operating Characteristic Curve (ROC-AUC), Area Under Precision Recall Curve (PR-AUC) and Accuracy. A common metric used for multi-label tasks is ROC-AUC, where the separation capabilities of the model are measured with different classifier thresholds. The number of correctly classified positive examples (TP) are plotted against the number of incorrectly classified negative examples (FP) for different thresholds. This produces a curve for which the area under is calculated. [3] The effective range of the ROC-AUC is [0.5, 1.0], i.e. a model with no separation capability would score 0.5, while a perfect model scores 1.0. [4]. AUC-ROC can present an overly optimistic view. Therefore by replacing the TP and FP rates of the AUC-ROC with Precision and Recall, the PR-AUC could be considered a more robust metric [3]. Lastly, accuracy is the most strict metric, a percentage of the samples which had all their labels classified correctly.

Test results on the test set is shown in the table below:

| ROC-AUC (macro) | ROC-AUC (micro) | PR-AUC (macro) | PR-AUC (micro) | Accuracy (subset)|
|-----------------|-----------------|----------------|----------------|----------|
| 0.938           | 0.966           | 0.415          | 0.543          | 0.249    |

### Inference and App(s)
After training, the best model was registered with Hopsworks together with the scaler for inference on new datapoints and a JSON file, translating outputs to tags. A second JSON file, for storing the freesound.org id of a sound together with the sound's 5 most probable tags output by the model is also stored on hopsworks. A script deployed on modal runs every three days, fetching the 5 most recent sounds on freesound.org, tags them, and stores .ogg files, spectrograms and outputs on hopsworks. The sound files and spectrograms are overwitten every 3 days by a new run, but the JSON of predictions are kept (see next section). A monitoring app on huggingface spaces fetches the sounds, predictions and spectrograms and can be interacted with. A second app in which a user can upload their own <10 s sound and have it classified by the model was also created.

### Discussion and Future Extensions
There were a number of planned functionalities I did not have time to implement. Notably i had planned for both of the apps to be more interactive. With the *monitoring app*, the user were meant to be able to remove bad suggestions by the model and add their own. Support for implementing this is present through the id2prediction JSON file, and could therefore be somewhat easily implemented. With the *Try-it-with-your-sound app* I had planned for the user's sound to also be uploaded to freesound.org, so that the purpose of using it, for auto-generating tags, would be clearer. By registering the user-uploaded sound on freesound.org it would also get an id consistent with the monitoring app and id2prediction store on hopsworks. Implementing these features would not be hard to do, but requires further API permissions. The *Try-it-with-your-sound app* currently also relies on a model stored with the app on huggingface, this is unnecessary as the files it needs are already present on hopsworks, a quick fix of fetching these instead could be implemented. This would also be better if, for example the model were to change. Lastly, there are two possible sources of training-set/inference skew. First, the scaler used for inference is the same scaler used in creating the dataset, but with clipping turned on. It seems as most new points in the inference pipeline get radically clipped, and i am not entirely sure why this is. Second, given the current API permissions, the current implementation downloads all files as high quality .ogg files. In the paper describing the dataset, it is not explicitly mentioned what sound filetype is used. It is likely however that the filetypes used for creating the dataset were the filetypes of which each sound was uploaded as. Downloading the sound files in their original filetype is possible with further API permissions, and could increase inference performance.

## Apps
you can view the app(s) here:
[Monitor](https://huggingface.co/spaces/karl-sim/FreesoundMonitor)
and,
[Try it with your sound!](https://huggingface.co/spaces/karl-sim/FreesoundTry)

## Repo structure
The repo is structured into six directories, such that each of them could run independent of the others:

  * __/Modal_script__
    * __tagging_daily.py__ - script which is deployed on modal
    * __/best_model__ - dir of the model, which the script fetches from Hopsworks (does not need to be here)

  * __/Monitor_app__
    * __app.py__ - the script of the monitoring app on Hopsworks

  * __/Training_local__ - scripts for training on local machine
    * __/best_model__ - dir storing the best model
    * __/checkpoint__ - dir storing checkpoints during training
    * __/data__ - dir for storing the datasets
    * __/logs__ - dir which test results are written to
    * __CNN_model.py__ - the pytorch definition of the model
    * __dataloader.py__ - definition of the MelSpecDataset using pytorch's Dataset class
    * __solver.py__ - definition of the solver class, which handles training and testing

  * __/Training_colab__ - scripts for training using google Colab
    * __/best_model__ - dir storing the best model
    * __/checkpoint__ - dir storing checkpoints during training
    * __/data__ - dir for storing the datasets
    * __/logs__ - dir which test results are written to
    * __Training.ipynb__ - notebook of the entire training + testing pipeline

  * __/Try_app__
    * __/example_sounds__ - dir of two sound files that can be used with the app
    * __app.py__ - the script of the app
    * __utils.py__ - a number of functions required for real time inference
    * __others__ - the rest of the files should not need to be here as app.py could fetch these from Hopsworks. In the current version, this is not implemented.

  * __/Utils__
    * __model_to_hops.ipynb__ - notebook for the initial push of the model and id2prediction JSON file.
    * __utils.py__ - a number of functions used in other places of the program

To run the training and testing pipeline I recommend the notebook /Training_colab/Training.ipynb
To run inference on a single datapoint I recommend having a look at the functions and the bottom of /Utils/utils.py

### References
[1] Favory, X. et al. (2020) Coala: Co-aligned autoencoders for learning semantically enriched audio representations, arXiv.org. Available at: https://arxiv.org/abs/2006.08386
[2] Tzanetakis, G. and Cook, P. (2002) musical genre classification of audio signals. IEEE Transactions on Speech and audio processing, 10, 293-302. - references - scientific research publishing. Available at: https://www.scirp.org/(S(351jmbntvnsjt1aadkozje))/reference/referencespapers.aspx?referenceid=2069610
[3] Jesse Davis and Mark Goadrich. The relationship between precision-recall and roc curves. In Proceedings of the 23rd International Conference on Machine Learning, ICML ’06, page 233–240, New York, NY, USA, 2006. Association for Computing Machinery
[4] Choi, K., Fazekas, G. and Sandler, M. (2016) Automatic tagging using deep convolutional neural networks, arXiv.org. Available at: https://arxiv.org/abs/1606.00298
