# ID2223-PROJECT

## Short description
*This project is a part of the course ID2223 Scalable machine learning, given at KTH, fall 2023*

I have created an app that suggests relevant tags for uploads to freesound.org. Posed as a machine learning problem it is a multiclass, multi-label classification task, where a number of string-labels are suggested based on an analysis of the audio data.

**Dataset:** The model is trained with the [dataset used in COALA](https://zenodo.org/records/3887261), a dataset of 170793+19103 mel-spectrograms computed over <10 s audio clips from [freesound.org](https://freesound.org/). For targets, the dataset uses multi-hot encoded tags from a vocabulary of 1000 user entered tags. The dataset came split in a training and validation split. I ended up splitting it further into training, validation and testing sets. Of the 170793 entries in the original training set 70% were used for training and the remaining 30% were used for validation. The set referenced as 'validation' in the dataset description, I used for testing.

**Model and Training:** With model architecture, I took inspiration from a number of papers, including one of the models explored by Favory et.al [1] which the dataset were created for. After some trial and error, I landed in using a Convolutional Neural Network architecture of four convolutional layers and two linear layers. I initially planned to train the model for 100 epochs, but after inspecting the loss after half the training time (50 epochs) it seemed as the model had reached the best performance after just 10 epochs. I therefore did not bother with training for longer. I trained with google Colab for ~7 hours, utilizing a T4 GPU. Test results on the test set i shown in the table below:

| ROC-AUC (macro) | ROC-AUC (micro) | PR-AUC (macro) | PR-AUC (micro) | accuracy |
|-----------------|-----------------|----------------|----------------|----------|
| 0.938           | 0.966           | 0.415          | 0.543          | 0.249    |

**Inference and App:** After training, the best model was registered with Hopsworks together with the scaler for inference on new datapoints and a JSON file, translating outputs to tags. A second JSON file, for storing the freesound.org id of a sound together with the sound's 5 most probable tags output by the model is also stored on hopsworks. A script deployed on modal runs every three days, fetching the 5 most recent sounds, tagging them, and storing .ogg files, spectrograms and outputs on hopsworks. The sound files and spectrograms are overwitten every 3 days by a new run, but the JSON of predictions are kept (see next section). A monitoring app on huggingface spaces fetches the sounds, predictions and spectrograms and can be interacted with. A second app in which a user can upload their own <10 s sound and have it classifed by the model was also created.

**Future extensions:** There were a number of planned functionalities I did not have time to implement. Notably i had planned for both of the apps to be more interactive. With the *monitoring app*, the user were meant to be able to remove bad suggestions by the model and add their own. Support for implementing this is present through the id2prediction JSON file, so it could be a future quick fix. With the *Try-it-with-your-sound app* I had planned for the user's sound to also be uploaded to freesound.org, so that there would be a purpose to using it, by auto-generating tags. By registering the user-uploaded sound on freesound.org it would also get an id consistent with the monitoring app and id2prediction store on hopsworks. The *Try-it-with-your-sound app* currently also relies on a model stored with the app on huggingface, this is unnecessary as the files it needs are already present on hopsworks, a quick fix of fetching these instead could be implemented. This would also be better if, for example the model were to change. Lastly, model preformence on new datapoints could likely be increased by experementing with other scaling settings.

## Apps
you can view the app(s) here:
[Monitor](https://huggingface.co/spaces/karl-sim/FreesoundMonitor)
and,
[Try it with your sound!](https://huggingface.co/spaces/karl-sim/FreesoundTry)

## Repo structure
The repo is structured into six directories, such that each of them could run independent of the others:

/**
  * __/Modal_script__
    * __tagging_daily.py__ - script which is deployed on modal
    * __/best_model__ - dir of the model, which the script fetches from Hopsworks (does not need to be here)

  * __/Monitor_app
    * __app.py__ - the script of the monitoring app on Hopsworks

  * __/Taining_local__ - scripts for training on local machine
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
    * __utils.py__ a number of functions used in other places of the program
**/

## Full report

## References
 [1] https://arxiv.org/abs/2006.08386
