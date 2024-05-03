# DFE Model Backbones (Pytorch)

## Summary 
This folder contains the code for constructing the DFE model using various backbones.
The code **models.py** is the main code, constructing the different DFE autoencoder where the backbones are from the other python files in this folder.
These codes are imported in the **DFE_train_torch.py**, **DFE_tracking_torch_Peter.py** and **DFE_eval_torch.py**.

[added by Peter]

### Enviroment setup

Same as the DFE_torch folder.

##Notes:

You can change the model structure here, but you need to make sure the output dimension of encoder is the same as the input dimension of decoder.
When you define a new encoder and decoder, you need to name them like "encoder" and "decoder" so that they can be distinguished when loading the encoder checkpoint.






