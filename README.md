# NELSON_IML_WORK

**This is a project that constitutes my research work on the systematic investigation of methods for slip detection and prediction in robotic manipulation carried out at the [IML laboratory](https://intmanlab.com/). See more on my work history [here](https://github.com/nelsonifechukwu/NELSON_IML_WORK/blob/b2a4897fc904ffaf1a70a4c53e4c45b92d5c31b0/Systematic%20investigation%20of%20methods%20for%20slip%20detection%20and%20prediction%20in%20robotic%20manipulation.pdf).**

**Access my presentation [here](https://docs.google.com/presentation/d/1YGkoTk0kPfvVGeXpBM2Hz6Qta-XnyCLG41HLs9zi9ws/edit?usp=sharing).**

# STEPS

## READING THE XELA SENSOR DATA
- Import the required libraries
- Read the xela sensor data (input) along its slip labels (output)
  - NB: Replace the directory with the path of the data in train2dof folder
- Concat all the data from all subfolders in the train2dof folder

## RE-ARRANGEMENT OF TABULAR DATA INTO IMAGES
- Arrange the tabular data into x, y, z
- Scale the data
- Arrange each row of the tabular data into images of form 3 * 4 * 4 of tx_x, tx_y, tx_z
- Rotate the image 90 deg anti-clockwise to align with the matrix of the xela sensor

## DENSE NEURAL NETWORK IMPLEMENTATION
- Open the dnn_imbalance_tactile.ipynb file
- Read the xela sensor data (input) along its slip labels (output) in tabular form 
- Define the metrics for assessment of the model for both slip detection (detection_metrics) and slip prediction (slip_metrics)

  - ### SLIP DETECTION
        - Optional: Apply SMOTE techniques for the imbalanced classification in the slip_labels
        - Split the data into train and test data
        - Scale the data
        - Convert to tensor values 
        - Create the batch training structure
        - Create the Neural Network Architecture (3 layers)
        - Instantiate the model, define the learning rate, loss, and optimizer
        - Train the model
        - Print the loss and accuracy per epoch
        - Check the metrics of the model using the actual slip label and predicted slip label

  - ### SLIP PREDICTION
        - Read the xela sensor data (input) along its slip labels (output) in tabular form using read_file(1, n) where n is the number of time steps into the future for prediction
        - Optional: Apply SMOTE techniques for the imbalanced classification in the slip_labels
        - Split the data into train and test data
        - Scale the data
        - Convert to tensor values 
        - Define the batch training structure
        - Define the Neural Network Architecture (3 layers)
        - Instantiate the model, define the learning rate, loss, and optimizer
        - Train the model
        - Print the loss and accuracy per epoch
        - Check the metrics of the model using the actual slip label and predicted slip label


## CNN IMPLEMENTATION
- Open the cnn_pytorch_tactile.ipynb file
- Start from the CN IMPLEMENTATION markdown
- Upscale the 3 * 4 * 4 image to 3 * 16 * 16 to have enough features to train the CNN model
- Convert the images to tensor values
- Split the data into test and training sets
- Define the batch training architecture
- Define the CNN architecture (the architecture present in the file gave an awesome performance during experimentation)
- Instantiate the model, define the learning rate, loss, and optimizer
- Train the model
- Print the loss and accuracy per epoch
- Define the metrics 
- Check the metrics of the model using the actual slip label and predicted slip label

## RESNET IMPLEMENTATION
- Follows a similar procedure to the CNN implementation
The Differences are:
    - Upscale the image from 3 * 4 * 4 to 3 * 224 * 224 (The cluster machine kept crashing on this task)
    - Create the batch training structure
    - Define the model to use the last layer of a pre-trained RESNET model by freezing all the RESNET network except the final layer
    - Train the model
    - Print the loss and accuracy per epoch
    
## RNN IMPLEMENTATION
- Define the feature of the RNN 
- Train the model
- Issues occur due to vanishing and exploding gradients

## LSTM IMPLEMENTATION
- Define the architecture of the LSTM network
- It's a many-to-one problem with 10 sequences

## CONVLSTM IMPLEMENTATION
- Rearrange the data to cater to both temporal and spatial features of the sensor data
- Form a convolution from the images
- Train an LSTM model
    
# OUTCOME
The outcome of this research experience showed that slip detection and slip prediction performed better with CNN than with the DNN. 

## FOR SLIP DETECTION
<img width="990" alt="slip detection" src="https://github.com/nelsonifechukwu/NELSON_IML_WORK/assets/44223263/5d822f53-5c1b-4d4a-825e-cdf00fb322d4">

## FOR SLIP PREDICTION
<img width="665" alt="Screenshot 2023-08-17 at 16 34 40" src="https://github.com/nelsonifechukwu/NELSON_IML_WORK/assets/44223263/5979cf2c-1671-4883-8514-fc9a2b0e03e2">

<img width="665" alt="Screenshot 2023-08-17 at 16 34 40" src="https://github.com/nelsonifechukwu/NELSON_IML_WORK/assets/44223263/76d6dcfe-3b1d-4ef6-bf2b-5a932c7f8feb">

N.B: My learning and practice of concepts in researching Artificial Intelligence can be found in practice
