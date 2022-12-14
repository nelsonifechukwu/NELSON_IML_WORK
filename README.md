# NELSON_IML_WORK

**This is a project tht constitutes my research work on the systematic investigation of methods for slip detection and prediction in robotic manipulation carried out at the [IML laboratory](intmanlab.com). See more [here](https://github.com/nelsonifechukwu/NELSON_IML_WORK/blob/b2a4897fc904ffaf1a70a4c53e4c45b92d5c31b0/Systematic%20investigation%20of%20methods%20for%20slip%20detection%20and%20prediction%20in%20robotic%20manipulation.pdf).**

# STEPS

## READING THE XELA SENSOR DATA
- Import the required libraries
- Read the xela sensor data (input) along it's slip labels (output)
  - NB: Replace directory with the path of the data in train2dof folder
- Concat all the data from all subfolders in the train2dof folder

## RE-ARRANGEMENT OF TABULAR DATA INTO IMAGES
- Arrange the tabular data into x, y, z
- Scale the data
- Arrange each row of the tabular data into images of the form 3 * 4 * 4 of tx_x, tx_y, tx_z
- Rotate the image 90 deg anti-clockwise to align with the matrix of the xela sensor

## DENSE NEURAL NETWORK IMPLEMENTATION
- Open the dnn_imbalance_tactile.ipynb file
- Read the xela sensor data (input) along it's slip labels (output) in tabular form 
- Define the metrics for assessment of thte model for both slip detection (detection_metrics) and slip prediction (slip_metrics)

  - ### SLIP DETECTION
        - Optional: Apply SMOTE teechniques for the imbalanced classification in the slip_labels
        - Split the data into train and test data
        - Scale the data
        - Convert to tensor values 
        - Create the batch training structure
        - Create the Neural Network Architecture (3 layers)
        - Instantiate the model, define the leqarning rate, loss, and optimizer
        - Train the model
        - Print the loss and accuracy per epoch
        - Check the metrics of the model using the actual slip label and predicted slip label

  - ### SLIP PREDICTION
        - Read the xela sensor data (input) along it's slip labels (output) in tabular form using read_file(1, n) where n is the number of time steps into the future for prediction
        - Optional: Apply SMOTE teechniques for the imbalanced classification in the slip_labels
        - Split the data into train and test data
        - Scale the data
        - Convert to tensor values 
        - Define the batch training structure
        - Define the Neural Network Architecture (3 layers)
        - Instantiate the model, define the leqarning rate, loss, and optimizer
        - Train the model
        - Print the loss and accuracy per epoch
        - Check the metrics of the model using the actual slip label and predicted slip label


## CNN IMPLEMENTATION
- Open the cnn_pytorch_tactile.ipynb file
- Start from the CN IMPLEMENTAATION markdown
- Upscale the 3 * 4 * 4 image to 3 * 16 * 16 to have enough features to train the CNN model
- Convert the images to tensor values
- Split the data into test and training sets
- Define the batch traning architecture
- Define the CNN architecture (the architecture present in the file gave an awesome performance during experimentation)
- Instantiate the model, define the leqarning rate, loss, and optimizer
- Train the model
- Print the loss and accuracy per epoch
- Define the metrics 
- Check the metrics of the model using the actual slip label and predicted slip label

## RESNET IMPLEMENTATION
- Follows similar procedure of the CNN implementation
The Differences are:
    - Upscale the image from 3 * 4 * 4 to 3 * 224 * 224 (The cluster machine kept crashing on this task)
    - Create the batch training structure
    - Define the model to use the last layer of a pretrained RESNET model by frezzing all the RESNET network except the final layer
    - Train the model
    - Print the loss and accuracy per epoch
    
## RNN IMPLEMENTATION
- This is still a work in progress
- My current work in on defining the RNN model architecture
    
# OUTCOME
The outcome of this research experience showed that slip detection and slip prediction performed better with CNN than the DNN. 

## FOR SLIP DETECTION
<img width="1131" alt="Screenshot 2022-11-21 at 19 11 21" src="https://user-images.githubusercontent.com/44223263/203129285-ecf62af4-88c7-4657-8ddd-a62738310133.png">

## FOR SLIP PREDICTION
<img width="1129" alt="Screenshot 2022-11-21 at 19 12 46" src="https://user-images.githubusercontent.com/44223263/203129569-7f803cfc-46b9-49a7-9267-73141fd3cce3.png">

<img width="1128" alt="Screenshot 2022-11-21 at 19 13 29" src="https://user-images.githubusercontent.com/44223263/203129695-62e0e777-910c-4ebe-aa89-0cb08e7b77bf.png">

See more analysis [here](https://github.com/nelsonifechukwu/NELSON_IML_WORK/blob/4e52efab9755bc6cb38aa423ce8af755f16b6fee/IML%20-%20Sheet1.pdf).


