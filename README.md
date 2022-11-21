# NELSON_IML_WORK

**This is a project on the systematic investigation of methods for slip detection and prediction in robotic manipulation carried out in the IML laboratory.**

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
  - The Differences are:
    - dwsdsd 



