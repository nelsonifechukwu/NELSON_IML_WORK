# NELSON_IML_WORK

**This is a project on the systematic investigation of methods for slip detection and prediction in robotic manipulation carried out in the IML laboratory.**

# Steps

## READING THE XELA SENSOR DATA
- Import the required libraries
- Read the xela sensor data along it's slip labels
  - NB: Replace directory with the path of the data in train2dof folder
- Concat all the data from all subfolders in the train2dof folder

## RE-ARRANGEMENT OF TABULAR DATA INTO IMAGES
- Arrange the tabular data into x, y, z
- Scale the data
- Arrange each row of the tabular data into images of the form 3 * 4 * 4 of tx_x, tx_y, tx_z
- Rotate the image 90 deg anti-clockwise to align with the matrix of the xela sensor

## CNN IMPLEMENTATION
