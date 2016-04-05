# Machine Learning project submission

## Goal
The purpose of this project is to prepare a machine learning algorithm that will be able to predict what kind of exercise is done based on data 
gathered from accelerometers on the belt, forearm, arm, and dumbell.
The data comes from the Human Activity Recognition study:
Ugulino, W.; Cardador, D.; Vega, K.; Velloso, E.; Milidiu, R.; Fuks, H. "Wearable Computing: Accelerometers' Data Classification of Body Postures and Movements".
More information along with the source can be found here: 
http://groupware.les.inf.puc-rio.br/har

## Model prepared
-Model was calculated in R using mainly Caret package
-The data was divided into 60% for the training partition, and 40% for the testing one
The following steps were taken:
  1. The data was filtered to leave only variables describing accelerators (except the outcome variable).
  2. The summarized data was analyzed. Three things were noted:
    - For variables ‘var_total_accel_belt’, ‘var_accel_arm’, ‘var_accel_dumbbell’, ‘var_accel_forearm’ very high number of values are missing.
    - For variables "accel_belt_x","accel_belt_z", "accel_arm_x", "accel_arm_y","accel_arm_z", "accel_dumbbell_x","accel_dumbbell_z",  "accel_forearm_x",  "accel_forearm_z"
    standard deviation relatively higher than mean.
    - Variables "total_accel_belt", "accel_belt_y" and "accel_belt_z" are correlated with each other.
  Based on the above:
  3. Variables with high number of NA were excluded from data set. This way the bias was reduced.
  4. All the variables in data set were standarized. This way variance of the model was reduced.
  5. Two principal components were calculated based on three correlated variables, and replaced them in the data set. This would simplify calculations and improve accuracy.
  6. Several models relatively easy to compute were tried - i.e. linear discriminant analysis, CART and bagged CART.
Bagged CART turned out to have the best accuracy on the testing set - it was 0.9003.
The out-of-sample error was measured by percentage of predictions were incorrect on the testing sample. This value was ~10%.