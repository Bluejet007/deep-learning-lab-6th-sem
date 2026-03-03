# Regularization Qn
Use the pretrained ResNet-18 model and fine-tune it on the CIFAR-10 dataset by randomly selecting 1000 images for training and 300 images for validation from the original dataset. During training, report the training loss and training accuracy for each epoch, along with the validation (test) loss and validation accuracy. After training, plot the loss and accuracy curves for both training and validation sets, and provide a brief comment on the observed model performance based on these results.

## 1. Study of Data Augmentation
1. Increase the number of training samples to 2000 (randomly selected from CIFAR-10), keep the validation set fixed at 300 samples, and train the pretrained ResNet-18 under the same settings. Report train loss/accuracy and validation loss/accuracy per epoch, plot the loss and accuracy curves, and briefly comment on how the model performance changes compared to the 1000-sample experiment (e.g., generalization gap, overfitting reduction, validation improvement).
2. Apply multiple data augmentation transformations (e.g., random rotation, horizontal flip, etc.) to each of the 1000 selected training images to generate an additional 1000 augmented images. Concatenate these augmented samples with the original 1000 images to form a 2000-sample training set, and then fine-tune the pretrained ResNet-18 using this expanded dataset. Keep the validation set fixed at 300 images, report train/validation loss and accuracy per epoch, plot the corresponding curves, and briefly comment on the effect of augmentation on the model’s generalization performance.

## 2. Study of L2 Regularization (Without Data Augmentation)
Use the pretrained ResNet-18 model on the CIFAR-10 dataset (with the same training and validation splits as before) and apply L2 regularization using weight decay. Vary the regularization strength across a range of values (e.g., 0.0001, 0.001, 0.01, 0.1).
1. For each value of weight decay:
    - Train the model under identical settings.
    - Plot the training and validation loss curves.
    - Plot the training and validation accuracy curves.
2. Plot Validation Accuracy vs Weight Decay to analyze the impact of different regularization strengths. Finally, provide a brief commentary on the model’s performance in each case, discussing trends such as overfitting, underfitting, generalization gap, and the effect of increasing regularization strength on accuracy.

## 3. Data Augmentation + L2 Regularization
Using the training setup from 1(b) (original 1000 samples + 1000 augmented samples), apply L2 regularization with the optimal weight decay value identified in 2(b). Fine-tune the pretrained ResNet-18 model using this configuration.
- Plot the training and validation loss curves.
- Plot the training and validation accuracy curves.

Compare the results with the baseline model (trained without augmentation and without L2 regularization). Provide a detailed commentary on:
- Changes in generalization error (train–validation gap)
- Improvement (or decline) in validation accuracy
- Behavior of training and validation loss
- Overall stability of training

Conclude by discussing whether combining augmentation and L2 regularization leads to improved generalization performance compared to the baseline setup.

## 4. dropout
Model.fc = nn.Sequential(
   nn.Dropout(0.5),
   nn.Linear(in_features, 10)
)

Apply dropout (0.1,0.2,0.3,0.4,0.5) on the FC layer and observe the performance. How do you apply droput in CNN layer (nn.Dropout2d)?

## 5. Early stopping
Apply early stopping with patience = 5