# Transfer-Learning-Using-PyTorch


# Overview
  * In this project, we utilize a pretrained VGG16 model to classify images using transfer learning. The following steps are implemented:

     * Setting Random Seeds for Reproducibility
     * Checking for GPU Availability
     * Data Preprocessing and Transformations (resize, center crop, tensor(scall), normalize)
     * Model Loading (Pretrained VGG16)
     * Training and Evaluation Loop
     * Model Evaluation on Test and Training Data
   

# Steps :-
  1. Setting Random Seeds for Reproducibility
     * Random seeds are set to ensure that the results are reproducible across different runs.
    
  2. Check for GPU Availability
     * Before training, the code checks if a GPU is available, ensuring faster model training if supported by your hardware.
    
  3. Create a 4x4 Grid of Images
     * Visualize the images in a 4x4 grid to get a better sense of the dataset.
    
  4. Data Transformations
     * Data is preprocessed using the following transformations to match the input requirements of VGG16:
         * Resize (3, 256, 256)
         * Center Crop (3,224, 224)
         * Tensor (0,1)  (# Scall)
         * Normalize
      
  5. Custom Dataset Class
     * In this project, we define a custom dataset class to handle image data. The CustomDataset class allows us to load the dataset, apply transformations, and return the images and their corresponding labels.
         * Key Features:
           * Resizing: Images are resized to the required dimensions (28x28 in this case).
           * Grayscale to RGB Conversion: If the image is grayscale, it is converted to an RGB format by repeating the grayscale values along the color channels.
           * Transformation: Any custom transformations, such as resizing, cropping, or normalization, are applied during the image loading process.
          
  6. Fetch Pretrained VGG16 Model
     * We fetch the VGG16 model pretrained on ImageNet and transfer it to the appropriate device (GPU or CPU).
     * The vgg16 model is loaded with weights pretrained on ImageNet. By using this pretrained model, we can leverage the learned features from ImageNet and fine-tune the model for our specific task.
    
  7. Training the Model
     * Define the optimizer, loss function, and the training loop. We will use a learning rate of 0.0001 and train for 10 epochs
    
  8. Model Evaluation (Test or Training)
     * After training, the model is evaluated on both the test and training datasets. Below are the evaluation results:
        * Test Accuracy: 91%
        * Training Accuracy: 100%
      

# Conclusion
  * This project demonstrates the effectiveness of transfer learning with the pretrained VGG16 model for image classification. By fine-tuning the model and freezing the feature layers, we were able to achieve:

   Test Accuracy: 91%
   
   Training Accuracy: 100%
   
This showcases the power of transfer learning for real-world applications.




      
