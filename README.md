# Transfer-Learning-Using-PyTorch


# Overview
  * In this project, we utilize a pretrained VGG16 model to classify images using transfer learning. The following steps are implemented:

     * Setting Random Seeds for Reproducibility
     * Checking for GPU Availability
     * Data Preprocessing and Transformations
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
