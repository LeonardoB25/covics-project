## Credits and Acknowledgments
This project builds upon and adapts code from the following official repositories:
- **DenseASPP**: [DeepMotionAIResearch/DenseASPP](https://github.com/DeepMotionAIResearch/DenseASPP)
- **DeepLabV3+**: [VainF/DeepLabV3Plus-Pytorch](https://github.com/VainF/DeepLabV3Plus-Pytorch)

The original architectures have been modernized and adapted to handle custom datasets, 
including weighted sampling and dynamic 512x512 random cropping during training.


## Colab Instructions

### Parameters
- if you run on GPU, you can put INFERENCE_IMG_AMOUNT up to SUBSET_SAMPLE_AMOUNT (usually 150), otherwise it is suggested to use the value 3

### Datasets
- Mapillary Vitas is a very heavy dataset (>20GiB): if you want a minimal run, instead of running all the colab, just press "Import and constants", "Datasets/Common utility functions", "Indian Driving Dataset", "Models/Common utility functions", "Models/Dense ASPP/Setup model and weights"