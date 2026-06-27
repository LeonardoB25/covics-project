## Credits and Acknowledgments
This project builds upon and adapts code from the following official repositories:
- **DenseASPP**: [DeepMotionAIResearch/DenseASPP](https://github.com/DeepMotionAIResearch/DenseASPP)
- **DeepLabV3+**: [VainF/DeepLabV3Plus-Pytorch](https://github.com/VainF/DeepLabV3Plus-Pytorch)

The original architectures have been modernized and adapted to handle custom modern datasets, 
including weighted sampling and dynamic 512x512 random cropping during transfer learning.

All the modifications can be found on my [project repository](https://github.com/LeonardoB25/covics-project)


## Colab Instructions

### Parameters
- if you run on GPU, you can put INFERENCE_IMG_AMOUNT up to SUBSET_SAMPLE_AMOUNT (usually 150), otherwise it is suggested to use the value 3

### Datasets
Mapillary Vitas is a very heavy dataset (>20GiB): download and extraction can take up to 20 minutes. If you want a minimal run, instead of running all the colab, just run:
1. "Imports and constants"
2. "Datasets/Common utility functions",
3. "Datasets/Indian Driving Dataset",
4. "Models/DenseASPP/Setup model and weights"
5. "Models/DenseASPP/Inference on IDD"
6. "Models/Deeplab v3+/Setup model and weights"
7. "Models/Deeplab v3+/Inference on IDD"