# model_compression
Exploring model compression techniques


### 1. Effect of pruning on interpretability using SHAP

- Code in [pruning_interpretability.ipynb](pruning_interpretability.ipynb)
- Network used: LeNet
- Training data: Oxford-IIIT Pets dataset 
- Interpretability technique examined: SHAP
- Structured pruning applied to convolutional & fully connected layers using `ln_structured()` function (available in PyTorch)
- Observation: After pruning, SHAP indicates lower confidence in the pixels that are considered 'relevant' for classification. However, upon fine-tuning the pruned network, the interpretability is regained - almost indistinguishable from the original network. 
