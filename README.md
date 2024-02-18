# model_compression
Exploring model compression techniques


### 1. Effect of pruning on interpretability using SHAP

- Code in [pruning_interpretability.ipynb](pruning_interpretability.ipynb)
- Network used: LeNet
- Dataset: Oxford-IIIT Pets 
- Interpretability technique examined: SHAP
- Structured pruning applied to convolutional & fully connected layers using `ln_structured()` function (available in PyTorch)
- Observation: After pruning, SHAP indicates lower confidence in the pixels that are considered 'relevant' for classification. However, upon fine-tuning the pruned network, the interpretability is regained - almost indistinguishable from the original network.


### 2. Is it "knowledge distillation" or just another regularization technique?

- Code in [knowledge_distillation.ipynb](knowledge_distillation.ipynb)
- Teacher network: ResNet18. Student network: LeNet
- Dataset: CIFAR10
- Attempting to answer: Should the performance improvement for a student network when a distillation loss is used be attributed to the teacher network's logits? Or does a random regularization term help equally? 
