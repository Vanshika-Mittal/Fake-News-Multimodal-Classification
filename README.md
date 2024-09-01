# Fake News Classification Using the Fakeddit Dataset

## Introduction
- Fakeddit is a fine-grained multimodal fake news detection dataset, designed to advance efforts to combat the spread of misinformation in multiple modalities.
- I worked on classifying data into 6 pre-defined classes: authentic/true news content, Satire/Parody, content with false connection, imposter content, manipulated content and misleading content.
- For the Image-Feature Extractor, I used a pre-trained ```ResNet50 model``` trained on the ImageNet dataset for image classification tasks.
- For the Text-Feature Extractor, I used a pre-trained ```Bertmodel``` trained on the English Wikipedia and Toronto Book Corpus in lower cased letters.

    - Base Reference Paper: [r/Fakeddit: A New Multimodal Benchmark Dataset for Fine-grained Fake News Detection](https://aclanthology.org/2020.lrec-1.755/)
 
## Model Architecture

Sample Model Architecture:
![image](https://github.com/user-attachments/assets/d6c31c57-ed76-4e51-a917-637a21eade95)

#### BERT and BERT embeddings
- BERT uses a bi-directional approach considering both the left and right context of words in a sentence, instead of analyzing the text sequentially.
- These vectors are used as high-quality feature inputs to downstream models. NLP models such as LSTMs or CNNs require inputs in the form of numerical vectors, hence BERT is a good option for encoding variable length text strings.

#### ResNet50
- ResNet50 is a deep learning model launched in 2015 by Microsoft Research for the purpose of visual recognition. The model is 50 layers deep.
- ResNet50's architecture (including shortcut connections between layers) significantly improves on the vanishing gradient problems that arise during backpropagation which allows for higher accuracy.
- The skip connections in ResNet50 facilitate smoother training and faster convergence. Thus making it easier for the model to learn and update weights during training.

#### Late Fusion
-  Late fusion processes the data of each sensor independently to make a local prediction. These individual results are then combined at a higher level to make the final fused prediction.
- The advantage of late fusion is its simplicity and isolation. Each model gets to learn super rich information on its modality.


