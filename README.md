# AI-Generated Synthetic Images Detection
![download (1)](https://github.com/user-attachments/assets/21f50705-ef92-4106-b776-157810235b6c)

### Overview
The boom of AI-powered content generation and increasing interest in the research field of Deep Learning has led to widely accessible (and trending) tools that can produce content of any kind: text, image, audio, and video. While AI isn't new, the the availability of powerful low-code generative AI applications to the public is. AI-generated content can often be indistinguishable from its authentic counterparts, posing a threat to the credibility of digital media. The underlying dangers of the misuse of GenAI have already come to surface with deepfakes, voice cloning, fakes news, disinformation, identity theft and various types of scams. In fact, a survey conducted by Microsoft in 2023 shows that 71% of respondents are worried about AI scams.

In this project, we focus on image generation, which can have multiple societal effects, especially on people not familiar with this kind of technology. Our task is to train a neural network to identify whether an image is real or AI-generated.

### Table of Contents
- [Files](#files)
- [Dataset](#dataset)
- [Authors](#authors)
- [License](#license)

### Files
- `functions.py` : All the preprocessing, model training, evaluation functions and CNN class used in the report.
- `report.ipynb` : The full machine learning pipeline. EDA, preprocessing, model training, transfer learning, evaluation, gradio deployment.

### Dataset
<a href="https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images">CIFAKE: Real and AI-Generated Synthetic Images</a> is a comprehensive collection of 60,000 synthetically-generated images and 60,000 real images (collected from CIFAR-10). The dataset contains two classes, labelled as "REAL" and "FAKE". There are 100,000 images for training (50k per class) and 20,000 for testing (10k per class). Since the training an test sets have 50% of each class, there is no class imbalance that needs to be taken care of for our binary classification task.

### Authors
- Stefania Syrsiri - [LinkedIn](https://www.linkedin.com/in/stefania-syrsiri/) | [GitHub](https://github.com/stefsyrsiri)
- Anna Galanopoulou - [LinkedIn](https://www.linkedin.com/in/anna-galanopoulou/) | [GitHub](https://github.com/tzitzi2662)

### License
Distributed under the [MIT](https://choosealicense.com/licenses/mit/) License
