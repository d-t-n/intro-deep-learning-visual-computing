# Intro to Deep Learning in Visual Computing
### Introduction to Deep Learning in Visual Computing

Deep Learning has revolutionized the field of visual computing, paving the way for remarkable advancements in image and video analysis, computer vision, and other related applications. Visual computing is a multidisciplinary domain that involves processing, analyzing, and understanding visual information, such as images, videos, and graphical data. Deep Learning, a subset of machine learning, has demonstrated unparalleled capabilities in tackling complex visual tasks that were once deemed nearly impossible.

### Foundations of Deep Learning in Visual Computing:

Deep Learning is built upon artificial neural networks, which are computational models inspired by the human brain's neural structure. These networks consist of layers of interconnected nodes (neurons) that process and transform data. In the context of visual computing, these networks are designed to automatically learn and extract hierarchical features from raw visual input, such as pixels in images or frames in videos. Convolutional Neural Networks (CNNs) are a prominent architecture within deep learning that excel in image-related tasks due to their ability to capture spatial hierarchies.

### Key Concepts:

- Convolutional Neural Networks (CNNs): These are specialized neural networks designed to process grid-like data, such as images. They employ convolutional layers to automatically learn local patterns and hierarchies, enabling them to identify features like edges, textures, and more complex structures.

- Transfer Learning: Training deep neural networks from scratch requires massive amounts of labeled data and computational resources. Transfer learning leverages pre-trained networks (often trained on large datasets like ImageNet) as a starting point. These networks already possess generalized feature extraction capabilities, which can then be fine-tuned for specific tasks with smaller datasets.

- Object Detection: Deep Learning has revolutionized object detection by enabling the development of algorithms that can accurately localize and classify objects within images or videos. This is crucial for applications like self-driving cars, surveillance, and medical imaging.

- Image Generation: Generative Adversarial Networks (GANs) are a class of deep learning models used to generate realistic images. One network generates images, and another network evaluates them for realism. This interplay results in the creation of images that can resemble photographs of real objects, even though they don't exist in reality.

- Semantic Segmentation: In this task, deep learning models assign a class label to each pixel in an image, effectively segmenting the image into meaningful regions. This is used in medical imaging for identifying structures within the body, in satellite imagery analysis, and even in augmented reality applications.

- Video Understanding: Deep Learning has also extended to video analysis, enabling tasks like action recognition, scene understanding, and video captioning. Recurrent Neural Networks (RNNs) and their variants are often used to process sequences of frames and extract temporal dependencies.

### Challenges:

Despite its transformative capabilities, deep learning in visual computing comes with challenges:

- Data Annotation: Deep learning models require large amounts of annotated data for training, which can be time-consuming and expensive to acquire, particularly for tasks requiring pixel-level annotations.

- Computational Resources: Training deep neural networks demands substantial computational power, which can limit access for smaller research teams or institutions.

- Interpretability: Deep learning models often operate as black boxes, making it challenging to understand the reasoning behind their decisions. This is a critical concern, especially in applications like medical diagnosis.

- Overfitting: Deep neural networks can easily overfit to training data, leading to poor generalization on unseen data. Techniques like regularization and data augmentation are employed to mitigate this.

### Conclusion:

Deep Learning has redefined the landscape of visual computing, enabling breakthroughs in image analysis, computer vision, and more. Its ability to automatically learn and extract intricate patterns from visual data has opened doors to applications ranging from medical diagnostics to entertainment. As technology advances, addressing challenges like interpretability and data scarcity will be pivotal in unlocking the full potential of deep learning in visual computing.

## Topics:
- Gain an understanding of the mathematics (Statistics, Probability, Calculus, Linear Algebra and optimization) needed for designing machine learning algorithms

- Learn how machine learning models fit data and how to handle small and large datasets

- Understand the workings of different components of deep neural networks

- Design deep neural networks for real-world applications in computer vision

- Learn to transfer knowledge across neural networks to develop models for applications with limited data

- Get introduced to deep learning approaches for unsupervised learning

- Transformers, Self-Attention and LLMs


## 1. Gain an understanding of the mathematics needed for designing machine learning algorithms

#### Introduction:
Machine Learning (ML) algorithms have transformed industries by enabling computers to learn from data and make intelligent decisions. Behind the scenes of these algorithms lies a solid foundation of mathematics. Statistics, probability, calculus, linear algebra, and optimization are essential components that power the design and development of effective machine learning models. This comprehensive guide will delve into the significance of each mathematical discipline and how they interplay in the context of ML algorithm design.

1. Statistics:
Statistics forms the bedrock of machine learning. It involves techniques for collecting, analyzing, interpreting, and presenting data. For ML, statistics is crucial in understanding data distributions, measuring central tendencies, and assessing the spread of data. Concepts such as mean, median, mode, variance, and standard deviation are fundamental in preparing data for ML models. Moreover, hypothesis testing, confidence intervals, and p-values aid in assessing the significance of results and making informed decisions about model performance.

- Applied Example: A common use of statistics in machine learning is in evaluating the performance of classification models. Metrics like accuracy, precision, recall, and F1-score help quantify how well a model is performing on different classes.

- Web Reference: Scikit-learn's documentation on model evaluation metrics: scikit-learn.org/stable/modules/model_evaluation.html

2. Probability:
Probability theory quantifies uncertainty, a core aspect of machine learning. Probability concepts, like random variables, probability distributions, and conditional probability, are vital for modeling uncertainty in data. In ML, probabilistic models such as Naïve Bayes, Hidden Markov Models, and Gaussian Processes leverage probability theory to make predictions and classifications based on observed data.

- Applied Example: Naïve Bayes is a probabilistic classifier that assumes features are conditionally independent given the class label. It's used in spam email detection, where the probability of words occurring in spam or non-spam emails is used to classify new emails.

- Web Reference: Towards Data Science article on Naïve Bayes for text classification: towardsdatascience.com/naive-bayes-for-text-classification-2b8a88a94a7c

3. Calculus:
Calculus provides the mathematical tools to understand how functions change. In ML, it's crucial for optimization and learning algorithms. Differential calculus helps in understanding gradients, which are instrumental in optimization methods like gradient descent—a fundamental technique for adjusting model parameters to minimize error. Integral calculus plays a role in calculating areas under curves, which is used in probability distributions and calculating expectations.

- Applied Example: Gradient descent is a core optimization technique used in training machine learning models. It's employed to adjust model parameters iteratively to minimize the loss function.

- Web Reference: Deep Learning Specialization on Coursera, covering gradient descent: coursera.org/specializations/deep-learning

4. Linear Algebra:
Linear algebra deals with vector spaces and linear mappings between them. In machine learning, data is often represented as vectors or matrices, and linear algebra facilitates efficient computation and manipulation of these representations. Concepts like eigenvalues, eigenvectors, and singular value decomposition (SVD) play a significant role in dimensionality reduction, feature extraction, and understanding the geometry of data in high-dimensional spaces.

- Applied Example: Principal Component Analysis (PCA) is used for dimensionality reduction in machine learning. It's applied to datasets with high dimensions to identify the most important features.

- Web Reference: Introduction to Linear Algebra by Khan Academy: khanacademy.org/math/linear-algebra

5. Optimization:
Optimization methods fine-tune model parameters to achieve the best performance. Gradient-based optimization techniques, like gradient descent and its variants (e.g., stochastic gradient descent), are used to minimize the loss function, which quantifies the difference between model predictions and actual data. Advanced optimization methods, such as convex optimization and constrained optimization, help tackle complex problems with specific constraints.

- Applied Example: Support Vector Machines (SVMs) use optimization techniques to find the hyperplane that best separates different classes in a dataset.

- Web Reference: Introduction to Support Vector Machines by Scikit-learn: scikit-learn.org/stable/modules/svm.html

6. Integration in ML Algorithm Design:
The integration of these mathematical disciplines is evident in the development of various machine learning algorithms. For instance:

- Linear Regression: Involves statistical techniques to estimate the relationship between variables, while calculus optimizes model parameters.
- Support Vector Machines: Utilizes linear algebra for mapping data to higher dimensions and optimization for finding the best separating hyperplane.
- Principal Component Analysis (PCA): Relies on linear algebra to perform dimensionality reduction, enhancing computational efficiency.
Neural Networks: Leverage calculus for gradient computation during training and linear algebra for matrix operations between layers.

- Applied Example: Neural networks, a cornerstone of deep learning, combine elements of calculus (backpropagation for gradient computation) and linear algebra (matrix operations in layers) for training complex models on large datasets.

- Web Reference: Deep Learning Specialization on Coursera, exploring neural networks: coursera.org/specializations/deep-learning

#### Conclusion:
A solid grasp of statistics, probability, calculus, linear algebra, and optimization is essential for designing and developing effective machine learning algorithms. These mathematical foundations enable practitioners to understand the underlying principles, optimize models, handle uncertainty, and make informed decisions throughout the ML pipeline. As machine learning continues to advance, a deep understanding of these mathematical concepts remains a fundamental skill for professionals in the field.

Examples described in 1.-project.ipynb


## 2. Learn how machine learning models fit data and how to handle small and large datasets

#### Introduction:
Machine learning models are designed to learn patterns and make predictions from data. Understanding how these models fit data is crucial for successful implementation. Moreover, handling varying dataset sizes, whether small or large, requires distinct strategies to ensure optimal performance. This guide will delve into the intricacies of how machine learning models fit data and the strategies to handle datasets of different scales.

1. Machine Learning Model Fitting:
Model fitting is the process of training a machine learning model to find the optimal parameters that best describe the relationship between features and labels in the data. This involves adjusting the model's internal parameters during training to minimize the difference between the predicted outputs and the actual labels. Techniques such as gradient descent and its variants are often used to iteratively update parameters and achieve the best fit.

2. Understanding Model Overfitting and Underfitting:
Overfitting occurs when a model learns the training data too well, capturing noise and irrelevant patterns, resulting in poor generalization to unseen data. Underfitting, on the other hand, happens when a model is too simple to capture the underlying patterns, leading to poor performance on both the training and test sets. Balancing model complexity to avoid overfitting or underfitting is a critical aspect of model fitting.

3. Strategies for Model Fitting:
a. Regularization: Adding regularization terms to the loss function helps prevent overfitting by penalizing overly complex models.
b. Cross-validation: Dividing the dataset into training and validation sets for assessing model performance and selecting the best model.
c. Hyperparameter tuning: Adjusting hyperparameters, such as learning rates and regularization strengths, to optimize model performance.
d. Ensemble methods: Combining predictions from multiple models to improve overall performance and reduce overfitting.

4. Handling Small Datasets:
a. Data Augmentation: Generating additional training examples by applying transformations like rotations, flips, and zooms to existing data, effectively increasing the dataset size.
b. Transfer Learning: Leveraging pre-trained models on larger datasets and fine-tuning them on the small dataset to adapt to the specific task.
c. Regularization: Using stronger regularization techniques to prevent overfitting given the limited data.

5. Handling Large Datasets:
a. Mini-batch Training: Dividing the large dataset into smaller, manageable subsets (mini-batches) to train the model incrementally, saving memory and speeding up training.
b. Parallel Processing: Exploiting distributed computing to train models simultaneously, accelerating training on large datasets.
c. Incremental Learning: Updating the model's parameters with new data in an incremental manner to adapt to evolving datasets.

#### Conclusion:
Understanding the nuances of model fitting and adopting appropriate strategies to handle small and large datasets are fundamental skills for machine learning practitioners. Adapting models to varying dataset sizes ensures robust performance and successful application of machine learning across a diverse range of real-world problems. By incorporating these practices, practitioners can enhance model efficiency, accuracy, and scalability, leading to more effective deployment and impact in various domains.

Examples described in 2.-project.ipynb


## 3. Understand the workings of different components of deep neural networks

Introduction:
Deep Neural Networks (DNNs) are the backbone of modern artificial intelligence, enabling machines to perform complex tasks, from image and speech recognition to natural language processing. A DNN comprises several fundamental components, each crucial for its functionality and effectiveness. Understanding these components is essential for anyone looking to delve into the realm of deep learning. This guide aims to elucidate the workings of different components within DNNs.

#### 1. Input Layer:
The input layer is the initial component of a DNN, responsible for receiving and encoding the raw data that the network will process. The number of nodes in the input layer is determined by the features or dimensions of the input data. Each node corresponds to a feature, and the values at these nodes represent the input data.

2. Hidden Layers:
Hidden layers are the intermediate layers between the input and output layers. Each hidden layer comprises multiple neurons (nodes) that perform complex computations. Each neuron receives inputs from the previous layer, applies an activation function, and passes the result to the next layer. Hidden layers allow the network to learn hierarchical features and patterns from the data.

3. Neurons (Nodes):
Neurons are the basic computation units within a neural network. They receive inputs, apply a weighted sum and an activation function, and produce an output that is passed to the next layer. Neurons are organized into layers, and each neuron's output becomes an input for the neurons in the subsequent layer.

4. Weights and Biases:
Weights and biases are parameters associated with each connection and neuron in the network. Weights determine the strength of connections between neurons, influencing the importance of specific inputs. Biases provide neurons with additional flexibility by allowing them to activate even when the weighted sum of inputs is zero.

5. Activation Functions:
Activation functions introduce non-linearity into the neural network, enabling it to approximate complex functions. Common activation functions include ReLU (Rectified Linear Unit), Sigmoid, Tanh, and Softmax. Each activation function serves a unique purpose, aiding in gradient flow and enhancing the network's capacity to model intricate relationships.

6. Output Layer:
The output layer provides the network's predictions or responses based on the learned features and patterns. The number of nodes in the output layer is determined by the nature of the task, such as binary classification, multi-class classification, or regression. The activation function in the output layer is tailored to suit the specific task requirements.

7. Loss Function:
The loss function quantifies the disparity between the predicted output and the true target. It acts as a guide for the network during training, helping it adjust its parameters to minimize the difference between predicted and actual values. Common loss functions include Mean Squared Error (MSE), Binary Cross-Entropy, and Categorical Cross-Entropy.

8. Backpropagation:
Backpropagation is a fundamental algorithm used to optimize the network's parameters during training. It involves computing gradients of the loss function with respect to the network's parameters, allowing efficient updates through techniques like gradient descent. Backpropagation ensures that the network learns from its mistakes and improves its predictions over time.

#### Conclusion:
Understanding the intricate workings of different components within deep neural networks is essential for effectively designing, training, and utilizing these powerful tools. Each component plays a vital role in shaping the network's ability to learn and generalize from data. Armed with this knowledge, practitioners can develop more robust and efficient neural network architectures for a wide array of applications in artificial intelligence and machine learning.

## 4. Design deep neural networks for real-world applications in computer vision
Designing deep neural networks for real-world applications in computer vision is a multifaceted and complex task that requires a combination of domain knowledge, understanding of neural network architectures, data preprocessing, model evaluation, and fine-tuning. Computer vision applications aim to enable machines to interpret and understand visual data, such as images and videos, and make informed decisions based on that understanding. Below is a comprehensive guide to designing effective deep neural networks for computer vision applications:

1. Understanding the Problem and Data:
Problem Understanding: Begin by thoroughly understanding the computer vision problem you want to solve. Identify the specific task, such as image classification, object detection, segmentation, or image generation.

Data Collection and Exploration: Collect a diverse and representative dataset for the problem. Explore the data to understand its characteristics, such as data distribution, class imbalance, and the types of features present in the images.

2. Preprocessing and Augmentation:
Data Preprocessing: Prepare the data for the model by applying necessary preprocessing steps such as resizing, normalization, and handling missing values. Preprocessing ensures that the data is in a suitable format for the neural network.

Data Augmentation: Augment the data to increase its diversity and improve model generalization. Techniques like rotation, flipping, scaling, and color adjustments can be employed to augment the dataset.

3. Model Architecture Selection:
Selecting Neural Network Architecture: Choose an appropriate neural network architecture based on the problem. Convolutional Neural Networks (CNNs) are commonly used for computer vision tasks due to their ability to capture spatial patterns.

Customizing Architectures: Tailor the selected architecture to match the complexity and requirements of the problem. Adjust the depth, width, and other parameters of the network based on the available computational resources and desired accuracy.

4. Model Training and Optimization:
Loss Function and Optimizer Selection: Choose an appropriate loss function that aligns with the problem (e.g., categorical cross-entropy for classification). Select an optimizer (e.g., Adam, SGD) to minimize the loss function during training.

Hyperparameter Tuning: Experiment with different learning rates, batch sizes, and regularization techniques to optimize model performance. Use techniques like learning rate schedules and early stopping to prevent overfitting and speed up convergence.

5. Evaluation and Fine-Tuning:
Model Evaluation Metrics: Define evaluation metrics relevant to the problem, such as accuracy, precision, recall, F1-score, or Mean Intersection over Union (mIoU).

Model Evaluation: Evaluate the model on a separate validation set or using cross-validation. Analyze the metrics to understand the model's performance and areas for improvement.

Fine-Tuning and Iteration: Based on the evaluation, fine-tune the model by adjusting hyperparameters or modifying the architecture. Iterate this process to achieve the desired level of performance.

6. Deployment and Integration:
Model Deployment: Once the model is trained and optimized, deploy it to a production environment. Optimize the model for inference speed and resource efficiency.

Integration with Applications: Integrate the trained model into the target application or system. Ensure compatibility and performance in the intended use case.

7. Continuous Monitoring and Improvement:
Performance Monitoring: Continuously monitor the model's performance in the production environment. If necessary, retrain or fine-tune the model with new data to adapt to changing requirements.

Feedback Loop and Iteration: Establish a feedback loop to gather user feedback and data from the deployed application. Use this feedback to iterate and improve the model continuously.

Conclusion:
Designing deep neural networks for real-world computer vision applications involves a systematic approach, encompassing problem understanding, data preprocessing, architecture selection, training, evaluation, deployment, and ongoing monitoring. By following this comprehensive guide and staying updated with advancements in the field, AI practitioners can effectively develop and deploy robust computer vision models for various applications.

Examples described in 4.-project.ipynb

## 5. Knowledge Transfer in Neural Networks: Enhancing Models with Limited Data

#### Introduction:
Knowledge transfer in neural networks, often referred to as transfer learning, is a powerful technique that addresses the challenge of training effective models with limited data. In various real-world scenarios, collecting an extensive labeled dataset can be costly, time-consuming, or sometimes unfeasible. Transfer learning leverages knowledge gained from pre-trained models and applies it to new, related tasks, boosting model performance and generalization even when data is scarce. This comprehensive guide will explore the principles and methodologies of knowledge transfer in neural networks to develop robust models for applications with limited data.

1. Understanding Transfer Learning:
Transfer learning involves utilizing knowledge from a pre-trained neural network, often on a large dataset and a related task, and applying this knowledge to a different but related task with a smaller dataset. The knowledge is transferred in the form of learned weights, architectures, or representations from layers of the pre-trained model.

2. Pre-trained Models and Architectures:
Pre-trained models, such as VGG, ResNet, Inception, and BERT, have been trained on extensive datasets like ImageNet or massive amounts of text. These models have learned complex features and patterns from the data, making them invaluable for transfer learning. The earlier layers learn generic features, while the deeper layers learn more task-specific features.

3. Transfer Learning Approaches:
a. Feature Extraction: The pre-trained model's weights are frozen, and the output layers are replaced or augmented to match the new task. The existing features are used, and only the final layers are retrained on the limited dataset.

b. Fine-Tuning: In this approach, a few top layers of the pre-trained model are unfrozen and retrained on the new dataset. Lower layers, responsible for generic features, are often kept frozen to preserve pre-learned representations.

4. Choosing the Right Approach:
The choice between feature extraction and fine-tuning depends on the size and similarity of the dataset to the pre-training data, as well as the complexity of the new task. Feature extraction is preferred when the dataset is small, and the pre-trained model's features are sufficient. Fine-tuning is suitable when the dataset is larger, and the task is more specific.

5. Practical Steps for Knowledge Transfer:
a. Select a Pre-trained Model: Choose a pre-trained model based on the nature of your task—image classification, natural language processing, etc.

b. Adapt the Model Architecture: Modify the model's architecture according to your specific task, keeping the pre-trained layers as needed.

c. Fine-tune or Train the Model: Depending on the approach chosen, fine-tune the model's top layers or train the modified architecture on your limited dataset.

d. Evaluate and Iterate: Evaluate the model's performance on validation data, make necessary adjustments, and iterate the process if needed.

6. Real-life Applications and Success Stories:
Transfer learning has been widely successful across various domains. For instance, in medical imaging, pre-trained models have significantly improved disease detection even with limited annotated medical images. In natural language processing, transfer learning has revolutionized sentiment analysis, language translation, and more, especially for languages with fewer resources.

#### Conclusion:
Knowledge transfer in neural networks empowers practitioners to overcome data limitations and build effective models. Understanding transfer learning approaches, selecting appropriate pre-trained models, and adapting them to specific tasks are essential steps. By leveraging pre-trained knowledge and fine-tuning models, we can create accurate, reliable, and efficient solutions for diverse applications, even in the face of limited data.

## 6. Get introduced to deep learning approaches for unsupervised learning

Unsupervised learning is a branch of machine learning where the model learns patterns and structures from unlabeled data without explicit supervision. In other words, the model is not provided with labeled examples (input-output pairs) but is expected to find hidden patterns and relationships within the data on its own.

Deep learning, on the other hand, refers to the use of neural networks with multiple layers (hence the term "deep") to analyze and learn from data. Deep learning has gained immense popularity due to its ability to automatically extract complex features from raw data and achieve state-of-the-art performance in various domains.

When applying deep learning to unsupervised learning, there are several common approaches:

1. Autoencoders:
Autoencoders are neural networks designed to learn efficient representations of the input data. The network is trained to encode the input data into a compact latent space representation and then decode it back to the original form. By minimizing the reconstruction error (the difference between the input and the reconstructed output), the autoencoder learns meaningful features.

2. Variational Autoencoders (VAEs):
VAEs extend traditional autoencoders by introducing probabilistic modeling. They aim to learn a probabilistic mapping between the input data and a latent variable space. VAEs not only generate new data points similar to the training data but also allow for interpolation and manipulation of data in the latent space.

3. Generative Adversarial Networks (GANs):
GANs consist of two neural networks, a generator and a discriminator, which are trained simultaneously. The generator creates samples that aim to mimic the real data distribution, while the discriminator distinguishes between real and generated data. This adversarial training results in the generator producing increasingly realistic data.

4. Self-Supervised Learning:
Self-supervised learning is a learning paradigm where the model is trained to predict a part of the input data from another part. For instance, in language modeling tasks, a model might be trained to predict the next word in a sentence. This type of training implicitly involves learning useful features from the data.

5. Clustering Algorithms:
Although not strictly deep learning, clustering algorithms like K-means can be combined with neural networks to perform unsupervised learning tasks. The neural network can learn representations that are then used for clustering in the latent space.

6. Deep Belief Networks (DBNs):
DBNs are probabilistic, generative models that consist of multiple layers of stochastic, latent variables. They combine the power of restricted Boltzmann machines and use unsupervised learning for pre-training and fine-tuning with supervised learning.

7. Sparse Coding:
Sparse coding is a technique that aims to represent data as a linear combination of a few basis functions. Deep networks utilizing sparse coding principles can learn a hierarchical representation of the data.

In summary, deep learning approaches for unsupervised learning aim to discover hidden patterns, features, and representations in the data without relying on labeled examples. Autoencoders, VAEs, GANs, self-supervised learning, and other methods play a crucial role in unsupervised learning by extracting meaningful and high-level representations from the data, enabling various applications in machine learning and artificial intelligence.

Examples described in 6.-project.ipynb

## 7. Transformers, Self-Attention, and Large Language Models (LLMs)

- Overview
Transformers are a type of deep learning model that has gained significant attention in the field of natural language processing (NLP) due to their ability to handle long-range dependencies and capture contextual information effectively. The fundamental building block of Transformers is the self-attention mechanism, which enables the model to weigh different words in a sentence differently during processing. Language Models (LMs) based on Transformers, often referred to as Language Models with self-attention, have shown state-of-the-art performance in various NLP tasks.

- Self-Attention Mechanism
The self-attention mechanism is a mechanism that weighs the importance of each word in a sentence based on its relationship with all other words in the sentence. This mechanism allows the model to give higher importance to certain words depending on the context. The self-attention score for a word in a sentence is calculated using the following equation:

![Alt text](image-1.png)

​
The softmax function is applied row-wise, producing a probability distribution over the words for each word in the sentence.

- Transformers Architecture
The Transformers architecture consists of several layers of self-attention and feedforward neural networks. Each layer processes the input in parallel and consists of the following sub-modules:

1. Multi-Head Self-Attention: This module involves multiple sets of learnable query, key, and value matrices, allowing the model to attend to different parts of the input in parallel.
2. Positional Encoding: Since Transformers don't inherently understand the order of words in a sentence, positional encodings are added to give the model information about the positions of words in the input sequence.
3. Feedforward Neural Networks: These networks are applied independently to each position.
The output of each sub-module is combined using concatenation and linear transformations to produce the output of the layer.

- Language Models with Self-Attention
Language Models based on Transformers utilize the self-attention mechanism to predict the next word in a sequence given the preceding context. These models are trained on large text corpora and learn to generate coherent and contextually appropriate text. The training objective typically involves maximizing the likelihood of the next word in a sentence.

Examples and architecture described in 7.-project.ipynb




