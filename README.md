# SVM (Support Vector Machine)

Support Vector Machine (SVM) is a supervised machine learning algorithm that can be used for both regression and classification challenges. However, it is mostly used in classification problems.

![image](https://github.com/TITHI-KHAN/Support-Vector-Machine-SVM/assets/65033964/7b80bc49-167b-4649-8677-b125775501f2)

Here, this is a Linearly Separable data. 

![image](https://github.com/TITHI-KHAN/Support-Vector-Machine-SVM/assets/65033964/619bc884-573c-4ef4-a9ae-aec045c59822)

![image](https://github.com/TITHI-KHAN/Support-Vector-Machine-SVM/assets/65033964/2e1394ee-1a36-4a18-a115-e5f5fa964b37)

![image](https://github.com/TITHI-KHAN/Support-Vector-Machine-SVM/assets/65033964/2a6898e0-2004-4707-8791-58ee01b318c4)

Hyper Plane -> It goes through the 2 classes.

![image](https://github.com/TITHI-KHAN/Support-Vector-Machine-SVM/assets/65033964/a4649ac7-ac33-44f9-a95f-a320d1101a19)

![image](https://github.com/TITHI-KHAN/Support-Vector-Machine-SVM/assets/65033964/2370f2f7-ae66-40f6-aa11-9d30c9d43758)

Margin -> The two lines go through the two sides of the Hyper Plane (D1 & D2).

We need the margin to take the Hyper Plane accurately. 

![image](https://github.com/TITHI-KHAN/Support-Vector-Machine-SVM/assets/65033964/d4523ff7-c5b6-46ee-8773-2974145600de)

![image](https://github.com/TITHI-KHAN/Support-Vector-Machine-SVM/assets/65033964/1b614888-38bd-4966-ab8c-1f773f74a4f1)

The more the distance of inter class is, the better. The less the distance of intra class is, the better.

The more the Marginal Distance is, that Hyper Parameter is the best. (Goal)

In general cases, marginal distance doesn't enter into the data.

**If the features contain numerical values, then use SVM.**


# The Role of Margins in SVMs

Sometimes, the data is linearly separable, but the margin is so small that the model becomes prone to overfitting or being too sensitive to outliers. Also, in this case, we can opt for a larger margin by using soft margin SVM in order to help the model generalize better.

# Hard vs Soft Margin

• When the data is linearly separable, and we don’t want to have any misclassifications, we use SVM with a hard margin.

• When a linear boundary is not feasible, or we want to allow some misclassifications in the hope of achieving better generality, we can opt for a soft margin for our classifier.

![image](https://github.com/TITHI-KHAN/Support-Vector-Machine-SVM/assets/65033964/9188d9b9-fb16-4d45-8ca5-c32a2be4936a)

In Hard Margin, if the margin is less, then there is no issue. It does not draw the decision boundary in a way where a margin clashes with another margin. 

In Soft Margin, there can be anomalies. It is used for better generalization and better classification. But, it is not applicable for all data.

It works to keep the Hyper Plane in a better place so that we can classify better. 

The more the marginal distance is, the better. Because, it reduces the chance of misclassification. 

**Let's explain the concepts of hard margin and soft margin in SVM along with figures:**

**1. Hard Margin SVM**:

Hard margin SVM is an approach in which the algorithm tries to find a hyperplane that perfectly separates the two classes with no data points falling within the margin. In other words, the decision boundary must not allow any misclassifications, and all data points of one class should be on one side of the hyperplane, and all data points of the other class should be on the other side.

However, hard margin SVM has limitations and can be sensitive to outliers. If the data is not perfectly separable, or if there are outliers that cannot be separated by a hyperplane, hard margin SVM may fail to find a feasible solution. The figure below illustrates a hard margin SVM with perfectly separable data:

![Hard Margin SVM](https://upload.wikimedia.org/wikipedia/commons/thumb/2/2a/Svm_max_sep_hyperplane_with_margin.png/600px-Svm_max_sep_hyperplane_with_margin.png)

**2. Soft Margin SVM:**

To overcome the limitations of hard margin SVM, soft margin SVM was introduced. Soft margin SVM allows some misclassifications and violations of the margin to achieve a balance between maximizing the margin and minimizing the misclassifications. This approach introduces a penalty for misclassified data points and allows a certain amount of tolerance, or "slack," for data points to fall within the margin.

The soft margin SVM is more robust to noisy or overlapping data, making it more practical for real-world applications. The figure below illustrates a soft margin SVM with a non-linearly separable dataset and a margin that allows some misclassifications:

![Soft Margin SVM](https://upload.wikimedia.org/wikipedia/commons/thumb/7/72/SVM_margin.png/600px-SVM_margin.png)

In this figure, the dotted lines indicate the margin, and the filled circles represent support vectors. Some data points are allowed to be within the margin or on the wrong side of the hyperplane while still finding a good decision boundary.

The choice between hard margin and soft margin SVM depends on the nature of the data and the trade-off between maximizing the margin and allowing some misclassifications. In practice, soft margin SVM is commonly used because it can handle more complex datasets and is less sensitive to outliers.

**Why do We Need Hard Margin and Soft Margin?**

We need both hard margin and soft margin SVM to address different types of datasets and classification problems effectively. The choice between hard and soft margin depends on the characteristics of the data and the trade-offs between maximizing the margin and allowing some misclassifications. **Let's explore the reasons why both hard and soft margin SVM are important:**

**1. Hard Margin SVM:**

- **Perfectly Separable Data**:

Hard margin SVM is suitable when the data is perfectly separable with a clear margin between the two classes. In such cases, hard margin SVM can find the optimal hyperplane that perfectly separates the classes without allowing any misclassifications. It provides a straightforward decision boundary and ensures no overlap between the classes.

- **Theoretical Understanding**:

Hard margin SVM serves as a foundation for understanding SVM. It helps in understanding the concept of finding the maximum margin hyperplane and forms the basis for further exploration of soft margin SVM.

- **Simplification**:

In some cases, when the data is perfectly separable, using hard margin SVM simplifies the optimization problem, leading to faster convergence and simpler models.

**2. Soft Margin SVM:**

- **Overlapping or Noisy Data**:

In real-world datasets, data points might not be perfectly separable due to overlapping classes or the presence of noisy data points. In such situations, soft margin SVM is more appropriate, as it allows a certain number of misclassifications and permits some data points to fall within the margin.

- **Robustness**:

Soft margin SVM is more robust to outliers and noisy data since it allows some flexibility in the decision boundary. It prevents overfitting and generalizes better to unseen data.

- **Practicality**:

In many real-world applications, it is rare to find perfectly separable data. Soft margin SVM is more practical as it can handle data with varying degrees of separability, making it more widely applicable.

- **Trade-off between Margin and Misclassification**s:

Soft margin SVM introduces a trade-off between maximizing the margin and allowing some misclassifications. This trade-off can be adjusted using the regularization parameter (C) to control the balance between the two objectives.

**In summary**, hard margin SVM is useful when the data is linearly separable with no noise or outliers. On the other hand, soft margin SVM is more versatile and practical, as it can handle complex datasets with overlapping classes and noisy data. It provides a more flexible decision boundary, making it a preferred choice in most real-world classification tasks.

# Linearly Separable and Non-Linearly Separable / Linearly Inseparable

![image](https://github.com/TITHI-KHAN/Support-Vector-Machine-SVM/assets/65033964/bd1a44de-05ed-4124-b1a9-89c6696d4cd3)

![image](https://github.com/TITHI-KHAN/Support-Vector-Machine-SVM/assets/65033964/df5954a7-5bb9-4b63-8c70-1b39f9ee6fe4)

![image](https://github.com/TITHI-KHAN/Support-Vector-Machine-SVM/assets/65033964/f3aef505-c761-4a7e-b25e-d624191029c5)

If the data is linearly separable, use Hard Margin. If it does not provide good result, then use Soft Margin.

The real-life data are hard to classify better. So, by default, it uses Soft Margin. Here, we consider C=1 (defualt) to classify.

**If the data is inseparable (this type of data are more), then use Kernel Trick.**

If there are two features, then we will have to increase the dimension from 2D to higher dimension. Suppose, we take 2D to 3D to classify better.

When we are taking low dimension to higher dimension, then the features of the data gets increased.

**Let's discuss the concepts of linearly separable, linear inseparable (non-linear separable), and non-linearly separable data with figures:**

**1. Linearly Separable Data:**

Linearly separable data is a scenario in which two classes of data points can be perfectly separated by a straight line (in 2D) or a hyperplane (in higher dimensions). The decision boundary can be drawn without any misclassifications, and all data points of one class are on one side of the line/hyperplane, while all data points of the other class are on the other side. The figure below shows an example of linearly separable data:

![Linearly Separable Data](https://upload.wikimedia.org/wikipedia/commons/thumb/2/20/Svm_separating_hyperplanes.png/600px-Svm_separating_hyperplanes.png)

**2. Linear Inseparable (Non-linear Separable) Data:**

Linear inseparable data, also known as non-linearly separable data, is a scenario in which no straight line (in 2D) or hyperplane (in higher dimensions) can separate the two classes perfectly. The data points of both classes are mixed together, making it impossible to draw a single linear decision boundary. In such cases, a linear classifier like the standard linear SVM cannot accurately classify the data. The figure below illustrates an example of linear inseparable data:

![image](https://github.com/TITHI-KHAN/Support-Vector-Machine-SVM/assets/65033964/5a814d24-cd2c-41a6-b8be-49d2b88e261a)

**3. Non-linearly Separable Data:**

Non-linearly separable data is a scenario where the classes can be separated by a non-linear decision boundary. In this case, the data points of different classes are not linearly separable, but they can be separated by curves or more complex decision boundaries. To handle non-linearly separable data, kernel methods, such as the polynomial kernel or radial basis function (RBF) kernel, are used with SVM. These kernels transform the original feature space into a higher-dimensional space, where linear separation becomes possible. The figure below depicts an example of non-linearly separable data:

![Non-linearly Separable Data](https://upload.wikimedia.org/wikipedia/commons/thumb/f/fe/Kernel_Machine.png/600px-Kernel_Machine.png)

In this example, the data points of different classes are not separable by a straight line, but when transformed into a higher-dimensional space using a kernel function, a non-linear decision boundary (represented by a curved line) can effectively separate the classes.

In summary, linearly separable data can be classified using a simple linear decision boundary, while linear inseparable (non-linear separable) data requires the use of more complex decision boundaries, which can be achieved through kernel methods with SVM.

# SVM Kernels

▪ ‘linear’,

▪ ‘poly’,

▪ ‘rbf’,

▪ ‘sigmoid’,

▪ ‘precomputed’

Default : ‘rbf’

Let's discuss the different SVM kernels with some visual representations:

1. **Linear Kernel**:

The linear kernel is the simplest and most straightforward kernel function. It creates a straight hyperplane to separate the data points of different classes. In two-dimensional space, the hyperplane is a straight line, and in higher dimensions, it becomes a hyperplane. The linear kernel works well when the data is linearly separable, as shown in the figure below:

![Linear Kernel](https://upload.wikimedia.org/wikipedia/commons/thumb/7/72/SVM_margin.png/800px-SVM_margin.png)

2. **Polynomial Kernel**:

The polynomial kernel maps the data points into a higher-dimensional space using a polynomial function. It is useful when the data has curved or non-linear decision boundaries. The degree of the polynomial can be adjusted to control the complexity of the decision boundary. Here's an example of a polynomial kernel with a degree of 3:

![Polynomial Kernel](https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/Kernel_Machine.png/800px-Kernel_Machine.png)

Here, 

![image](https://github.com/TITHI-KHAN/Support-Vector-Machine-SVM/assets/65033964/439178cf-e2b2-43bc-8ca0-dcba8ac25f27)

x = f = Feature 

Feature = F1, F2 **(Initially) (2D)**

y = Function of F1 & F2 = f(F1, F2)

![image](https://github.com/TITHI-KHAN/Support-Vector-Machine-SVM/assets/65033964/da85288b-5b0f-4452-8df9-f67c6bcb50eb)

![image](https://github.com/TITHI-KHAN/Support-Vector-Machine-SVM/assets/65033964/2066978c-d746-4d81-958f-839a1bbc14d4)

**Now, 5D** (Features have increased due to increasing dimension).

**When you are taking lower dimension to higher dimension, then use Kernel Trick.**

From 2D space, we are taking it to higher dimension.

Here, 1 -> Identity Matrix

f -> Feature Matrix

d -> Dimension

From f1^2 and f2^2, we will take one feature. No need to keep similar type of 2 features (f1f2).

3. **Radial Basis Function (RBF) Kernel**:

The RBF kernel is one of the most popular and widely used kernel functions. It creates a circular decision boundary around each data point and combines them to create a non-linear decision boundary. The RBF kernel is very flexible and can handle complex data distributions effectively. The figure below shows how the RBF kernel captures non-linear decision boundaries:

![RBF Kernel](https://upload.wikimedia.org/wikipedia/commons/thumb/9/97/Kernel_trick_idea.svg/800px-Kernel_trick_idea.svg.png)

4. **Sigmoid Kernel**:

The sigmoid kernel maps the data points into a non-linear space using the sigmoid function. It is often used in neural networks but is less common in SVMs. The sigmoid kernel can be useful when dealing with binary classification problems. However, it is sensitive to feature scaling and can be less effective compared to other kernels.

5. **Precomputed Kernel**:

The precomputed kernel allows users to provide a precomputed kernel matrix instead of passing the original data to the SVM algorithm. This can be beneficial when dealing with specialized kernels or when the kernel matrix is already computed from other sources.

Remember that the choice of the SVM kernel depends on the specific characteristics of the data and the problem at hand. It is essential to experiment with different kernels and their parameters to find the best fit for your particular task.

**How does Kernel Trick work in SVM?**

The **kernel trick** is a powerful technique used in Support Vector Machines (SVM) to handle non-linearly separable data by implicitly mapping the data into a higher-dimensional feature space. It enables SVM to find a non-linear decision boundary in this transformed space, even though the algorithm operates in the original input space. This approach allows SVM to handle complex data distributions without explicitly computing the transformations, making it computationally efficient.

**Let's go through an example of how the kernel trick works in SVM with the help of a figure:**

Consider a simple 2D dataset with two classes (red and blue points) that are not linearly separable in the input space:

![image](https://github.com/TITHI-KHAN/Support-Vector-Machine-SVM/assets/65033964/8e022736-1eae-4a4f-97c8-be5de87cab25)

In the 2D input space, the data points look like this:

![image](https://github.com/TITHI-KHAN/Support-Vector-Machine-SVM/assets/65033964/7e2708b1-5c48-4f5e-8341-176f289d31ba)

Since the data is not linearly separable, a linear SVM cannot find a single straight line to separate the two classes effectively.

Now, let's apply the kernel trick to transform the data points into a higher-dimensional space using a polynomial kernel of degree 2:

![image](https://github.com/TITHI-KHAN/Support-Vector-Machine-SVM/assets/65033964/86038418-1dd9-4145-b7ea-61bc1c629c69)

where `x` and `y` are the original 2D input feature vectors.

By computing the kernel function for all pairs of data points, we get the following transformed data points in the higher-dimensional space:

![image](https://github.com/TITHI-KHAN/Support-Vector-Machine-SVM/assets/65033964/6f951bc3-99c4-4a8d-8f4a-906186d362b7)

Now, the transformed data looks like this:

![image](https://github.com/TITHI-KHAN/Support-Vector-Machine-SVM/assets/65033964/501d3f6e-771c-4c9b-9391-d6cf1f2d08b6)

In this higher-dimensional space, the data points become linearly separable. A linear SVM can now find a hyperplane to separate the classes effectively. When the decision boundary is projected back into the original 2D input space, it appears as a non-linear boundary, as shown in the figure below:

![Kernel Trick in SVM](https://upload.wikimedia.org/wikipedia/commons/1/1b/Kernel_Machine.png)

The curved decision boundary in the higher-dimensional space corresponds to a non-linear decision boundary in the original 2D input space. This is the essence of the kernel trick in SVM, allowing the algorithm to handle non-linearly separable data by implicitly operating in a higher-dimensional space without explicitly transforming the data points.

The choice of the kernel function (e.g., polynomial, radial basis function, etc.) and its parameters plays a crucial role in the effectiveness of the SVM for non-linear classification tasks. The kernel trick enables SVM to work with complex data distributions, making it a widely used algorithm in various machine learning applications.

# Hyperparameter Optimization in SVM

Hyperparameter optimization in SVM is crucial for finding the best set of hyperparameters that result in the most accurate and well-generalized model. Two important hyperparameters in SVM are the regularization parameter (C) and the kernel parameters (e.g., degree for polynomial kernel or gamma for RBF kernel). Hyperparameter optimization techniques, such as Grid Search and Random Search, help us find the optimal hyperparameters efficiently.

**Let's go through an example of hyperparameter optimization in SVM using Grid Search and visualize the results with a figure:**

Consider a 2D dataset with two classes, "red" and "blue," that are not linearly separable:

![image](https://github.com/TITHI-KHAN/Support-Vector-Machine-SVM/assets/65033964/8dbb1752-2f80-48a2-bd18-a054763a8fee)

In the 2D input space, the data points would look like this:

![image](https://github.com/TITHI-KHAN/Support-Vector-Machine-SVM/assets/65033964/c8bc7ef1-84c4-46f0-9368-781f5f500317)

We want to use an SVM with an RBF kernel for this example. The SVM has two hyperparameters to optimize: the regularization parameter (C) and the kernel parameter (gamma).

1. **Hyperparameter Optimization using Grid Search**:
   
Grid Search is a technique that exhaustively searches through a predefined range of hyperparameters and evaluates the model's performance using cross-validation to find the best combination.

Let's assume the following ranges for hyperparameters:
- C: [0.1, 1, 10, 100]
- gamma: [0.01, 0.1, 1, 10]

Grid Search will try all combinations of C and gamma from the given ranges and evaluate the model's performance using cross-validation. It will select the combination that gives the best performance.

1. **Visualization of Decision Boundaries**:
   
For each combination of hyperparameters, we can train an SVM on the dataset and visualize the decision boundary to understand how the model is performing.

Below is a figure that shows the decision boundaries for different combinations of C and gamma:

```
C=0.1, gamma=0.01  |   C=1, gamma=0.01   |   C=10, gamma=0.01  |   C=100, gamma=0.01
------------------|-------------------|-------------------|--------------------
C=0.1, gamma=0.1   |   C=1, gamma=0.1    |   C=10, gamma=0.1   |   C=100, gamma=0.1
------------------|-------------------|-------------------|--------------------
C=0.1, gamma=1     |   C=1, gamma=1     |   C=10, gamma=1     |   C=100, gamma=1
------------------|-------------------|-------------------|--------------------
C=0.1, gamma=10    |   C=1, gamma=10    |   C=10, gamma=10    |   C=100, gamma=10
```

In the figure above, each cell represents the decision boundary obtained using a specific combination of C and gamma. The colors represent the predicted classes, and the shaded regions indicate the decision boundary. The best combination of C and gamma will result in a decision boundary that separates the classes effectively and minimizes misclassifications.

After running Grid Search, the optimal hyperparameters (C and gamma) can be determined based on the best performance metrics, such as accuracy, F1 score, or cross-validation score.

Hyperparameter optimization is essential to improve the SVM model's performance and ensure it generalizes well to new, unseen data. The visualization of decision boundaries helps us understand how different combinations of hyperparameters affect the model's performance and aids in selecting the best hyperparameters for the task at hand.

**Besides Grid Search, there are several other hyperparameter optimization techniques** used to find the best combination of hyperparameters for Support Vector Machines (SVM) or other machine learning models. Some popular alternative techniques include:

1. **Random Search**:
   
Random Search is a hyperparameter optimization technique that randomly samples hyperparameter values from predefined ranges. It doesn't explore all possible combinations like Grid Search, but it can be more efficient and effective, especially when the number of hyperparameter combinations is large. Random Search provides a good trade-off between optimization performance and computational cost.

2. **Bayesian Optimization**:
   
Bayesian Optimization is a probabilistic model-based optimization technique that uses a surrogate model to predict the performance of different hyperparameter configurations. It iteratively updates the surrogate model based on the evaluation results and focuses the search on promising areas of the hyperparameter space. Bayesian Optimization is useful when evaluating hyperparameter configurations is computationally expensive.

3. **Genetic Algorithms**:
   
Genetic Algorithms are inspired by the process of natural selection and evolution. In hyperparameter optimization, they create a population of possible hyperparameter configurations and evolve them over generations, selecting the best-performing configurations for the next iterations. Genetic Algorithms can be effective for complex optimization problems but might require more computational resources.

4. **Particle Swarm Optimization (PSO)**:
 
PSO is a population-based optimization technique inspired by the social behavior of birds flocking or fish schooling. Each potential solution (a particle) moves through the search space based on its own experience and the experiences of its neighbors. The algorithm seeks to find the best solution by iteratively updating the positions of particles. PSO can be efficient and useful in high-dimensional search spaces.

5. **Simulated Annealing**:

Simulated Annealing is inspired by the annealing process in metallurgy, where metals are cooled slowly to reduce defects in their crystal structures. In hyperparameter optimization, it starts with an initial configuration and probabilistically accepts worse configurations at the beginning (like high temperatures) and gradually reduces the probability of accepting worse configurations over time (like cooling down). Simulated Annealing can effectively escape local optima and explore the search space.

Each hyperparameter optimization technique has its strengths and weaknesses, and the choice of the method depends on factors like the size of the search space, the number of hyperparameters, and the available computational resources. It's common to try different techniques and compare their performance to select the most suitable one for a particular problem.
