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

**Hard vs Soft Margin:**

• When the data is linearly separable, and we don’t want to have any misclassifications, we use SVM with a hard margin.

• When a linear boundary is not feasible, or we want to allow some misclassifications in the hope of achieving better generality, we can opt for a soft margin for our classifier.

![image](https://github.com/TITHI-KHAN/Support-Vector-Machine-SVM/assets/65033964/9188d9b9-fb16-4d45-8ca5-c32a2be4936a)

In Hard Margin, if the margin is less, then there is no issue. It does not draw the decision boundary in a way where a margin clashes with another margin. 

In Soft Margin, there can be anomalies. It is used for better generalization and better classification. But, it is not applicable for all data.

It works to keep the Hyper Plane in a better place so that we can classify better. 

The more the marginal distance is, the better. Because, it reduces the chance of misclassification. 

**Linearly Separable and Non-Linearly Separable / Linearly Inseparable:**

![image](https://github.com/TITHI-KHAN/Support-Vector-Machine-SVM/assets/65033964/bd1a44de-05ed-4124-b1a9-89c6696d4cd3)

![image](https://github.com/TITHI-KHAN/Support-Vector-Machine-SVM/assets/65033964/df5954a7-5bb9-4b63-8c70-1b39f9ee6fe4)

![image](https://github.com/TITHI-KHAN/Support-Vector-Machine-SVM/assets/65033964/f3aef505-c761-4a7e-b25e-d624191029c5)

If the data is linearly separable, use Hard Margin. If it does not provide good result, then use Soft Margin.

The real-life data are hard to classify better. So, by default, it uses Soft Margin. Here, we consider C=1 (defualt) to classify.

**If the data is inseparable (this type of data are more), then use Kernel Trick.**

If there are two features, then we will have to increase the dimension from 2D to higher dimension. Suppose, we take 2D to 3D to classify better.

When we are taking low dimension to higher dimension, then the features of the data gets increased.

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

Feature = F1, F2

y = Function of F1 & F2 = f(F1, F2)

![image](https://github.com/TITHI-KHAN/Support-Vector-Machine-SVM/assets/65033964/da85288b-5b0f-4452-8df9-f67c6bcb50eb)

![image](https://github.com/TITHI-KHAN/Support-Vector-Machine-SVM/assets/65033964/2066978c-d746-4d81-958f-839a1bbc14d4)

Here, 

3. **Radial Basis Function (RBF) Kernel**:

The RBF kernel is one of the most popular and widely used kernel functions. It creates a circular decision boundary around each data point and combines them to create a non-linear decision boundary. The RBF kernel is very flexible and can handle complex data distributions effectively. The figure below shows how the RBF kernel captures non-linear decision boundaries:

![RBF Kernel](https://upload.wikimedia.org/wikipedia/commons/thumb/9/97/Kernel_trick_idea.svg/800px-Kernel_trick_idea.svg.png)

4. **Sigmoid Kernel**:

The sigmoid kernel maps the data points into a non-linear space using the sigmoid function. It is often used in neural networks but is less common in SVMs. The sigmoid kernel can be useful when dealing with binary classification problems. However, it is sensitive to feature scaling and can be less effective compared to other kernels.

5. **Precomputed Kernel**:

The precomputed kernel allows users to provide a precomputed kernel matrix instead of passing the original data to the SVM algorithm. This can be beneficial when dealing with specialized kernels or when the kernel matrix is already computed from other sources.

Remember that the choice of the SVM kernel depends on the specific characteristics of the data and the problem at hand. It is essential to experiment with different kernels and their parameters to find the best fit for your particular task.


