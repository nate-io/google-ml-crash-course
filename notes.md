# Course Notes

[TOC]

## Intro

ML is different from software engineering in that SE focuses on logic whereas ML engineering is 'based on making assertions about an uncertain world' and, hence, more akin to a natural science which requires the scientific method.

## Framing
(Supervised) ML systems learn how to combine input to *produce useful predictions on never-before-seen data*. [Link](https://developers.google.com/machine-learning/crash-course/framing/ml-terminology) Data must be quantifiable to be useful.

1. **Label** is the variable we're trying to predict 
   * typically represented by variable ```y```
   * ex: spam or not spam
2. **Features** are input variables describing our data
   * typically represented by variables `{ x₁, x₂... }`
   * ex: when email was sent, word in email body, sender's address, etc.
3. **Example**: a particular instance of data, `x`
   1. **Labeled** example has `{ features, label }: (x, y)`
      * used to train model
   2. **Unlabeled** example has `{ features, ? }: (x, ?)`
      * used for making predictions on new data
4. **Model** maps examples to predicted labels: `y'`
   * defined by internal params, which are learned
   * **Training/learning**: means creating the model by showing the model labeled examples which trains the model to learn relationships between features & label.
   * **Inference**: applying the model to unlabeled examples
5. **Regression**: model provides continuous values
   * what is value of a house in California?
6. **Classification**: model predicts discrete values
   * is an image of a dog, a cat, or a hamster?

## Descending Into ML

1. **Linear regression**: method for finding the straight line (hyperplane) that best fits a set of points. For simple models thinks of a line charting the general behavior of the inputs which uses equation  `y = mx + b`  to determine a prediction.

2. Algebra uses `y = mx + b` to determine slope of a line where

   * y = value we're attemping to find
   * m = slope of the line
   * x = value of the input
   * b = y-axis intercept

3. ML uses the form `y' = b + w₁x₁` where

   * y' = is the predicted label
   * b = is the bias 
   * w₁ = the weight of the input feature
   * x₁ = the feature istelf (remembering spam, say, the email subject line)
   * The above is a simple model; more complex models are defined by adding more features each with their own weights:

   * y' = b + w~1~x~1~ + w~2~x~2~ + w~3~x~3~ 

4. **Training** then simply means learning good values for all the weights and the bias from labeled examples. In supervised learning, a machine learning algorithm builds a model by examining many examples and attempting to find a model that minimizes loss; this process is called **empirical risk minimization**.

5. Loss is the penalty for a bad prediction. That is, **loss** is a number indicating how bad the model's prediction was on a single example. If the model's prediction is perfect, the loss is zero; otherwise, the loss is greater. The goal of training a model is to find a set of weights and biases that have *low* loss, on average, across all examples. Reducing loss, then, is a technique to improve the model.

6. The linear regression models we'll examine here use a loss function called **squared loss** (also known as **L~2~ loss**). When graphed this function is known as a **loss curve**. The squared loss for a single example is as follows:

   ```
   = the square of the difference between the label and the prediction  
   = (observation - prediction(**x**))^2 
   = (y - y')^2
   ```

7. **Mean square error** (**MSE**) is the average squared loss per example over the whole dataset. 

## Reducing Loss

![The cycle of moving from features and labels to models and predictions.](https://developers.google.com/machine-learning/crash-course/images/GradientDescentDiagram.svg)

1. Reducing loss improves the predictions of the model. One approach to reducing loss is via an iterative approach wherein you adjust the feature weights & bias one step at a time, calculate your loss, and adjust the weights in an attempt to find the minimum loss value point. _Iterative strategies are prevalent in machine learning, primarily because they scale so well to large data sets._ A Machine Learning model is trained by starting with an initial guess for the weights and bias and iteratively adjusting those guesses until learning the weights and bias with the lowest possible loss.
2. Usually, you iterate until overall loss stops changing or at least changes extremely slowly. When that happens, we say that the model has **converged**.
3. The process we use to traverse the loss function in search of loss minimum is called **gradient descent**.
4. In gradient descent, a **batch** is the total number of examples you use to calculate the gradient in a single iteration.
5. What if we could get the right gradient *on average* for much less computation? By choosing examples at random from our data set, we could estimate (albeit, noisily) a big average from a much smaller one. **Stochastic gradient descent** (**SGD**) takes this idea to the extreme--it uses only a single example (a batch size of 1) per iteration. Given enough iterations, SGD works but is very noisy. The term "stochastic" indicates that the one example comprising each batch is chosen at random.
6. **Mini-batch stochastic gradient descent** (**mini-batch SGD**) is a compromise between full-batch iteration and SGD. A mini-batch is typically between 10 and 1,000 examples, chosen at random. Mini-batch SGD reduces the amount of noise in SGD but is still more efficient than full-batch.

## Tooling

Tensorflow toolkit hierarchy:

![Simplified hierarchy of TensorFlow toolkits. tf.keras API is at    the top.](https://developers.google.com/machine-learning/crash-course/images/TFHierarchyNew.svg)

#### Hyperparameter Tuning Guidlines

Most machine learning problems require a lot of hyperparameter tuning. Unfortunately, we can't provide concrete tuning rules for every model. Lowering the learning rate can help one model converge efficiently but make another model converge much too slowly. You must experiment to find the best set of hyperparameters for your dataset. That said, here are a few rules of thumb:

- Training loss should steadily decrease, steeply at first, and then more slowly until the slope of the curve reaches or approaches zero.
- If the training loss does not converge, train for more epochs.
- If the training loss decreases too slowly, increase the learning rate. Note that setting the training loss too high may also prevent training loss from converging.
- If the training loss varies wildly (that is, the training loss jumps around), decrease the learning rate.
- Lowering the learning rate while increasing the number of epochs or the batch size is often a good combination.
- Setting the batch size to a *very* small batch number can also cause instability. First, try large batch size values. Then, decrease the batch size until you see degradation.
- For real-world datasets consisting of a very large number of examples, the entire dataset might not fit into memory. In such cases, you'll need to reduce the batch size to enable a batch to fit into memory.

Remember: the ideal combination of hyperparameters is data dependent, so you must always experiment and verify.









