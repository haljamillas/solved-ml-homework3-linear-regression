Download Link: https://assignmentchef.com/product/solved-ml-homework3-linear-regression
<br>



<h1></h1>

<ol>

 <li>Consider a noisy target , where <strong>x </strong>∈ R<em><sup>d</sup></em><sup>+1 </sup>(including the added coordinate <em>x</em><sub>0 </sub>= 1), <em>y </em>∈ R, <strong>w</strong><em><sub>f </sub></em>∈ R<em><sup>d</sup></em><sup>+1 </sup>is an unknown vector, and is an i.i.d. noise term with zero mean and <em>σ</em><sup>2 </sup> Assume that we run linear regression on a training data set D = {(<strong>x</strong><sub>1</sub><em>,y</em><sub>1</sub>)<em>,…,</em>(<strong>x</strong><em><sub>N</sub>,y<sub>N</sub></em>)} generated i.i.d. from some <em>P</em>(<strong>x</strong>) and the noise process above, and obtain the weight vector <strong>w</strong><sub>lin</sub>. As briefly discussed in Lecture 9, it can be shown that the expected in-sample error <em>E</em><sub>in</sub>(<strong>w</strong><sub>lin</sub>) with respect to D is given by:</li>

</ol>

<em>.</em>

For <em>σ </em>= 0<em>.</em>1 and <em>d </em>= 11, what is the smallest number of examples <em>N </em>such that ED [<em>E</em><sub>in</sub>(<strong>w</strong><sub>lin</sub>)] is no less than 0<em>.</em>006? Choose the correct answer; explain your answer.

<ul>

 <li>25</li>

 <li>30</li>

 <li>35</li>

 <li>40</li>

 <li>45</li>

</ul>

<ol start="2">

 <li>As shown in Lecture 9, minimizing <em>E</em><sub>in</sub>(<strong>w</strong>) for linear regression means solving ∇<em>E</em><sub>in</sub>(<strong>w</strong>) = 0, which in term means solving the so-called <em>normal equation</em></li>

</ol>

X<em><sup>T</sup></em>X<strong>w </strong>= X<em><sup>T</sup></em><strong>y</strong><em>.</em>

Which of the following statement about the normal equation is correct for any features X and labels <strong>y</strong>? Choose the correct answer; explain your answer.

<ul>

 <li>There exists at least one solution for the normal equation.</li>

 <li>If there exists a solution for the normal equation, <em>E</em><sub>in</sub>(<strong>w</strong>) = 0 at such a solution.</li>

 <li>If there exists a <em>unique </em>solution for the normal equation, <em>E</em><sub>in</sub>(<strong>w</strong>) = 0 at the solution.</li>

 <li>If <em>E</em><sub>in</sub>(<strong>w</strong>) = 0 at some <strong>w</strong>, there exists a <em>unique </em>solution for the normal equation. <strong>[e] </strong>none of the other choices</li>

</ul>

<ol start="3">

 <li>In Lecture 9, we introduced the hat matrix H = XX<sup>† </sup>for linear regression. The matrix projects the label vector <strong>y </strong>to the “predicted” vector <strong>y</strong>ˆ = H<strong>y </strong>and helps us analyze the error of linear regression. Assume that X<em><sup>T</sup></em>X is invertible, which makes H = X(X<em><sup>T</sup></em>X)<sup>−1</sup>X<em><sup>T</sup></em>. Now, consider the following operations on X. Which operation can possibly change H? Choose the correct answer; explain your answer.

  <ul>

   <li>multiplying the whole matrix X by 2 (which is equivalent to scaling all input vectors by 2)</li>

   <li>multiplying each of the <em>i</em>-th column of X by <em>i </em>(which is equivalent to scaling the <em>i</em>-th feature by <em>i</em>)</li>

   <li>multiplying each of the <em>n</em>-th row of X by (which is equivalent to scaling the <em>n</em>-th example by</li>

   <li>adding three randomly-chosen columns <em>i,j,k </em>to column 1 of X</li>

  </ul></li>

</ol>

(i.e., <em>x</em><em>n,</em>1 ← <em>x</em><em>n,</em>1 + <em>x</em><em>n,i </em>+ <em>x</em><em>n,j </em>+ <em>x</em><em>n,k</em>)

<ul>

 <li>none of the other choices (i.e. all other choices are guaranteed to keep H unchanged.)</li>

</ul>

<h1>Likelihood and Maximum Likelihood</h1>

<ol start="4">

 <li>Consider a coin with an unknown head probability <em>θ</em>. Independently flip this coin <em>N </em>times to get <em>y</em><sub>1</sub><em>,y</em><sub>2</sub><em>,…,y<sub>N</sub></em>, where <em>y<sub>n </sub></em>= 1 if the <em>n</em>-th flipping results in head, and 0 otherwise. Define</li>

</ol>

. How many of the following statements about <em>ν </em>are true? Choose the correct

answer; explain your answer by illustrating why those statements are true.

<ul>

 <li>Pr() for all <em>N </em>∈ N and</li>

 <li><em>ν </em>maximizes likelihood(<em>θ</em><sup>ˆ</sup>) over all <em>θ</em><sup>ˆ</sup>∈ [0<em>,</em>1].</li>

 <li><em>ν </em>minimizes over all ˆ<em>y </em>∈ R.</li>

 <li>2 · <em>ν </em>is the negative gradient direction −∇<em>E</em><sub>in</sub>(<em>y</em>ˆ) at ˆ<em>y </em>= 0.</li>

</ul>

(<em>Note: θ is similar to the role of the “target function” and θ</em><sup>ˆ </sup><em>is similar to the role of the “hypothesis” in our machine learning framework.</em>)

<ul>

 <li>0</li>

 <li>1</li>

 <li>2</li>

 <li>3</li>

 <li>4</li>

</ul>

<ol start="5">

 <li>Let <em>y</em><sub>1</sub><em>,y</em><sub>2</sub><em>,…,y<sub>N </sub></em>be <em>N </em>values generated i.i.d. from a uniform distribution [0<em>,θ</em>] with some unknown <em>θ</em>. For any <em>θ</em><sup>ˆ </sup>≥ max(<em>y</em><sub>1</sub><em>,y</em><sub>2</sub><em>,…,y<sub>N</sub></em>), what is its likelihood? Choose the correct answer; explain your answer.</li>

</ol>

<em>(Hint: Those who are interested in more math [who isn’t? :-)] are encouraged to try to derive the maximum-likelihood estimator.)</em>

<h1>Gradient and Stochastic Gradient Descent</h1>

<ol start="6">

 <li>In the perceptron learning algorithm, we find one example (<strong>x</strong><em><sub>n</sub></em><sub>(<em>t</em>)</sub><em>,y<sub>n</sub></em><sub>(<em>t</em>)</sub>) that the current weight vector <strong>w</strong><em><sub>t </sub></em>mis-classifies, and then update <strong>w</strong><em><sub>t </sub></em>by</li>

</ol>

<strong>w</strong><em><sub>t</sub></em>+1 ← <strong>w</strong><em><sub>t </sub></em>+ <em>y<sub>n</sub></em>(<em>t</em>)<strong>x</strong><em><sub>n</sub></em>(<em>t</em>)<em>.</em>

A variant of the algorithm finds <em>all </em>examples (<strong>x</strong><em><sub>n</sub>,y<sub>n</sub></em>) that the weight vector <strong>w</strong><em><sub>t </sub></em>mis-classifies (e.g. <em>y<sub>n </sub></em>6= sign(<strong>w</strong><em><sub>t</sub><sup>T</sup></em><strong>x</strong><em><sub>n</sub></em>)), and then update <strong>w</strong><em><sub>t </sub></em>by

<strong>w</strong><em>.</em>

<em>n</em>: <em>y<sub>n</sub></em>6=sign(<strong><sup>w</sup></strong><em><sub>t</sub></em><em><sup>T</sup></em><strong>x</strong><em><sub>n</sub></em>)

The variant can be viewed as optimizing some <em>E</em><sub>in</sub>(<strong>w</strong>) that is composed of one of the following pointwise error functions with a fixed learning rate gradient descent (neglecting any non-differentiable spots of <em>E</em><sub>in</sub>). What is the error function? Choose the correct answer; explain your answer.

<ul>

 <li>err(<strong>w</strong><em>,</em><strong>x</strong><em>,y</em>) = |1 − <em>y</em><strong>w</strong><em><sup>T</sup></em><strong>x</strong>|</li>

 <li>err(<strong>w</strong><em>,</em><strong>x</strong><em>,y</em>) = max(0<em>,</em>−<em>y</em><strong>w</strong><em><sup>T</sup></em><strong>x</strong>) <strong>[c] </strong>err(<strong>w</strong><em>,</em><strong>x</strong><em>,y</em>) = −<em>y</em><strong>w</strong><em><sup>T</sup></em><strong>x</strong></li>

 <li>err(<strong>w</strong><em>,</em><strong>x</strong><em>,y</em>) = min(0<em>,</em>−<em>y</em><strong>w</strong><em><sup>T</sup></em><strong>x</strong>)</li>

 <li>err(<strong>w</strong><em>,</em><strong>x</strong><em>,y</em>) = max(0<em>,</em>1 − <em>y</em><strong>w</strong><em><sup>T</sup></em><strong>x</strong>)</li>

</ul>

<ol start="7">

 <li>Besides the error functions introduced in the lectures so far, the following error function, exponential error, is also widely used by some learning models. The exponential error is defined by err<sub>exp</sub>(<strong>w</strong><em>,</em><strong>x</strong><em>,y</em>) = exp(−<em>y</em><strong>w</strong><em><sup>T</sup></em><strong>x</strong>). If we want to use stochastic gradient descent to minimize an <em>E</em><sub>in</sub>(<strong>w</strong>) that is composed of the error function, which of the following is the update direction −∇err<sub>exp</sub>(<strong>w</strong><em>,</em><strong>x</strong><em><sub>n</sub>,y<sub>n</sub></em>) for the chosen (<strong>x</strong><em><sub>n</sub>,y<sub>n</sub></em>) with respect to <strong>w</strong><em><sub>t</sub></em>? Choose the correct answer; explain your answer.

  <ul>

   <li>+<em>y<sub>n</sub></em><strong>x</strong><em><sub>n </sub></em>exp(−<em>y<sub>n</sub></em><strong>w</strong><em><sup>T</sup></em><strong>x</strong><em><sub>n</sub></em>)</li>

   <li>−<em>y<sub>n</sub></em><strong>x</strong><em><sub>n </sub></em>exp(−<em>y<sub>n</sub></em><strong>w</strong><em><sup>T</sup></em><strong>x</strong><em><sub>n</sub></em>)</li>

   <li>+<strong>x</strong><em><sub>n </sub></em>exp(−<em>y<sub>n</sub></em><strong>w</strong><em><sup>T</sup></em><strong>x</strong><em><sub>n</sub></em>)</li>

   <li>−<strong>x</strong><em><sub>n </sub></em>exp(−<em>y<sub>n</sub></em><strong>w</strong><em><sup>T</sup></em><strong>x</strong><em><sub>n</sub></em>)</li>

   <li>none of the other choices</li>

  </ul></li>

</ol>

<h1>Hessian and Newton Method</h1>

<ol start="8">

 <li>Let <em>E</em>(<strong>w</strong>): R<em><sup>d </sup></em>→ R be a function. Denote the gradient <strong>b</strong><em><sub>E</sub></em>(<strong>w</strong>) and the Hessian A<em><sub>E</sub></em>(<strong>w</strong>) by

  <ul>

   <li>and A<em>.</em></li>

  </ul></li>

</ol>

Then, the second-order Taylor’s expansion of <em>E</em>(<strong>w</strong>) around <strong>u </strong>is:

<em>.</em>

Suppose A<em><sub>E</sub></em>(<strong>u</strong>) is positive definite. What is the optimal direction <strong>v </strong>such that <strong>w </strong>← <strong>u</strong>+<strong>v </strong>minimizes the right-hand-side of the Taylor’s expansion above? Choose the correct answer; explain your answer. (<em>Note that iterative optimization with </em><strong>v </strong><em>is generally called Newton’s method.</em>)

<ul>

 <li>+(A<em><sub>E</sub></em>(<strong>u</strong>))<sup>−1</sup><strong>b</strong><em><sub>E</sub></em>(<strong>u</strong>)</li>

 <li>−(A<em><sub>E</sub></em>(<strong>u</strong>))<sup>−1</sup><strong>b</strong><em><sub>E</sub></em>(<strong>u</strong>)</li>

 <li>+(A<em><sub>E</sub></em>(<strong>u</strong>))<sup>+1</sup><strong>b</strong><em><sub>E</sub></em>(<strong>u</strong>)</li>

 <li>−(A<em><sub>E</sub></em>(<strong>u</strong>))<sup>+1</sup><strong>b</strong><em><sub>E</sub></em>(<strong>u</strong>)</li>

 <li>none of the other choices</li>

</ul>

<ol start="9">

 <li>Following the previous problem, considering minimizing <em>E</em><sub>in</sub>(<strong>w</strong>) in linear regression problem with Newton’s method. For any given <strong>w</strong><em><sub>t</sub></em>, what is the Hessian A<em><sub>E</sub></em>(<strong>w</strong><em><sub>t</sub></em>) with <em>E </em>= <em>E</em><sub>in</sub>? Choose the correct answer; explain your answer.

  <ul>

   <li><em>N</em><u>2 </u>X<em>T</em>X</li>

  </ul></li>

</ol>

X

<strong>[e] </strong>none of the other choices

<h1>Multinomial Logistic Regression</h1>

<ol start="10">

 <li>In Lecture 11, we solve multiclass classification by OVA or OVO decompositions. One alternative to deal with multiclass classification is to extend the original logistic regression model to Multinomial Logistic Regression (MLR). For a <em>K</em>-class classification problem, we will denote the output space</li>

</ol>

Y = {1<em>,</em>2<em>,</em>··· <em>,K</em>}. The hypotheses considered by MLR can be indexed by a matrix

W = <em>,</em>

that contains weight vectors (<strong>w</strong><sub>1</sub><em>,</em>··· <em>,</em><strong>w</strong><em><sub>K</sub></em>), each of length <em>d</em>+1. The matrix represents a hypothesis

that can be used to approximate the target distribution <em>P</em>(<em>y</em>|<strong>x</strong>) for any (<strong>x</strong><em>,y</em>). MLR then seeks for the maximum likelihood solution over all such hypotheses. For a given data set {(<strong>x</strong><sub>1</sub><em>,y</em><sub>1</sub>)<em>,…,</em>(<strong>x</strong><em><sub>N</sub>,y<sub>N</sub></em>)} generated i.i.d. from some <em>P</em>(<strong>x</strong>) and target distribution <em>P</em>(<em>y</em>|<strong>x</strong>), the likelihood of <em>h<sub>y</sub></em>(<strong>x</strong>) is proportional to). That is, minimizing the negative log likelihood is equivalent to minimizing an <em>E</em><sub>in</sub>(W) that is composed of the following error function

<em>K</em>

err(W<em>,</em><strong>x</strong><em>,y</em>) = −ln<em>h<sub>y</sub></em>(<strong>x</strong>) = <sup>−X </sup><em>y </em>= <em>k </em>ln<em>h<sub>k</sub></em>(<strong>x</strong>)<em>. </em><em>k</em>=1 J     K

When minimizing <em>E</em><sub>in</sub>(W) with SGD, we need to compute <em><sup>∂</sup></em><sup>err(W</sup><em><sub>∂</sub></em><sub>W</sub><em>ik<sup>,</sup></em><strong><sup>x</sup></strong><em><sup>,y</sup></em><sup>)</sup>. What is the value of the partial derivative? Choose the correct answer; explain your answer.

J K

<strong>[e] </strong>none of the other choices

<ol start="11">

 <li>Following the previous problem, consider a data set with <em>K </em>= 2 and obtain the optimal solution from MLR as (<strong>w</strong>). Now, relabel the same data set by replacing <em>y<sub>n </sub></em>with 3 to form a binary classification data set. Which of the following is an optimal solution for logistic regression on the binary classification data set? Choose the correct answer; explain your answer.</li>

</ol>

<h1>Nonlinear Transformation</h1>

<ol start="12">

 <li>Given the following training data set:</li>

</ol>

<strong>x</strong><sub>1 </sub>= (0<em>,</em>1)<em>,y</em><sub>1 </sub>= −1            <strong>x</strong><sub>2 </sub>= (1<em>,</em>−0<em>.</em>5)<em>,y</em><sub>2 </sub>= −1                 <strong>x</strong><sub>3 </sub>= (−1<em>,</em>0)<em>,y</em><sub>3 </sub>= −1

<strong>x</strong><sub>4 </sub>= (−1<em>,</em>2)<em>,y</em><sub>4 </sub>= +1                   <strong>x</strong><sub>5 </sub>= (2<em>,</em>0)<em>,y</em><sub>5 </sub>= +1            <strong>x</strong><sub>6 </sub>= (1<em>,</em>−1<em>.</em>5)<em>,y</em><sub>6 </sub>= +1             <strong>x</strong><sub>7 </sub>= (0<em>,</em>−2)<em>,y</em><sub>7 </sub>= +1

Using the quadratic transform ), which of the following weights <strong>w</strong>˜ <em><sup>T </sup></em>in the Z-space can separate all of the training data correctly? Choose the correct answer; (<em>no, you don’t need to explain your answer &#x1f642;</em>).

<ul>

 <li>[−9<em>,</em>−1<em>,</em>0<em>,</em>2<em>,</em>−2<em>,</em>3]</li>

 <li>[−5<em>,</em>−1<em>,</em>2<em>,</em>3<em>,</em>−7<em>,</em>2]</li>

 <li>[9<em>,</em>−1<em>,</em>4<em>,</em>2<em>,</em>−2<em>,</em>3]</li>

 <li>[2<em>,</em>1<em>,</em>−4<em>,</em>−2<em>,</em>7<em>,</em>−4]</li>

 <li>[−7<em>,</em>0<em>,</em>0<em>,</em>2<em>,</em>−2<em>,</em>3]</li>

</ul>

<ol start="13">

 <li>Consider the following feature transform, which maps <strong>x </strong>∈ R<em><sup>d </sup></em>to <strong>z </strong>∈ R<sup>1+1</sup>, keeping only the <em>k</em>th coordinate of <strong>x</strong>: <strong>Φ</strong><sub>(<em>k</em>)</sub>(<strong>x</strong>) = (1<em>,x<sub>k</sub></em>). Let H<em><sub>k </sub></em>be the set of hypothesis that couples <strong>Φ</strong><sub>(<em>k</em>) </sub>with perceptrons. Among the following choices, which of is the tightest upper bound of for <em>d </em>≥ 4? Choose the correct answer; explain your answer. (<em>Hint: You can use the fact that </em><em>for d </em>≥ 4 <em>if needed.</em>)

  <ul>

   <li>2((log<sub>2 </sub>log<sub>2 </sub><em>d</em>) + 1)</li>

   <li>2((log<sub>2 </sub><em>d</em>) + 1)</li>

   <li>2((<em>d</em>log<sub>2 </sub><em>d</em>) + 1)</li>

   <li>2(<em>d </em>+ 1)</li>

   <li>2(<em>d</em><sup>2 </sup>+ 1)</li>

  </ul></li>

</ol>

<h1>Experiments with Linear and Nonlinear Models</h1>

Next, we will play with linear regression, logistic regression, non-linear transform, and their use for binary classification. Please use the following set for training:

https://www.csie.ntu.edu.tw/~htlin/course/ml20fall/hw3/hw3_train.dat

and the following set for testing (estimating <em>E</em><sub>out</sub>):

https://www.csie.ntu.edu.tw/~htlin/course/ml20fall/hw3/hw3_test.dat

Each line of the data set contains one (<strong>x</strong><em><sub>n</sub>,y<sub>n</sub></em>) with <strong>x</strong><em><sub>n </sub></em>∈ R<sup>10</sup>. The first 10 numbers of the line contains the components of <strong>x</strong><em><sub>n </sub></em>orderly, the last number is <em>y<sub>n</sub></em>, which belongs to {−1<em>,</em>+1} ⊆ R. That is, we can use those <em>y<sub>n </sub></em>for either binary classification or regression.

<ol start="14">

 <li>(*) Add <em>x<sub>n,</sub></em><sub>0 </sub>= 1 to each <strong>x</strong><em><sub>n</sub></em>. Then, implement the linear regression algorithm on page 11 of Lecture 9. What is <em>E</em><sub>in</sub><sup>sqr</sup>(<strong>w</strong><sub>lin</sub>), where <em>E</em><sub>in</sub><sup>sqr </sup>denotes the <em>averaged </em>squared error over <em>N </em>examples? Choose the closest answer; provide your code.

  <ul>

   <li>0<em>.</em>00</li>

   <li>0<em>.</em>20</li>

   <li>0<em>.</em>40</li>

   <li>0<em>.</em>60</li>

   <li>0<em>.</em>80</li>

  </ul></li>

 <li>(*) Add <em>x<sub>n,</sub></em><sub>0 </sub>= 1 to each <strong>x</strong><em><sub>n</sub></em>. Then, implement the SGD algorithm for linear regression using the results on pages 10 and 12 of Lecture 11. Pick one example uniformly at random in each iteration, take <em>η </em>= 0<em>.</em>001 and initialize <strong>w </strong>with <strong>w</strong><sub>0 </sub>= <strong>0</strong>. Run the algorithm until <em>E</em><sub>in</sub><sup>sqr</sup>(<strong>w</strong><em><sub>t</sub></em>) ≤ 1<em>.</em>01<em>E</em><sub>in</sub><sup>sqr</sup>(<strong>w</strong><sub>lin</sub>), and record the total number of iterations taken. Repeat the experiment 1000 times, each with a different random seed. What is the average number of iterations over the 1000 experiments? Choose the closest answer; provide your code.

  <ul>

   <li>600</li>

   <li>1200</li>

   <li>1800</li>

   <li>2400</li>

   <li>3000</li>

  </ul></li>

 <li>(*) Add <em>x<sub>n,</sub></em><sub>0 </sub>= 1 to each <strong>x</strong><em><sub>n</sub></em>. Then, implement the SGD algorithm for logistic regression by replacing the SGD update step in the previous problem with the one on page 10 of Lecture 11. Pick one example uniformly at random in each iteration, take <em>η </em>= 0<em>.</em>001 and initialize <strong>w </strong>with <strong>w</strong><sub>0 </sub>= <strong>0</strong>. Run the algorithm for 500 iterations. Repeat the experiment 1000 times, each with a different random seed. What is the average ) over the 1000 experiments, where <em>E</em><sub>in</sub><sup>ce </sup>denotes the <em>averaged </em>cross-entropy error over <em>N </em>examples? Choose the closest answer; provide your code.

  <ul>

   <li>0<em>.</em>44</li>

   <li>0<em>.</em>50</li>

   <li>0<em>.</em>56</li>

   <li>0<em>.</em>62</li>

   <li>0<em>.</em>68</li>

  </ul></li>

 <li>(*) Repeat the previous problem, but with <strong>w </strong>initialized by <strong>w</strong><sub>0 </sub>= <strong>w</strong><sub>lin </sub>of Problem 14 instead. Repeat the experiment 1000 times, each with a different random seed. What is the average) over the 1000 experiments? Choose the closest answer; provide your code.

  <ul>

   <li>0<em>.</em>44</li>

   <li>0<em>.</em>50</li>

   <li>0<em>.</em>56</li>

   <li>0<em>.</em>62</li>

   <li>0<em>.</em>68</li>

  </ul></li>

 <li>(*) Following Problem 14, what is(<strong>w</strong><sub>lin</sub>) (<strong>w</strong><sub>lin</sub>), where 0<em>/</em>1 denotes the 0/1 error (i.e.</li>

</ol>

(0<em>/</em>1) using <strong>w</strong><sub>lin </sub>for binary classification), and <em>E</em><sub>out </sub>is estimated using the test set provided above? Choose the closest answer; provide your code.

<ul>

 <li>0<em>.</em>32</li>

 <li>0<em>.</em>36</li>

 <li>0<em>.</em>40</li>

 <li>0<em>.</em>44</li>

 <li>0<em>.</em>48</li>

</ul>

<ol start="19">

 <li>(*) Next, consider the following <em>homogeneous </em>order-<em>Q </em>polynomial transform</li>

</ol>

<em>.</em>

Transform the training and testing data according to <strong>Φ</strong>(<strong>x</strong>) with <em>Q </em>= 3, and again implement the linear regression algorithm on page 11 of lecture 9. What is, where <em>g </em>is the hypothesis returned by the transform + linear regression procedure? Choose the closest answer; provide your code.

<ul>

 <li>0<em>.</em>32</li>

 <li>0<em>.</em>36</li>

 <li>0<em>.</em>40</li>

 <li>0<em>.</em>44</li>

 <li>0<em>.</em>48</li>

</ul>

<ol start="20">

 <li>(*) Repeat the previous problem, but with <em>Q </em>= 10 instead. What is ? Choose the closest answer; provide your code.

  <ul>

   <li>0<em>.</em>32</li>

   <li>0<em>.</em>36</li>

   <li>0<em>.</em>40</li>

   <li>0<em>.</em>44</li>

   <li>0<em>.</em>48</li>

  </ul></li>

</ol>