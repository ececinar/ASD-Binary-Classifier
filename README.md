# ASD-Binary-Classifier
Binary classifier ML algorithms to determine whether a patient has ASD or not.


Autism Spectrum Disorder(ASD) is a disorder which affects the learning ability, social communication skills
and behaviours of the patients. This study aimed to explore
a pattern between the brain regions in order to differentiate
patients with ASD from healthy individuals. Dimensionality reduction
was necessary since the given dataset had 595 features.
Using Principal Component Analysis(PCA), dimensionality was
reduced to 24 features. A classification model was trained
using the AdaBoost classifier with Gaussian Naive Bayes base
estimator. Overall classification accuracy of this model was
%60 in the public leaderboard and %52.5 in the private
leaderboard using the test set provided for the competition.
Although the classification accuracy was not significantly high
when compared to the other competition participants’ scores,
building up a robust algorithm was an important benchmark
which we managed to provide.

*Results*
KNN(K-Nearest Neighnors) and XGB classifier
algorithms were tried on datasets and very inadequate
results were obtained from these trials. KNN performs not
quite well with large amount of features and lower amount
of samples since the computations of N distances become
more complex.

In addition, random forest algorithm was implemented
but it cannot create the optimum effect on small samples.
Therefore, the use of the random forest algorithm has been
abandoned due to poor performance. Random Forest algorithm
was also combined with feature selection methods it
provides, such as finding feature importances. Based on the
obtained importance values, the features with importance
values lower than the mean of importances were eliminated.
This method is generally powerful with large datasets, but
since the dataset in the term project was not as large, it did
not perform quite well.

After that, AdaBoost which is a boosting algorithm to
create a strong classifier from a number of weak classifiers
was tested. At first, bernoulli was performed on datasets with
pca. After that gaussian naive bayes is used on datasets and
these results were better than their predecessors.

Also, svm with linear kernel trick and bagging are incorprated
into AdaBoost algorithm respectively. However, the
desired results could not be reached.

Lastly, random forest with feature importance selection
and adaboosting was employed, however random forest
perfoms well with bagging algorithms, not with boosting
algorithms. That’s the reason why random forest with adaboosting
got lower results, which was expected.

*Read the given pdf file for more detailed report of this experiment.*
