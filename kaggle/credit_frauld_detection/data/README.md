There are several ways to approach this classification problem taking into consideration this unbalance.
1) Collect more data? Nice strategy but not applicable in this case
2) Changing the performance metric:
   --Use the confusio nmatrix to calculate Precision, Recall
   --F1-score (weighted average of precision recall)
   --Use Kappa - which is a classification accuracy normalized by the imbalance of the classes in the data
   --ROC curves - calculates sensitivity/specificity ratio.
3) Resampling the dataset
   Essentially this is a method that will process the data to have an approximate 50-50 ratio.
   One way to achieve this is by OVER-sampling, which is adding copies of the under-represented class (better when you have little data)
   Another is UNDER-sampling, which deletes instances from the over-represented class (better when he have lot's of data)


