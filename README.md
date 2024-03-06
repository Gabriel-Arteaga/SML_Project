# Do (wo)men talk too much in films? - Statistical Machine Learning
This repository presents our solution for a project assigned during the Statistical Machine Learning course (5 ECTS - [1RT700](https://www.uu.se/en/study/syllabus?query=41831)) at Uppsala University.

## Table Of Contents
 * [Project Description](#project-description)
 * [Data Analysis](#data-analysis)
 * [Feature Engineering](#feature-engineering)
 * [Model Tuning](#model-tuning)
 * [Results](#results)
 * [Project Members](#project-members)
## Project Description
In this project, we were provided with a dataset comprising various features of different Hollywood movies. Specifically, the dataset included 13 features, and our objective was to classify the gender of the lead actor based on these features. Thus, the main task of the project involved conducting binary classification to determine the gender of lead actors in these Hollywood movies. The provided variables and their corresponding types are outlined in the tables below.

<div align="center">
  
| Feature                                          | Type     |
|--------------------------------------------------|----------|
| Number of words spoken by female                | Ratio    |
| Total number of words                           | Ratio    |
| Number of words spoken by lead actor            | Ratio    |
| Difference in words spoken between lead and co-lead | Ratio    |
| Number of male actors                           | Ratio    |
| Year                                             | Interval |
| Number of female actors                         | Ratio    |
| Number of words spoken by male                  | Ratio    |
| Gross                                            | Ratio    |
| Mean age of male actors                         | Ratio    |
| Mean age of female actors                       | Ratio    |
| Age of lead actor                               | Ratio    |
| Age of co-lead actor                            | Ratio    |
| Lead (Target Variable)                            | Nominal  |
</div>

## Data Analysis
Additionally, alongside the classification task, we were assigned to address specific inquiries based on the data, leading to a comprehensive analysis. The questions posed were as follows:

* **Do men or women dominate speaking roles in Hollywood movies?**
  Our analysis revealed that males comprised 68% of all actors with speaking roles, while females accounted for only 21% of the total words spoken in movies.
<p align="center">
  <img src="/data/readme_pics/question1.png" width="95%"  />
</p>
<p align="center">
  
* **Has gender balance in speaking roles changed over time?**  
  Our findings indicate that gender balance in speaking roles has become more homogeneous over the decades, particularly evident during the 2010s. However, despite this trend, the proportion of female share of words in movies remains low, at 24%, signifying a significant underrepresentation.
  <p align="center">
  <img src="/data/readme_pics/question2.png" width="95%"  />
</p>
<p align="center">

* **Do films in which men do more speaking make a lot more money than films in which women
speak more?**
Our analysis indicates that movies where male actors have the majority word count tend to generate higher average gross revenue compared to those where female actors dominate the word count. However, this trend seems to deviate as we approach the 2010s.
  <p align="center">
  <img src="/data/readme_pics/question3.png" height= 220px width="95%"  />
</p>
<p align="center">  
  
## Feature Engineering
To derive additional insights from our dataset, we employed feature engineering techniques to create new attributes based on our original variables. These techniques encompassed various methods such as creating shares (e.g., Female word share), aggregating data (e.g., Decade), and calculating differences (e.g., Age lead vs. co-lead). From our initial set of 13 features, we expanded our feature set to 40 attributes. Subsequently, we conducted comprehensive testing to identify the most relevant features for our respective models. Feature selection was carried out through a combination of manual methods (e.g., permutation testing) and automated approaches (e.g., `ExhaustiveFeatureSelector` & `SequentialFeatureSelector`).

## Model Tuning
We performed hyperparameter tuning and feature selection for each model within our model family, including Logistic Regression, Boosting, kNN, and Discriminant Analysis. As our primary project objective is to identify the best-performing model for "production," our focus lies on optimizing each model individually rather than benchmarking them against each other. Consequently, we applied different tuning efforts for the different models.

To facilitate hyperparameter optimization, we employed automated tools such as `RandomizedSearchCV` and `GridSearchCV`. For simpler models like kNN, we utilized GridSearch due to the manageable computational requirements for tuning parameters such as k. Conversely, for more complex models like XGBoost, we initially utilized RandomizedSearch to prune the parameter search space. Subsequently, after narrowing down the search space, we employed GridSearch to exhaustively identify optimal hyperparameters.
## Results
In our analysis, we aimed to identify the model that best generalizes to new data, prioritizing criteria such as high average accuracy in cross-validation, minimal spread across different folds, balanced accuracy for both classes, and simplicity in terms of features. Comparing the performance of various models, we found that more flexible, non-linear models like QDA and boosting outperformed simpler, linear ones and kNN. Notably, QDA emerged as the top-performing model overall within its family. Despite concerns of overfitting in some models, particularly XGB, QDA demonstrated robustness and accuracy. Feature engineering and selection proved to be crucial in improving model performance, highlighting core attributes such as Number words female and Number of words lead. While automated tools were helpful, manual refinement was also essential for optimal results. Despite computational constraints, our approach successfully identified a high-performing model suitable for production deployment.  

<div align="center">
  
| Metric           | Log. reg.  | QDA       | kNN       | XGB       |
|------------------|------------|-----------|-----------|-----------|
| Avg. accuracy   | 0.889       | 0.925      | 0.885      | 0.911      |
| Acc., min-max   | 0.861 - 0.899 | 0.885 - 0.955 | .875 - .908 | 0.885 - 0.947 |
| Acc. fem / male | 0.681 / 0.957 | 0.831 / 0.952 | 0.682 / 0.952 | 0.709 / 0.977 |
| Train accuracy  | 0.890       | 0.935      | 0.912      | 0.984      |
| N features      | 14         | 13        | 5         | 12        |
</div> 

## Project members
| [Gabriel Arteaga](https://github.com/Gabriel-Arteaga)  | [Jakob Nyström](https://github.com/j-nystrom)  |  [Inga Wohlert](https://github.com/IngaKristin)  |  [Alexander Sabelström](https://github.com/Sabelz) |

