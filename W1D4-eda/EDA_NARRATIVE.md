# W1D4 Exploratory Data Analysis Narrative

The dataset contains employee information such as Name, Age, City, Salary, and Department. I used Pandas to inspect the dataset, check its structure, view data types, generate summary statistics, and identify missing values and duplicate records. Age and Salary are the main numerical columns, while Name, City, and Department are categorical columns.

The analysis shows that the dataset contains missing values in some columns. These missing values need to be handled before using the dataset for further analysis or machine learning. I also checked duplicate records using Name, Age, and City because repeated records can affect the results of later analysis.

The distribution plots show how Age and Salary are spread across the available records. The Age vs Salary scatter plot provides a visual view of their relationship, while the correlation heatmap shows the numerical correlation between the two variables. The City value counts also show that some cities occur more frequently than others.

Some suspicious areas are the missing values and duplicate records. The data should also be checked for unusual Age or Salary values and inconsistent category names. Before applying machine learning, I would clean the missing values, review duplicates, check possible outliers, and validate the categorical values. These steps will make the dataset more reliable for further analysis.
