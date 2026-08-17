# CAB330 Assignment 1 – Hotel Booking Cancellation Prediction

This repository contains the code and supporting files for **CAB330 Assignment 1**.

The project focuses on analysing hotel booking data and developing machine learning models to predict whether a hotel booking will be **cancelled or not cancelled**.

The project includes:

- Exploratory Data Analysis (EDA)
- Data quality investigation
- Data preprocessing
- Data partitioning
- Decision Tree modelling
- Logistic Regression modelling
- Neural Network modelling
- Model evaluation and comparison

---

## Project Structure

```text
CAB330_Assignment1/
│
├── data/
│   ├── clean_data.csv
│   ├── clean_group_data.csv
│   ├── clean_test_A.csv
│   ├── clean_test_B.csv
│   └── clean_test_C.csv
│
├── models/
│   └── Model-related code
│
├── evaluation/
│   └── Model evaluation and plotting functions
│
├── notebooks/
│   └── Jupyter notebooks for EDA, preprocessing and modelling
│
├── requirements.txt
└── README.md
```

### `data/`

Contains the cleaned and partitioned datasets used throughout the project.

- `clean_data.csv` – Preprocessed hotel booking dataset
- `clean_group_data.csv` – Dataset used by the group for Tasks 2–4
- `clean_test_A.csv` – Individual test dataset for Student A
- `clean_test_B.csv` – Individual test dataset for Student B
- `clean_test_C.csv` – Individual test dataset for Student C

### `models/`

Contains code related to the machine learning models developed for the assignment.

The three main modelling approaches are:

1. Decision Tree
2. Logistic Regression
3. Neural Network

### `evaluation/`

Contains reusable functions for evaluating model performance, such as:

- Accuracy
- Precision
- Recall
- F1-score
- Classification reports
- ROC curves
- ROC-AUC
- Model comparison plots

### `notebooks/`

Contains the Jupyter notebooks used to perform the analysis, preprocessing, model development and evaluation.

---

# Task 1 – Data Selection and Distribution

Task 1 focuses on understanding and preparing the hotel booking dataset before model development.

The preprocessing workflow includes:

1. Inspecting the dataset and variable types
2. Examining the target variable distribution
3. Identifying missing values
4. Identifying invalid or suspicious values
5. Investigating redundant variables
6. Selecting appropriate preprocessing methods
7. Cleaning the dataset
8. Validating the cleaned dataset
9. Partitioning the data for group and individual modelling

## Target Variable

The target variable is:

```text
is_canceled
```

where:

```text
Y = Booking cancelled
N = Booking not cancelled
```

---

## Data Quality and Preprocessing

Several data-quality issues were identified during EDA.

### Lead Time

Some `lead_time` observations contained an invalid value of `-1`.

Since lead time represents the number of days between booking and arrival, negative values were treated as invalid.

The invalid values were converted to missing values and replaced using the **median lead time**.

The variable was then converted back to integer type because lead time represents a number of days.

### Children

Four observations contained missing values for `children`.

The relationship between:

```text
total_guests = adults + children + babies
```

was examined.

The missing child values were determined to represent zero children and were replaced with `0`.

### Country

A small number of observations contained missing `country` values.

Rather than assuming that these observations belonged to the most common country, they were assigned to an:

```text
Unknown
```

category.

### Reserved Room Type

`reserved_room_type` contained both:

- Missing (`NaN`) values
- `"?"` values

Both represented unavailable room-type information and were therefore combined into an:

```text
Unknown
```

category.

### ADR

Missing `adr` values were replaced using the **median ADR**.

Median imputation was selected because it is less sensitive to unusually high values than the mean.

### Zero-Guest Bookings

Some observations contained:

```text
adults = 0
children = 0
babies = 0
total_guests = 0
```

These records were considered invalid because they represented hotel bookings with no recorded guests.

The affected records were removed.

### Removed Variables

The following variables were removed during preprocessing:

#### `booking_id`

Removed because it is a unique record identifier and does not provide meaningful information for predicting cancellation.

#### `company`

Removed because the majority of observations were missing, making reliable use or imputation of the variable difficult.

#### `total_guests`

Removed because it is redundant and can be calculated using:

```text
adults + children + babies
```

The individual guest variables were retained because they provide more information about the composition of each booking.

---

# Data Partitioning

After preprocessing, the cleaned dataset is saved as:

```text
clean_data.csv
```

The supplied data partitioning procedure is then used to generate:

```text
clean_group_data.csv
clean_test_A.csv
clean_test_B.csv
clean_test_C.csv
```

The group dataset is used for model development and comparison in **Tasks 2–4**.

The individual datasets are reserved for the individual model evaluation performed in **Task 5**.

The group dataset is further divided into training and test sets for model development.

---

# Task 2 – Decision Tree

A Decision Tree classifier is developed to predict hotel booking cancellations.

The task includes:

- Training a default Decision Tree
- Evaluating training and test performance
- Investigating overfitting
- Examining tree depth and structure
- Identifying important variables
- Hyperparameter tuning
- Evaluating the tuned model

---

# Task 3 – Logistic Regression

A Logistic Regression model is developed and evaluated.

The task includes:

- Preparing the data for Logistic Regression
- Training the default model
- Evaluating model performance
- Investigating overfitting
- Examining important variables
- Hyperparameter tuning
- Evaluating the tuned model

---

# Task 4 – Neural Network

A Neural Network classifier is developed for hotel booking cancellation prediction.

The task includes:

- Preparing data for neural network training
- Designing the network architecture
- Training the default network
- Evaluating model performance
- Investigating overfitting
- Hyperparameter tuning
- Evaluating the tuned network

The final tuned Decision Tree, Logistic Regression and Neural Network models are then compared.

---

# Task 5 – Individual Evaluation

Each group member evaluates the final tuned models using their assigned individual test dataset.

The individual datasets are:

```text
clean_test_A.csv
clean_test_B.csv
clean_test_C.csv
```

The performance on these unseen datasets is compared with the group test results to evaluate how well the models generalise to new data.

---

# Model Evaluation

Model performance is evaluated using appropriate classification metrics, including:

- Accuracy
- Precision
- Recall
- F1-score
- Classification report
- ROC curve
- ROC-AUC

Training and test performance are also compared to identify possible overfitting.

---

# Installation

Clone the repository:

```bash
git clone https://github.com/Matsuzaki-Yuta/CAB330_Assignment1.git
```

Move into the project directory:

```bash
cd CAB330_Assignment1
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

---

# Running the Project

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open the relevant notebook from the `notebooks/` directory and run the cells in order.

The notebooks should be run from the project root so that relative paths such as:

```python
pd.read_csv("data/clean_group_data.csv")
```

work correctly.

---

# Technologies

The project uses:

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

Additional libraries may be included depending on the implementation of the neural network.

---

# Repository

CAB330 Assignment 1 Group Project

GitHub Repository:

https://github.com/Matsuzaki-Yuta/CAB330_Assignment1
