{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "b7f9234d",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "from sklearn.ensemble import RandomForestRegressor\n",
    "from sklearn.model_selection import train_test_split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "34ea700e",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load the data\n",
    "df = pd.read_csv('../resources/fhfagov-2021.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "277b295e",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Define the input and output variables\n",
    "ind_vars = ['FIPSStateNumericCode', 'FIPSCountyCode', 'CoreBasedStatisticalAreaCode', 'CensusTractIdentifier']\n",
    "target_var = 'NoteAmount'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "1a26f690",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Split the data into training and test sets\n",
    "train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "13dd0978",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create a Random Forest Regressor\n",
    "model = RandomForestRegressor(n_estimators=300, max_depth=30, min_samples_split=15, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "a8fcf9b1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "RandomForestRegressor(max_depth=30, min_samples_split=15, n_estimators=300,\n",
       "                      random_state=42)"
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Fit the model to the training data\n",
    "model.fit(train_data[ind_vars], train_data[target_var])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "d6709f9a",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Make predictions on the test data\n",
    "test_data['predictions'] = model.predict(test_data[ind_vars])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "id": "ad7362ad",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Calculate the mean absolute error\n",
    "mae = (test_data[target_var] - test_data['predictions']).abs().mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "8a5e427d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Mean Absolute Error: 77265.48\n"
     ]
    }
   ],
   "source": [
    "print(f\"Mean Absolute Error: {mae:.2f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8f482a70",
   "metadata": {},
   "outputs": [],
   "source": [
    "# What does the above output Mean Absolute Error: 77265.48 actually mean? It means the \n",
    "# predictions made by the Random Forest Regressor are off by about $77,265.48 in terms \n",
    "# of our csv column 'NoteAmount'. So if the actual value of a house in the test data is \n",
    "# $200,000, the model might predict that the house is worth around $277,265."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
