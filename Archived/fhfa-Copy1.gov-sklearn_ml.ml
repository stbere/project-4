{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 58,
   "id": "e4b94e59",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "from sklearn.ensemble import RandomForestRegressor\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import r2_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "3b1f9630",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Easy part: load the data\n",
    "df = pd.read_csv('../resources/fhfagov-2021.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "6508e0f8",
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
   "id": "8d401597",
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
   "id": "2ffc9ba3",
   "metadata": {},
   "outputs": [],
   "source": [
    "# This is 1 of my fave models: a Random Forest Regressor\n",
    "model = RandomForestRegressor(n_estimators=300, max_depth=30, min_samples_split=15, random_state=42)\n",
    "\n",
    "# Steve note: no matter how much I adjusted the hyperparameters,\n",
    "# 300, 30 were the best values for n_estimators and max_depth.\n",
    "# Adjusting too high took too long with no desirable results;\n",
    "# Adjusting too low yielded a worse MAE"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "c0092af9",
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
    "# Very important to fit the model to the training data! See below\n",
    "model.fit(train_data[ind_vars], train_data[target_var])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "328e2f95",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Let's make predictions on the test data\n",
    "test_data['predictions'] = model.predict(test_data[ind_vars])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "id": "a17404f9",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Now we shall calculate the mean absolute error\n",
    "mae = (test_data[target_var] - test_data['predictions']).abs().mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "c1468c23",
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
   "execution_count": 59,
   "id": "db9e90af",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "The R^2 score of the model is 0.34.\n"
     ]
    }
   ],
   "source": [
    "# r_squared = model.score(test_data[ind_vars], test_data[target_var])\n",
    "\n",
    "# Calculate the R^2 and MAE\n",
    "r2 = r2_score(test_data[target_var], test_data['predictions'])\n",
    "mae = (test_data[target_var] - test_data['predictions']).abs().mean()\n",
    "\n",
    "# Print the results\n",
    "print(f\"The R^2 score of our model using is {r2:.2f}.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "id": "c53a14c1",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model Accuracy: 33.77%\n"
     ]
    }
   ],
   "source": [
    "r_squared = model.score(test_data[ind_vars], test_data[target_var])\n",
    "accuracy = round(r_squared * 100, 2)\n",
    "print(f\"Model Accuracy: {accuracy}%\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "id": "3032e33c",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Let's right the output to a text file\n",
    "with open('2021fhfa.gov_skearlnml-results.txt', 'w') as f:\n",
    "    f.write(f\"For sklearn.ml file: what does our model output 'Mean Absolute Error: {mae:.2f}' actually explain? It means the predictions made by the Random Forest Regressor are off by about ${mae:.2f} in terms of our csv column 'NoteAmount'. So if the actual value of a house in the test data is $200,000, the model might predict that the house is worth around ${200000+mae:.2f}.\\n\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7dd70a4e",
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
