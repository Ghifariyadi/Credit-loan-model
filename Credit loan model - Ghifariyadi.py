{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "feb5bf28-cb7a-47f3-9f29-6f32f1bfdd14",
   "metadata": {},
   "source": [
    "# Import library"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "e0c14257-dc51-4a23-8470-2db60cfa0d2e",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "\n",
    "import warnings\n",
    "warnings.filterwarnings(\"ignore\")\n",
    "pd.set_option('display.max_rows', None)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "e99432d7-470c-41b3-b307-371191662b55",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score\n",
    "\n",
    "def eval_classification(model, pred, xtrain, ytrain, xtest, ytest):\n",
    "    print(\"Accuracy (Test Set): %.2f\" % accuracy_score(ytest, pred))\n",
    "    print(\"Precision (Test Set): %.2f\" % precision_score(ytest, pred))\n",
    "    print(\"Recall (Test Set): %.2f\" % recall_score(ytest, pred))\n",
    "    print(\"F1-Score (Test Set): %.2f\" % f1_score(ytest, pred))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e5f823ac-8646-4340-a280-b7f33179c461",
   "metadata": {
    "tags": []
   },
   "source": [
    "# Read the dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "082062c2-bd3c-42e7-81b1-777ee90e3a6a",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv('dataset.csv')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f6116dfb-3c2f-46e0-a8c1-fd3ba699ee33",
   "metadata": {},
   "source": [
    "## Dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "49b8fd4b-7883-42ab-a009-096766828e5d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Unnamed: 0</th>\n",
       "      <th>id</th>\n",
       "      <th>member_id</th>\n",
       "      <th>loan_amnt</th>\n",
       "      <th>funded_amnt</th>\n",
       "      <th>funded_amnt_inv</th>\n",
       "      <th>term</th>\n",
       "      <th>int_rate</th>\n",
       "      <th>installment</th>\n",
       "      <th>grade</th>\n",
       "      <th>...</th>\n",
       "      <th>total_bal_il</th>\n",
       "      <th>il_util</th>\n",
       "      <th>open_rv_12m</th>\n",
       "      <th>open_rv_24m</th>\n",
       "      <th>max_bal_bc</th>\n",
       "      <th>all_util</th>\n",
       "      <th>total_rev_hi_lim</th>\n",
       "      <th>inq_fi</th>\n",
       "      <th>total_cu_tl</th>\n",
       "      <th>inq_last_12m</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>413875</th>\n",
       "      <td>413875</td>\n",
       "      <td>13257156</td>\n",
       "      <td>15299376</td>\n",
       "      <td>10000</td>\n",
       "      <td>10000</td>\n",
       "      <td>10000.0</td>\n",
       "      <td>36 months</td>\n",
       "      <td>7.9</td>\n",
       "      <td>312.91</td>\n",
       "      <td>A</td>\n",
       "      <td>...</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>57800.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>1 rows × 75 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "        Unnamed: 0        id  member_id  loan_amnt  funded_amnt  \\\n",
       "413875      413875  13257156   15299376      10000        10000   \n",
       "\n",
       "        funded_amnt_inv        term  int_rate  installment grade  ...  \\\n",
       "413875          10000.0   36 months       7.9       312.91     A  ...   \n",
       "\n",
       "       total_bal_il il_util open_rv_12m open_rv_24m  max_bal_bc all_util  \\\n",
       "413875          NaN     NaN         NaN         NaN         NaN      NaN   \n",
       "\n",
       "       total_rev_hi_lim inq_fi total_cu_tl inq_last_12m  \n",
       "413875          57800.0    NaN         NaN          NaN  \n",
       "\n",
       "[1 rows x 75 columns]"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.sample()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "c1d25ec5-9b9b-42db-8b1a-5c4284c1d9c5",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 466285 entries, 0 to 466284\n",
      "Data columns (total 75 columns):\n",
      " #   Column                       Non-Null Count   Dtype  \n",
      "---  ------                       --------------   -----  \n",
      " 0   Unnamed: 0                   466285 non-null  int64  \n",
      " 1   id                           466285 non-null  int64  \n",
      " 2   member_id                    466285 non-null  int64  \n",
      " 3   loan_amnt                    466285 non-null  int64  \n",
      " 4   funded_amnt                  466285 non-null  int64  \n",
      " 5   funded_amnt_inv              466285 non-null  float64\n",
      " 6   term                         466285 non-null  object \n",
      " 7   int_rate                     466285 non-null  float64\n",
      " 8   installment                  466285 non-null  float64\n",
      " 9   grade                        466285 non-null  object \n",
      " 10  sub_grade                    466285 non-null  object \n",
      " 11  emp_title                    438697 non-null  object \n",
      " 12  emp_length                   445277 non-null  object \n",
      " 13  home_ownership               466285 non-null  object \n",
      " 14  annual_inc                   466281 non-null  float64\n",
      " 15  verification_status          466285 non-null  object \n",
      " 16  issue_d                      466285 non-null  object \n",
      " 17  loan_status                  466285 non-null  object \n",
      " 18  pymnt_plan                   466285 non-null  object \n",
      " 19  url                          466285 non-null  object \n",
      " 20  desc                         125983 non-null  object \n",
      " 21  purpose                      466285 non-null  object \n",
      " 22  title                        466265 non-null  object \n",
      " 23  zip_code                     466285 non-null  object \n",
      " 24  addr_state                   466285 non-null  object \n",
      " 25  dti                          466285 non-null  float64\n",
      " 26  delinq_2yrs                  466256 non-null  float64\n",
      " 27  earliest_cr_line             466256 non-null  object \n",
      " 28  inq_last_6mths               466256 non-null  float64\n",
      " 29  mths_since_last_delinq       215934 non-null  float64\n",
      " 30  mths_since_last_record       62638 non-null   float64\n",
      " 31  open_acc                     466256 non-null  float64\n",
      " 32  pub_rec                      466256 non-null  float64\n",
      " 33  revol_bal                    466285 non-null  int64  \n",
      " 34  revol_util                   465945 non-null  float64\n",
      " 35  total_acc                    466256 non-null  float64\n",
      " 36  initial_list_status          466285 non-null  object \n",
      " 37  out_prncp                    466285 non-null  float64\n",
      " 38  out_prncp_inv                466285 non-null  float64\n",
      " 39  total_pymnt                  466285 non-null  float64\n",
      " 40  total_pymnt_inv              466285 non-null  float64\n",
      " 41  total_rec_prncp              466285 non-null  float64\n",
      " 42  total_rec_int                466285 non-null  float64\n",
      " 43  total_rec_late_fee           466285 non-null  float64\n",
      " 44  recoveries                   466285 non-null  float64\n",
      " 45  collection_recovery_fee      466285 non-null  float64\n",
      " 46  last_pymnt_d                 465909 non-null  object \n",
      " 47  last_pymnt_amnt              466285 non-null  float64\n",
      " 48  next_pymnt_d                 239071 non-null  object \n",
      " 49  last_credit_pull_d           466243 non-null  object \n",
      " 50  collections_12_mths_ex_med   466140 non-null  float64\n",
      " 51  mths_since_last_major_derog  98974 non-null   float64\n",
      " 52  policy_code                  466285 non-null  int64  \n",
      " 53  application_type             466285 non-null  object \n",
      " 54  annual_inc_joint             0 non-null       float64\n",
      " 55  dti_joint                    0 non-null       float64\n",
      " 56  verification_status_joint    0 non-null       float64\n",
      " 57  acc_now_delinq               466256 non-null  float64\n",
      " 58  tot_coll_amt                 396009 non-null  float64\n",
      " 59  tot_cur_bal                  396009 non-null  float64\n",
      " 60  open_acc_6m                  0 non-null       float64\n",
      " 61  open_il_6m                   0 non-null       float64\n",
      " 62  open_il_12m                  0 non-null       float64\n",
      " 63  open_il_24m                  0 non-null       float64\n",
      " 64  mths_since_rcnt_il           0 non-null       float64\n",
      " 65  total_bal_il                 0 non-null       float64\n",
      " 66  il_util                      0 non-null       float64\n",
      " 67  open_rv_12m                  0 non-null       float64\n",
      " 68  open_rv_24m                  0 non-null       float64\n",
      " 69  max_bal_bc                   0 non-null       float64\n",
      " 70  all_util                     0 non-null       float64\n",
      " 71  total_rev_hi_lim             396009 non-null  float64\n",
      " 72  inq_fi                       0 non-null       float64\n",
      " 73  total_cu_tl                  0 non-null       float64\n",
      " 74  inq_last_12m                 0 non-null       float64\n",
      "dtypes: float64(46), int64(7), object(22)\n",
      "memory usage: 266.8+ MB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4bff4a1d-df92-4ae9-96ff-aae6a3c82bde",
   "metadata": {},
   "source": [
    "# EDA"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "16b40f83-6966-464e-bb5c-a8db840ecb57",
   "metadata": {},
   "source": [
    "## Loan based on user purpose"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "5a4326f1-140f-408d-92ae-968d83df83d2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:ylabel='purpose'>"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAdUAAAD4CAYAAAC6/HyrAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjQuMywgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/MnkTPAAAACXBIWXMAAAsTAAALEwEAmpwYAAAt0ElEQVR4nO3de5yVZb338c9XRBBFPOaDWkFKGgKBAp7QQFEzM3XnMU3Jym1ldtjWdm8zMbem5qOVlkUnNXVLnpKdeyuKB0QFBIEBPCu0Q31KUfCICv6eP65rwc0wa2YNrDWLmfm+X695zVr3uu/rvq61xN/ch3V9FRGYmZnZutug3h0wMzPrKFxUzczMqsRF1czMrEpcVM3MzKrERdXMzKxKNqx3B6y+tt566+jTp0+9u2Fm1q7MnDnzlYjYpvFyF9VOrk+fPsyYMaPe3TAza1ck/bWp5T79a2ZmViUuqmZmZlXiotrJzX1hKX3OuoM+Z91R766YmbV7vqZaZ5JGAmdGxGebeG0hMDQiXpH0cETs3cbdM7P10Pvvv8+iRYtYtmxZvbvS4XXv3p0ddtiBrl27VrS+i2o74YJqZiWLFi2iZ8+e9OnTB0n17k6HFREsXryYRYsW0bdv34q28enfdSDp+5LOyI8vl3RvfnyApOskHSTpEUmPSbpJ0qb59U9LelLSFOCfCu1tJWmipFmSfg2o8Nqb+fdISfdLujm3cb3yvypJnym1K+nnkv7Sdu+GmbWVZcuWsdVWW7mg1pgkttpqq1adEXBRXTeTgX3z46HAppK6AiOAucAPgNERsRswA/iupO7Ab4DD8rb/p9DeucCUiBgCTAA+Uma/Q4BvA/2BjwH75HZ/DRwSESOANb4/VSLpVEkzJM1Y8fbS1o/azOrOBbVttPZ9dlFdNzOB3SX1BN4FHiEV132Bd0hF7yFJs4GTgY8CuwALIuKZSLl71xXa26/0PCLuAF4rs9/pEbEoIj4AZgN9crvPR8SCvM5/lut0RIyLiKERMbRLj16tHrSZmTXN11TXQUS8n28m+hLwMNAAjAJ2BBYAd0fE8cVtJA0GmguxrSTg9t3C4xWkz9F/tpp1UtW+e3/hRYdWtb3OxEeq624ycGb+/SBwGunocSrptOxOAJJ6SPo48CTQV9KOefvjG7V1Ql7/EGCLVvTjSeBjkvrk58euzWDMzNYHI0eOrPpsby+++CJHHXVUTffnI9V19yBwNvBIRLwlaRnwYES8LGkM8J+SuuV1fxART0s6FbhD0ivAFGBAfv28vP5jwAPA/1baiYh4R9LXgTtzu9Mr2W7g9r2Y4b9KzawDWb58ORtuuGZ522677bj55ptrum8fqa6jiJgUEV0j4q38/OMRcVl+fG9EDIuIQflnQl5+Z0TsEhEjIuKs0ndUI2JxRBwUEbtFxHci4qMR8Up+bdP8+/7id1oj4vSIuDo/vS8idiFd0+1OujnKzKyqFi5cyC677MJXvvIVBgwYwAknnMA999zDPvvsQ79+/Zg+fTpvvfUWp5xyCsOGDWPIkCHcfvvtAFx99dUcccQRHHbYYfTt25crr7ySyy67jCFDhrDnnnvy6quvrtzPddddx957782AAQOYPj0dJzTX7tFHH81hhx3GQQcdVLbfAwakY5h33nmH4447jkGDBnHsscfyzjvvVOW98ZFqx/JVSScDGwGzSHcDm5lV3bPPPstNN93EuHHjGDZsGDfccANTpkxhwoQJXHjhhfTv35/999+f3//+9yxZsoThw4czevRoAObNm8esWbNYtmwZO+20ExdffDGzZs3iO9/5Dtdeey3f/va3gVRAH374YSZPnswpp5zCvHnzuOCCC8q2+8gjj9DQ0MCWW27ZYv+vuuoqevToQUNDAw0NDey2225VeV9cVDuQiLgcuLze/TCzjq9v374MHDgQgF133ZUDDjgASQwcOJCFCxeyaNEiJkyYwKWXXgqk79b+7/+mK1qjRo2iZ8+e9OzZk169enHYYYcBMHDgQBoaGlbu4/jj0y0n++23H6+//jpLlixh4sSJZds98MADKyqoAJMnT+aMM84AYNCgQQwaNGhd3xLARdXMzNZCt27dVj7eYIMNVj7fYIMNWL58OV26dOGWW25h5513Xm27adOmtbhtSePviEoiIsq2u8kmm7RqDLX4rq+LqplZO7c+fgXm4IMP5oorruCKK65AErNmzWLIkCGtamP8+PGMGjWKKVOm0KtXL3r16lWVdiEd/V5//fWMGjWKefPmrXaEvC58o5KZmVXdOeecw/vvv8+gQYMYMGAA55xzTqvb2GKLLdh777057bTT+N3vfle1dgG+9rWv8eabbzJo0CAuueQShg8fvlbtNKY0qY91VkOHDo1qfxfMzGrriSee4BOf+ES9u9FpNPV+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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "df.groupby(['purpose']).count()[['member_id']].plot(kind='barh')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a45467df-9f4d-405d-8d5e-754340053bdb",
   "metadata": {},
   "source": [
    "From bar chart above, we see that the majority of the loan purpose is to to a debt consolidation. Based on investopedia debt consolidation refers to the act of taking out a new loan to pay off other liabilities and consumer debt. Debt consolidation quite identic with lower interest rate, lower paymont, or even both. So if we want to compete with other banks we need to set a number of interest as proportional as possible while we have to still maintaining the low NPL (and that's why we create a model in this project)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "9df2a4d7-6f46-40a2-b578-acd49d72b90c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:ylabel='Frequency'>"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAYsAAAD4CAYAAAAdIcpQAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjQuMywgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/MnkTPAAAACXBIWXMAAAsTAAALEwEAmpwYAAAXgUlEQVR4nO3dfbRddX3n8feHgAlgRB4CExPahJoFhhAbDBmELkZEBLUInRnGsKyyCpbR0kGcmTVCHYrzR1zoUNqKA5UWRlCQRyuZ6TjlobUslpo0GAzhqcSSwjUZSGEoygAS/M4fZwcO4Sb73Jt77j3hvl9rnXX2/p3f3r/v3UnuJ/vh7J2qQpKk7dlloguQJA0+w0KS1MqwkCS1MiwkSa0MC0lSq10nuoB+2W+//WrOnDkTXYYk7VTuueeef6yqGVu3v2HDYs6cOaxatWqiy5CknUqSfxiu3cNQkqRWhoUkqZVhIUlq9YY9ZyFp5/LSSy8xNDTECy+8MNGlTArTpk1j9uzZ7Lbbbj31NywkDYShoSGmT5/OnDlzSDLR5byhVRVPPfUUQ0NDzJ07t6dlPAwlaSC88MIL7LvvvgbFOEjCvvvuO6K9OMNC0sAwKMbPSLe1YSFJauU5C0kDac55fzGm61t/0YfGdH2TjWExjLH+S9or/zJLb3zvec97uPjii1m8ePGYrXPDhg2cc8453HzzzX0bz8NQkrST2Lx587Dtb3vb24YNirFkWEhSY/369RxyyCF84hOfYMGCBXz0ox/ljjvu4Oijj2bevHmsXLmS5557jjPOOIMjjjiCRYsWceuttwLwta99jVNOOYWTTjqJuXPn8pWvfIVLLrmERYsWceSRR/L000+/Ms43vvENjjrqKBYsWMDKlSsBtrveU089lZNOOon3v//926x7wYIFADz//PMsXbqUhQsX8pGPfITnn39+TLaNh6Ekqcu6deu46aabuOKKKzjiiCO47rrruPvuu1m+fDlf+MIXmD9/Pu9973u56qqreOaZZ1iyZAnve9/7AFi7di2rV6/mhRde4O1vfztf/OIXWb16NZ/5zGe45pprOPfcc4FOMHzve9/jrrvu4owzzmDt2rUsW7Zsm+v9/ve/z5o1a9hnn31a67/88svZY489WLNmDWvWrOHwww8fk+1iWEhSl7lz53LYYYcBcOihh3LccceRhMMOO4z169czNDTE8uXLufjii4HO90Mee+wxAI499limT5/O9OnT2WuvvTjppJMAOOyww1izZs0rY5x22mkAHHPMMTz77LM888wz3Hbbbdtc7/HHH99TUADcddddnHPOOQAsXLiQhQsX7ugmAQwLSXqNqVOnvjK9yy67vDK/yy67sHnzZqZMmcItt9zCwQcf/JrlVqxY0brsFlt/xyEJVbXN9e65554j+hn68X0Vw0LSQBrUqwNPOOEELr30Ui699FKSsHr1ahYtWjSiddxwww0ce+yx3H333ey1117stddeY7Je6OytXHvttRx77LGsXbv2NXs0O8IT3JI0AhdccAEvvfQSCxcuZMGCBVxwwQUjXsfee+/NUUcdxSc/+UmuvPLKMVsvwKc+9Sl+9rOfsXDhQr70pS+xZMmSUa1na6mqMVnRoFm8eHGN9kl5fs9CGn8PPvgg73jHOya6jElluG2e5J6qet2XMtyzkCS18pyFJO0k7rvvPj72sY+9pm3q1KmsWLGi72MbFpIGRlV559ntOOyww7j33nvHZF0jPQXhYShJA2HatGk89dRTI/4lppHb8vCjadOm9byMexaSBsLs2bMZGhpi06ZNE13KpLDlsaq9MiwkDYTddtut50d8avx5GEqS1MqwkCS16ltYJLkqyZNJ1na1/dckDyVZk+TPk7y167Pzk6xL8nCSE7ra35XkvuazL8dLJSRp3PVzz+JrwIlbtd0OLKiqhcDfAecDJJkPLAUObZa5LMmUZpnLgbOAec1r63VKkvqsb2FRVXcBT2/VdltVbbn14g+ALafiTwaur6oXq+pRYB2wJMlM4C1V9f3qXE93DXBKv2qWJA1vIs9ZnAF8p5meBTze9dlQ0zarmd66fVhJzkqyKskqL7+TpLEzIWGR5HPAZuDaLU3DdKvttA+rqq6oqsVVtXjGjBk7XqgkCZiA71kkOR34deC4evWrmkPAgV3dZgMbmvbZw7RLksbRuO5ZJDkR+Czw4ar6f10fLQeWJpmaZC6dE9krq2oj8NMkRzZXQX0cuHU8a5Yk9XHPIsk3gfcA+yUZAi6kc/XTVOD25grYH1TVJ6vq/iQ3Ag/QOTx1dlW93KzqU3SurNqdzjmO7yBJGld9C4uqOm2Y5iu3038ZsGyY9lXAgjEsTZI0Qn6DW5LUyrCQJLUyLCRJrQwLSVIrw0KS1MqwkCS1MiwkSa0MC0lSK8NCktTKsJAktTIsJEmtDAtJUivDQpLUyrCQJLUyLCRJrQwLSVIrw0KS1MqwkCS1MiwkSa0MC0lSK8NCktTKsJAktepbWCS5KsmTSdZ2te2T5PYkjzTve3d9dn6SdUkeTnJCV/u7ktzXfPblJOlXzZKk4fVzz+JrwIlbtZ0H3FlV84A7m3mSzAeWAoc2y1yWZEqzzOXAWcC85rX1OiVJfda3sKiqu4Cnt2o+Gbi6mb4aOKWr/fqqerGqHgXWAUuSzATeUlXfr6oCrulaRpI0Tsb7nMUBVbURoHnfv2mfBTze1W+oaZvVTG/dPqwkZyVZlWTVpk2bxrRwSZrMBuUE93DnIWo77cOqqiuqanFVLZ4xY8aYFSdJk914h8UTzaElmvcnm/Yh4MCufrOBDU377GHaJUnjaLzDYjlwejN9OnBrV/vSJFOTzKVzIntlc6jqp0mObK6C+njXMpKkcbJrv1ac5JvAe4D9kgwBFwIXATcmORN4DDgVoKruT3Ij8ACwGTi7ql5uVvUpOldW7Q58p3lJksZR38Kiqk7bxkfHbaP/MmDZMO2rgAVjWJokaYQG5QS3JGmAGRaSpFaGhSSplWEhSWplWEiSWhkWkqRWhoUkqZVhIUlqZVhIkloZFpKkVoaFJKmVYSFJamVYSJJaGRaSpFaGhSSplWEhSWplWEiSWhkWkqRWPYVFEh9rKkmTWK97Fn+SZGWS30ny1n4WJEkaPD2FRVX9GvBR4EBgVZLrkhzf18okSQOj53MWVfUI8J+BzwL/AvhykoeS/Mt+FSdJGgy9nrNYmOQPgQeB9wInVdU7muk/HOmgST6T5P4ka5N8M8m0JPskuT3JI8373l39z0+yLsnDSU4Y6XiSpB3T657FV4AfAu+sqrOr6ocAVbWBzt5Gz5LMAs4BFlfVAmAKsBQ4D7izquYBdzbzJJnffH4ocCJwWZIpIxlTkrRjeg2LDwLXVdXzAEl2SbIHQFV9fRTj7grsnmRXYA9gA3AycHXz+dXAKc30ycD1VfViVT0KrAOWjGJMSdIo9RoWdwC7d83v0bSNWFX9BLgYeAzYCPxTVd0GHFBVG5s+G4H9m0VmAY93rWKoaXudJGclWZVk1aZNm0ZTniRpGL2GxbSq+tmWmWZ6j9EM2JyLOBmYC7wN2DPJb25vkWHaariOVXVFVS2uqsUzZswYTXmSpGH0GhbPJTl8y0ySdwHPj3LM9wGPVtWmqnoJ+BZwFPBEkpnN+mcCTzb9h+hcsrvFbDqHrSRJ42TXHvudC9yUZMsv6ZnAR0Y55mPAkc05j+eB44BVwHPA6cBFzfutTf/lwHVJLqGzJzIPWDnKsSVJo9BTWFTV3yY5BDiYzmGhh5q9ghGrqhVJbqZzddVmYDVwBfBm4MYkZ9IJlFOb/vcnuRF4oOl/dlW9PJqxJUmj0+ueBcARwJxmmUVJqKprRjNoVV0IXLhV84t09jKG678MWDaasSRJO66nsEjydeBXgHuBLf+rL2BUYSFJ2rn0umexGJhfVcNehSRJemPr9WqotcA/62chkqTB1euexX7AA0lW0jm3AEBVfbgvVUmSBkqvYfH5fhYhSRpsvV46+zdJfhmYV1V3NN+R8GZ+kjRJ9HqL8t8Gbga+2jTNAr7dp5okSQOm1xPcZwNHA8/CKw9C2n+7S0iS3jB6DYsXq+rnW2aaW4t7Ga0kTRK9hsXfJPk9Os+gOB64Cfgf/StLkjRIeg2L84BNwH3AvwX+FyN8Qp4kaefV69VQvwD+tHlJkiaZXu8N9SjDnKOoqoPGvCJJ0sAZyb2htphG5/bh+4x9OZKkQdTTOYuqeqrr9ZOq+iPgvf0tTZI0KHo9DHV41+wudPY0pvelIknSwOn1MNQfdE1vBtYD/2bMq5EkDaRer4Y6tt+FSJIGV6+Hof799j6vqkvGphxJ0iAaydVQRwDLm/mTgLuAx/tRlCRpsIzk4UeHV9VPAZJ8Hripqj7Rr8IkSYOj19t9/BLw8675nwNzxrwaSdJA6jUsvg6sTPL5JBcCK4BrRjtokrcmuTnJQ0keTPLuJPskuT3JI8373l39z0+yLsnDSU4Y7biSpNHp9Ut5y4DfAv4v8AzwW1X1hR0Y94+B/11VhwDvBB6kc7PCO6tqHnBnM0+S+cBS4FDgROCyJD6lT5LGUa97FgB7AM9W1R8DQ0nmjmbAJG8BjgGuBKiqn1fVM8DJwNVNt6uBU5rpk4Hrq+rFqnoUWAcsGc3YkqTR6fWxqhcCnwXOb5p2A74xyjEPonO78/+eZHWSP0uyJ3BAVW0EaN63PIlvFq+96mqoaRuuzrOSrEqyatOmTaMsT5K0tV73LH4D+DDwHEBVbWD0t/vYFTgcuLyqFjXrPG87/TNM27BP6auqK6pqcVUtnjFjxijLkyRtrdew+HlVFc0v6WZPYLSGgKGqWtHM30wnPJ5IMrNZ/0zgya7+B3YtPxvYsAPjS5JGqNewuDHJV4G3Jvlt4A5G+SCkqvo/wONJDm6ajgMeoPOFv9ObttOBW5vp5cDSJFOb8yTzgJWjGVuSNDqtX8pLEuAG4BDgWeBg4Per6vYdGPffAdcmeRPw93SutNqFTiidCTxG55kZVNX9SW6kEyibgbOr6uUdGFuSNEKtYVFVleTbVfUuYEcConud9/LaByptcdw2+i8Dlo3F2JKkkev1MNQPkhzR10okSQOr13tDHQt8Msl6Olcvhc5Ox8J+FSZJGhzbDYskv1RVjwEfGKd6JEkDqG3P4tt07jb7D0luqap/NQ41SZIGTNs5i+4vxB3Uz0IkSYOrLSxqG9OSpEmk7TDUO5M8S2cPY/dmGl49wf2WvlYnSRoI2w2LqvJW4JKkEd2iXJI0SRkWkqRWhoUkqZVhIUlqZVhIkloZFpKkVoaFJKmVYSFJamVYSJJaGRaSpFaGhSSplWEhSWplWEiSWhkWkqRWExYWSaYkWZ3kfzbz+yS5PckjzfveXX3PT7IuycNJTpiomiVpsprIPYtPAw92zZ8H3FlV84A7m3mSzAeWAocCJwKXJfE5G5I0jiYkLJLMBj4E/FlX88nA1c301cApXe3XV9WLVfUosA5YMk6lSpKYuD2LPwL+E/CLrrYDqmojQPO+f9M+C3i8q99Q0/Y6Sc5KsirJqk2bNo150ZI0WY17WCT5deDJqrqn10WGaavhOlbVFVW1uKoWz5gxY9Q1SpJea7vP4O6To4EPJ/kgMA14S5JvAE8kmVlVG5PMBJ5s+g8BB3YtPxvYMK4VS9IkN+57FlV1flXNrqo5dE5c/1VV/SawHDi96XY6cGszvRxYmmRqkrnAPGDlOJctSZPaROxZbMtFwI1JzgQeA04FqKr7k9wIPABsBs6uqpcnrkxJmnwmNCyq6rvAd5vpp4DjttFvGbBs3AqTJL2G3+CWJLUyLCRJrQwLSVIrw0KS1MqwkCS1MiwkSa0MC0lSK8NCktTKsJAktTIsJEmtDAtJUivDQpLUyrCQJLUyLCRJrQwLSVIrw0KS1MqwkCS1MiwkSa0MC0lSK8NCktTKsJAktTIsJEmtxj0skhyY5K+TPJjk/iSfbtr3SXJ7kkea9727ljk/ybokDyc5YbxrlqTJbiL2LDYD/6Gq3gEcCZydZD5wHnBnVc0D7mzmaT5bChwKnAhclmTKBNQtSZPWuIdFVW2sqh820z8FHgRmAScDVzfdrgZOaaZPBq6vqher6lFgHbBkXIuWpEluQs9ZJJkDLAJWAAdU1UboBAqwf9NtFvB412JDTZskaZxMWFgkeTNwC3BuVT27va7DtNU21nlWklVJVm3atGksypQkMUFhkWQ3OkFxbVV9q2l+IsnM5vOZwJNN+xBwYNfis4ENw623qq6oqsVVtXjGjBn9KV6SJqGJuBoqwJXAg1V1SddHy4HTm+nTgVu72pcmmZpkLjAPWDle9UqSYNcJGPNo4GPAfUnubdp+D7gIuDHJmcBjwKkAVXV/khuBB+hcSXV2Vb087lVL0iQ27mFRVXcz/HkIgOO2scwyYFnfipIkbZff4JYktTIsJEmtDAtJUivDQpLUyrCQJLUyLCRJrQwLSVIrw0KS1MqwkCS1MiwkSa0MC0lSK8NCktTKsJAktTIsJEmtDAtJUivDQpLUyrCQJLUyLCRJrQwLSVIrw0KS1MqwkCS1MiwkSa0MC0lSq50mLJKcmOThJOuSnDfR9UjSZLJThEWSKcB/Az4AzAdOSzJ/YquSpMlj14kuoEdLgHVV9fcASa4HTgYemNCqxtic8/5iokvQOFh/0YcmuoRxN1F/tyfjtu6XnSUsZgGPd80PAf98605JzgLOamZ/luThUY63H/CPo1x2Iln3+BpV3fliHyoZmUmzvQdgW8POt71/ebjGnSUsMkxbva6h6grgih0eLFlVVYt3dD3jzbrHl3WPL+ueWDvFOQs6exIHds3PBjZMUC2SNOnsLGHxt8C8JHOTvAlYCiyf4JokadLYKQ5DVdXmJL8L/CUwBbiqqu7v45A7fChrglj3+LLu8WXdEyhVrzv0L0nSa+wsh6EkSRPIsJAktTIsugziLUWSrE9yX5J7k6xq2vZJcnuSR5r3vbv6n9/U/3CSE7ra39WsZ12SLycZ7nLkHanzqiRPJlnb1TZmdSaZmuSGpn1Fkjl9rPvzSX7SbPN7k3xwAOs+MMlfJ3kwyf1JPt20D/Q2307dA73Nk0xLsjLJj5q6/0vTPtDbe0xVla/OeZspwI+Bg4A3AT8C5g9AXeuB/bZq+xJwXjN9HvDFZnp+U/dUYG7z80xpPlsJvJvOd1a+A3xgjOs8BjgcWNuPOoHfAf6kmV4K3NDHuj8P/Mdh+g5S3TOBw5vp6cDfNfUN9DbfTt0Dvc2bMd7cTO8GrACOHPTtPZavCS9gUF7NH95fds2fD5w/AHWt5/Vh8TAws5meCTw8XM10rh57d9Pnoa7204Cv9qHWObz2l+6Y1bmlTzO9K51vxKZPdW/rF9dA1b1VbbcCx+8s23yYuneabQ7sAfyQzl0kdqrtvSMvD0O9arhbisyaoFq6FXBbknvSuZ0JwAFVtRGged+/ad/WzzCrmd66vd/Gss5XlqmqzcA/Afv2rXL43SRrmsNUWw4tDGTdzeGKRXT+t7vTbPOt6oYB3+ZJpiS5F3gSuL2qdqrtvaMMi1f1dEuRCXB0VR1O5467Zyc5Zjt9t/UzDNrPNpo6x/NnuBz4FeBXgY3AH7TUMGF1J3kzcAtwblU9u72u26hjQmofpu6B3+ZV9XJV/SqdO0gsSbJgO90Hpu6xYli8aiBvKVJVG5r3J4E/p3MH3ieSzARo3p9sum/rZxhqprdu77exrPOVZZLsCuwFPN2PoqvqieYXwy+AP6WzzQeu7iS70fmFe21VfatpHvhtPlzdO8s2b2p9BvgucCI7wfYeK4bFqwbuliJJ9kwyfcs08H5gbVPX6U230+kc96VpX9pcVTEXmAesbHaPf5rkyObKi493LdNPY1ln97r+NfBX1RzcHWtb/vE3foPONh+ouptxrgQerKpLuj4a6G2+rboHfZsnmZHkrc307sD7gIcY8O09pib6pMkgvYAP0rk648fA5wagnoPoXFHxI+D+LTXROY55J/BI875P1zKfa+p/mK4rnoDFdP4B/hj4CmN/ovKbdA4fvETnf0hnjmWdwDTgJmAdnatJDupj3V8H7gPW0PkHPHMA6/41Ooco1gD3Nq8PDvo2307dA73NgYXA6qa+tcDvj/W/xX79XRmrl7f7kCS18jCUJKmVYSFJamVYSJJaGRaSpFaGhSSplWEhSWplWEiSWv1/3y5LQP8UuEUAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "df.groupby(['loan_amnt']).count()[['member_id']].plot(kind='hist')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b4602801-a061-4ae5-9eba-7fad29cf4438",
   "metadata": {},
   "source": [
    "From graph above and below we can see that the majority of loan amount is below than 5,000 (idk what currency is this lol) but we assume that the number is multiplied by 1,000 so the amount of loan will make sense. This number also can we utilzed as next promo with combination of interest rate in this range of amount of loan. So user acquisition would be increase too"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "3144f95a-7118-498d-804d-4c5712f2c541",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:>"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "df.groupby(['loan_amnt']).count()[['member_id']].plot(kind='box')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ef1dc158-18d3-42d9-8fe2-296d4954c137",
   "metadata": {
    "tags": []
   },
   "source": [
    "# Pre-processing"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b211bd2d-609e-423d-8cf6-39559622bc8b",
   "metadata": {
    "tags": []
   },
   "source": [
    "## Convert label data to boolean"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3c710c0b-e320-42a1-97c0-68a10c604e78",
   "metadata": {},
   "source": [
    "In the first process of pre-processing I exclude loan that has loan_status current since the status is neutral, neither a good one, nor the bad one"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "d67da31a-3fe7-4ae4-9db1-b2533d209320",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:ylabel='loan_status'>"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "df.groupby(['loan_status']).count()[['member_id']].plot(kind='barh')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "1792e799-734b-4538-a3b6-580025207111",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_clean = df[df['loan_status']!='Current']"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9c711ab9-b9cf-443d-bfba-6ea13505841e",
   "metadata": {},
   "source": [
    "To make 'the machine' learn easily to create a model, we have to change the label into boolean. 1 categorized as fully paid or a good loan, others (late, charged off, dll) categorized as 0 or not a good loan"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "61b36351-22dc-4e62-9d49-b8d46c599ed2",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_clean['is_quality_loan_good'] = np.where(df_clean['loan_status']=='Fully Paid', 1, 0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "adeb1d8f-7078-4f0c-91ab-fd2b62b4a337",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<AxesSubplot:ylabel='loan_status'>"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "df_clean.groupby(['loan_status']).count()[['member_id']].plot(kind='barh')"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e6c892dd-8b33-4af3-afd8-205f24d197c7",
   "metadata": {},
   "source": [
    "## Missing value "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "526a2874-c4a6-4509-8e87-ef6e17e3e083",
   "metadata": {},
   "outputs": [],
   "source": [
    "def missing_values_table(df):\n",
    "        # Total missing values\n",
    "        mis_val = df.isnull().sum()\n",
    "        \n",
    "        # Percentage of missing values\n",
    "        mis_val_percent = 100 * df.isnull().sum() / len(df)\n",
    "        \n",
    "        # Make a table with the results\n",
    "        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)\n",
    "        \n",
    "        # Rename the columns\n",
    "        mis_val_table_ren_columns = mis_val_table.rename(\n",
    "        columns = {0 : 'Missing Values', 1 : '% of Total Values'})\n",
    "        \n",
    "        # Sort the table by percentage of missing descending\n",
    "        mis_val_table_ren_columns = mis_val_table_ren_columns[\n",
    "            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(\n",
    "        '% of Total Values', ascending=False).round(1)\n",
    "        \n",
    "        # Print some summary information\n",
    "        print (\"Your selected dataframe has \" + str(df.shape[1]) + \" columns.\\n\"      \n",
    "            \"There are \" + str(mis_val_table_ren_columns.shape[0]) +\n",
    "              \" columns that have missing values.\")\n",
    "        \n",
    "        # Return the dataframe with missing information\n",
    "        return mis_val_table_ren_columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "4f9a43c0-ae5f-4038-b87e-495be9c4a95b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Your selected dataframe has 75 columns.\n",
      "There are 40 columns that have missing values.\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Missing Values</th>\n",
       "      <th>% of Total Values</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>dti_joint</th>\n",
       "      <td>466285</td>\n",
       "      <td>100.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>annual_inc_joint</th>\n",
       "      <td>466285</td>\n",
       "      <td>100.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>total_cu_tl</th>\n",
       "      <td>466285</td>\n",
       "      <td>100.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>inq_fi</th>\n",
       "      <td>466285</td>\n",
       "      <td>100.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>all_util</th>\n",
       "      <td>466285</td>\n",
       "      <td>100.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max_bal_bc</th>\n",
       "      <td>466285</td>\n",
       "      <td>100.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>open_rv_24m</th>\n",
       "      <td>466285</td>\n",
       "      <td>100.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>open_rv_12m</th>\n",
       "      <td>466285</td>\n",
       "      <td>100.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>il_util</th>\n",
       "      <td>466285</td>\n",
       "      <td>100.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>total_bal_il</th>\n",
       "      <td>466285</td>\n",
       "      <td>100.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mths_since_rcnt_il</th>\n",
       "      <td>466285</td>\n",
       "      <td>100.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>open_il_24m</th>\n",
       "      <td>466285</td>\n",
       "      <td>100.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>open_il_12m</th>\n",
       "      <td>466285</td>\n",
       "      <td>100.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>open_il_6m</th>\n",
       "      <td>466285</td>\n",
       "      <td>100.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>open_acc_6m</th>\n",
       "      <td>466285</td>\n",
       "      <td>100.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>verification_status_joint</th>\n",
       "      <td>466285</td>\n",
       "      <td>100.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>inq_last_12m</th>\n",
       "      <td>466285</td>\n",
       "      <td>100.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mths_since_last_record</th>\n",
       "      <td>403647</td>\n",
       "      <td>86.6</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mths_since_last_major_derog</th>\n",
       "      <td>367311</td>\n",
       "      <td>78.8</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>desc</th>\n",
       "      <td>340302</td>\n",
       "      <td>73.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mths_since_last_delinq</th>\n",
       "      <td>250351</td>\n",
       "      <td>53.7</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>next_pymnt_d</th>\n",
       "      <td>227214</td>\n",
       "      <td>48.7</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>total_rev_hi_lim</th>\n",
       "      <td>70276</td>\n",
       "      <td>15.1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>tot_coll_amt</th>\n",
       "      <td>70276</td>\n",
       "      <td>15.1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>tot_cur_bal</th>\n",
       "      <td>70276</td>\n",
       "      <td>15.1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>emp_title</th>\n",
       "      <td>27588</td>\n",
       "      <td>5.9</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>emp_length</th>\n",
       "      <td>21008</td>\n",
       "      <td>4.5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>last_pymnt_d</th>\n",
       "      <td>376</td>\n",
       "      <td>0.1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>revol_util</th>\n",
       "      <td>340</td>\n",
       "      <td>0.1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>collections_12_mths_ex_med</th>\n",
       "      <td>145</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>last_credit_pull_d</th>\n",
       "      <td>42</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>open_acc</th>\n",
       "      <td>29</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>total_acc</th>\n",
       "      <td>29</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>inq_last_6mths</th>\n",
       "      <td>29</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>earliest_cr_line</th>\n",
       "      <td>29</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>delinq_2yrs</th>\n",
       "      <td>29</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>acc_now_delinq</th>\n",
       "      <td>29</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>pub_rec</th>\n",
       "      <td>29</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>title</th>\n",
       "      <td>20</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>annual_inc</th>\n",
       "      <td>4</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                             Missing Values  % of Total Values\n",
       "dti_joint                            466285              100.0\n",
       "annual_inc_joint                     466285              100.0\n",
       "total_cu_tl                          466285              100.0\n",
       "inq_fi                               466285              100.0\n",
       "all_util                             466285              100.0\n",
       "max_bal_bc                           466285              100.0\n",
       "open_rv_24m                          466285              100.0\n",
       "open_rv_12m                          466285              100.0\n",
       "il_util                              466285              100.0\n",
       "total_bal_il                         466285              100.0\n",
       "mths_since_rcnt_il                   466285              100.0\n",
       "open_il_24m                          466285              100.0\n",
       "open_il_12m                          466285              100.0\n",
       "open_il_6m                           466285              100.0\n",
       "open_acc_6m                          466285              100.0\n",
       "verification_status_joint            466285              100.0\n",
       "inq_last_12m                         466285              100.0\n",
       "mths_since_last_record               403647               86.6\n",
       "mths_since_last_major_derog          367311               78.8\n",
       "desc                                 340302               73.0\n",
       "mths_since_last_delinq               250351               53.7\n",
       "next_pymnt_d                         227214               48.7\n",
       "total_rev_hi_lim                      70276               15.1\n",
       "tot_coll_amt                          70276               15.1\n",
       "tot_cur_bal                           70276               15.1\n",
       "emp_title                             27588                5.9\n",
       "emp_length                            21008                4.5\n",
       "last_pymnt_d                            376                0.1\n",
       "revol_util                              340                0.1\n",
       "collections_12_mths_ex_med              145                0.0\n",
       "last_credit_pull_d                       42                0.0\n",
       "open_acc                                 29                0.0\n",
       "total_acc                                29                0.0\n",
       "inq_last_6mths                           29                0.0\n",
       "earliest_cr_line                         29                0.0\n",
       "delinq_2yrs                              29                0.0\n",
       "acc_now_delinq                           29                0.0\n",
       "pub_rec                                  29                0.0\n",
       "title                                    20                0.0\n",
       "annual_inc                                4                0.0"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "missing_values = missing_values_table(df)\n",
    "missing_values.head(75)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5e94a11c-b6c2-49cf-a713-f6bd93b0cb7f",
   "metadata": {},
   "source": [
    "## Define and clean dataset from missing value"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f6935f24-a463-416a-800d-8080fcb54508",
   "metadata": {},
   "source": [
    "all_missing_value below contains columns that have exactly 100% missing value. So we need to exclude them in order to make sure we use the right columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "0bf1c21c-e94e-45b4-ad59-438097bae9be",
   "metadata": {
    "tags": []
   },
   "outputs": [],
   "source": [
    "all_missing_value = ['policy_code','annual_inc_joint','dti_joint','verification_status_joint', \n",
    "                                      'open_acc_6m', 'open_il_6m', 'open_il_12m', 'open_il_24m', 'mths_since_rcnt_il', \n",
    "                                      'total_bal_il', 'il_util', 'open_rv_12m', 'open_rv_24m', 'max_bal_bc', 'all_util', \n",
    "                                      'inq_fi', 'total_cu_tl', 'inq_last_12m']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "bb787dc2-d79f-4be8-8ce0-94f96cdc03c3",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_final = df_clean.drop(columns=all_missing_value)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3610468a-c52e-459e-933c-170ff5371ffe",
   "metadata": {},
   "source": [
    "## See the top 5 correlation between features and label"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "69cfbabd-2038-43bb-83b3-4e8ac91d1371",
   "metadata": {},
   "source": [
    "To make sure that we use the right columns, we can see the top 5 positive and negative pearson correlation between features and the label. Also the correlation can be translated into:  \n",
    "\n",
    "00-.19 -> “very weak”  \n",
    "20-.39 -> “weak”  \n",
    "40-.59 -> “moderate”  \n",
    "60-.79 -> “strong”  \n",
    "80-1.0 -> “very strong”\n",
    "\n",
    "source: http://www.statstutor.ac.uk/resources/uploaded/pearsons.pdf "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "c3e7bd8e-81cc-480b-97fd-cd036b183d80",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Most Positive Correlations:\n",
      " total_pymnt             0.357109\n",
      "total_pymnt_inv         0.358951\n",
      "last_pymnt_amnt         0.413463\n",
      "total_rec_prncp         0.470304\n",
      "is_quality_loan_good    1.000000\n",
      "Name: is_quality_loan_good, dtype: float64\n",
      "\n",
      "Most Negative Correlations:\n",
      " recoveries                -0.389417\n",
      "out_prncp                 -0.333436\n",
      "out_prncp_inv             -0.333420\n",
      "collection_recovery_fee   -0.262526\n",
      "int_rate                  -0.255863\n",
      "Name: is_quality_loan_good, dtype: float64\n"
     ]
    }
   ],
   "source": [
    "correlations = df_final.corr()['is_quality_loan_good'].sort_values()\n",
    "\n",
    "# Display correlations\n",
    "print('Most Positive Correlations:\\n', correlations.tail(5))\n",
    "print('\\nMost Negative Correlations:\\n', correlations.head(5))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3abef90f-21b5-4eff-96e8-22488d531261",
   "metadata": {},
   "source": [
    "# Modeling"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d5f24e82-2020-43a1-9dd6-25473ca26b7d",
   "metadata": {},
   "source": [
    "## First trial"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e71b953d-f024-4d30-b973-729b6b27c0b1",
   "metadata": {},
   "source": [
    "Model training can be very long in terms of process-time if we use very-very big data. So, we have to use as proportional as possible using columns that highly affecting the score of the model. Therefore, in the first trial, I use four features which are the highest positive correlation to the label."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "0215dc3e-69c5-4708-92d9-4d93077ea200",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Split Feature and Label\n",
    "X = df_final[['last_pymnt_amnt','total_rec_prncp','total_pymnt','total_pymnt_inv']]\n",
    "y = df_final['is_quality_loan_good'] # target / label\n",
    "\n",
    "#Splitting the data into Train and Test\n",
    "from sklearn.model_selection import train_test_split \n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "9a0008e2-1c4f-4875-89e7-0da8bef32e8a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "LogisticRegression(random_state=42)"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.linear_model import LogisticRegression\n",
    "model = LogisticRegression(random_state=42)\n",
    "model.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "ddaf217a-ea48-4f41-a9c3-99da4dff4f1e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1, 1, 0, ..., 1, 1, 1])"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_pred = model.predict(X_test)\n",
    "y_pred"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "078d0b9d-3534-4fb2-89a2-be51b7458b7d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy (Test Set): 0.95\n",
      "Precision (Test Set): 0.96\n",
      "Recall (Test Set): 0.97\n",
      "F1-Score (Test Set): 0.97\n"
     ]
    }
   ],
   "source": [
    "eval_classification(model, y_pred, X_train, y_train, X_test, y_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e93ec70a-d60f-4e03-8b54-b0c79fc117ad",
   "metadata": {},
   "source": [
    "Since our main metric is precision, we can see the precision score is a good one. But if the number translated into NPL score, it can be up to 4% of default risk. If we want to compete with other banks in terms of NPL. we have to increase the number by adding the features which have highest negative correlation in the second trial"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "79a368b5-a497-4073-8679-6af3bcc75902",
   "metadata": {},
   "source": [
    "## First trial - train vs test score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "15e668dc-c064-45b7-b474-e2eefebd6f80",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Train score: 0.9513754050082329\n",
      "Test score:0.9506320746922251\n"
     ]
    }
   ],
   "source": [
    "print('Train score: ' + str(model.score(X_train, y_train))) #accuracy\n",
    "print('Test score:' + str(model.score(X_test, y_test))) #accuracy"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "38990844-2f7d-4bb4-a4d4-e2edc7d894a7",
   "metadata": {},
   "source": [
    "## Second trial"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "8a34bc84-8f80-45f1-ad31-9bef0cab0e86",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Split Feature and Label\n",
    "X = df_final[['last_pymnt_amnt','total_rec_prncp','total_pymnt','total_pymnt_inv','out_prncp','out_prncp_inv']]\n",
    "y = df_final['is_quality_loan_good'] # target / label\n",
    "\n",
    "#Splitting the data into Train and Test\n",
    "from sklearn.model_selection import train_test_split \n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "d439bcfa-3084-4032-b23d-0ff9086e3015",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "LogisticRegression(random_state=42)"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.linear_model import LogisticRegression\n",
    "model = LogisticRegression(random_state=42)\n",
    "model.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "20818995-4d53-4c4b-8f2f-2faaa1b3bd69",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([1, 1, 0, ..., 1, 1, 1])"
      ]
     },
     "execution_count": 26,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_pred = model.predict(X_test)\n",
    "y_pred"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "f1d611c0-cbaf-41db-9927-7b7bc9479894",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy (Test Set): 0.96\n",
      "Precision (Test Set): 0.97\n",
      "Recall (Test Set): 0.98\n",
      "F1-Score (Test Set): 0.98\n"
     ]
    }
   ],
   "source": [
    "eval_classification(model, y_pred, X_train, y_train, X_test, y_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d45c93b1-bb72-47a6-93dc-bf3b052a674a",
   "metadata": {},
   "source": [
    "By adding two more features in second trial, we get a high score for all of the metrics (including the precision). So I think this score is a good number and if we translate into NPL metric also can be categorized as good number (only 2.7% - 3%) where the limit of good NPL is 5%"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b1059632-edbd-47d6-8bdc-b5b9a773cfb1",
   "metadata": {},
   "source": [
    "## Second trial - train vs test score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "39290f8a-9066-4177-ac01-e1ef2a6ef45b",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Train score: 0.9625120248346032\n",
      "Test score:0.9620755184664959\n"
     ]
    }
   ],
   "source": [
    "print('Train score: ' + str(model.score(X_train, y_train))) #accuracy\n",
    "print('Test score:' + str(model.score(X_test, y_test))) #accuracy"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d8e5566d-4129-4422-ae2c-a2939f626de3",
   "metadata": {},
   "source": [
    "# Business insights"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "179bac62-8291-4c22-a910-11db037b498b",
   "metadata": {},
   "source": [
    "## Conclusion"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "542ccb5d-f3b0-4ee1-97e9-ddc6c87122a5",
   "metadata": {},
   "source": [
    "Business insights - Conclusion\n",
    "1. Metric for model -> Precision & Metric for business -> NPL  \n",
    "For this credit loan model, we use Precision as our main model metrics. Why we have to use Precision? It's simply because of nature of lending business strongly related with non-performing loan (NPL) metrics that shows number of credit default (in indonesia we usually say gagal bayar). Precision emphasize on small amount of false positive, we predict positive but in reality negative, that in this case can be translated we predict the label is a good loan but in reality it could be turn to default, late, and so on  \n",
    "2. Model score (precision) -> 97% & NPL -> 2.91%  \n",
    "It can be translated that this model has an opportunity to predict incorrectly about 2.91% which can be translated to an NPL about 2.91%. This number is a good one if we compared to NPL of biggest bank in Indonesia* such as BNI (3.81%), BRI (3.28%), CIMB (3.35%), etc  \n",
    "  \n",
    "  \n",
    "https://www.cnbcindonesia.com/market/20211029140854-17-287550/laba-5-bank-kakap-q3-melesat-duh-angka-npl-bikin-waswas-nih"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5ec64f96-1051-4655-8373-6468a36cb4ff",
   "metadata": {},
   "source": [
    "## Confusion matrix "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "d9df0741-fb10-4b0b-b39c-9486c5a87d34",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[15732,  1625],\n",
       "       [ 1129, 54132]], dtype=int64)"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.metrics import confusion_matrix\n",
    "\n",
    "confusion_matrix(y_test, y_pred)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "12a59f79-8292-422a-858b-30cd0e3707df",
   "metadata": {},
   "source": [
    "# My profile"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "13ec4ae4-e890-4271-8e27-da00ac543574",
   "metadata": {},
   "source": [
    "Name: Muhammad Ghifariyadi (Ghifar)  \n",
    "Linkedin: https://www.linkedin.com/in/ghifariyadi/"
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
   "version": "3.9.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
