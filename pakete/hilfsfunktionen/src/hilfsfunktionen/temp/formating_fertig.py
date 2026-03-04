import pandas as pd
from .db import read_table_to_df, get_engine

engine = get_engine()
df_customers = read_table_to_df("SELECT * FROM bank_customers;", engine=engine)

def text_to_tokens(text):
    return text.split()

def create_n_sentences(table,n):
    textified_data = table.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    sentences = textified_data.apply(lambda text: (' ' + text) * n)
    return [text_to_tokens(text) for text in sentences]

def get_unchanged_table(table = df_customers):
    return table.copy()

def get_formated_table(table = df_customers):  
    df_formated = table.assign(
        customer_id=table['customer_id'], #Nicht nötig, wenn churn propensity gesucht ist.
        credit_score=format_credit_scores(table['credit_score']),
        country=table['country'],
        gender=table['gender'],
        age=format_ages(table['age']),
        tenure=format_tenures(table['tenure']),
        balance=format_balances(table['balance']),
        products_number=table['products_number'],
        credit_card=table['credit_card'],
        active_member=table['active_member'],
        estimated_salary=format_salaries(table['estimated_salary']),
        churn=table['churn']
    )


    df_formated['gender'] = 'Gender_' + df_formated['gender']
    df_formated['country'] = 'Country_' + df_formated['country']
    df_formated['products_number'] = 'ProductsNumber_' + df_formated['products_number'].astype(str)
    df_formated['credit_card'] = df_formated['credit_card'].map({1:'CreditCard_Yes', 0:'CreditCard_No'})
    df_formated['active_member'] = df_formated['active_member'].map({1:'ActiveMember_Yes', 0:'ActiveMember_No'})
    df_formated['churn'] = df_formated['churn'].map({1:'Churn_Yes', 0:'Churn_No'})
    
    df_formated = df_formated.drop(columns=['customer_id']) #Nicht nötig, wenn churn propensity gesucht ist.
    return df_formated

def get_table_with_keys(table, interval = 2):
    df_with_keys = table.copy()
    cols = list(df_with_keys.columns)
    new_cols = []
    key_counter = 1

    for i, col in enumerate(cols, start=1):
        if (i - 1) % interval == 0:
            key_name = f'key{key_counter}'
            df_with_keys[key_name] = ['key_' + str(j+1) for j in range(len(table))]
            new_cols.append(key_name)
            key_counter += 1
        new_cols.append(col)

    df_with_keys = df_with_keys[new_cols]
    return df_with_keys

def format_balances(balances):
    labels = [f"Balance_Cluster_{i}" for i in range(1, 11)]
    return pd.cut(balances, bins=10, labels=labels)

def format_ages(ages):
    bins = [17, 30, 45, 55, 65, 100]
    labels = [
        'Age_Young_Adults',
        'Age_Adults_in_their_Prime',
        'Age_Middle_aged',
        'Age_Pre_retirees',
        'Age_Young_Seniors'
    ]
    return pd.cut(ages, bins=bins, labels=labels, right=False)

def format_salaries(salaries):
    bins = [0, 1000, 10000, 30000, 60000, 100000, 150000, 200000]
    labels = [
        'Salary_Very_low',
        'Salary_Low',
        'Salary_Below_average',
        'Salary_Average',
        'Salary_Above_average',
        'Salary_High',
        'Salary_Very_high'
    ]
    return pd.cut(salaries, bins=bins, labels=labels, right=False)

def format_credit_scores(credit_scores):
    bins = [0, 580, 670, 740, 800, 851]
    labels = [
        'Credit_Score_Poor',
        'Credit_Score_Fair',
        'Credit_Score_Good',
        'Credit_Score_Very_Good',
        'Credit_Score_Excellent'
    ]
    return pd.cut(credit_scores, bins=bins, labels=labels, right=False)

def format_tenures(tenures):
    return "Tenure_" + tenures.astype(str)
