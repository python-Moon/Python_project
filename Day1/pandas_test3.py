import pandas as pd

csv_path = 'calc_case_description_test_set.csv'
df =pd.read_csv(csv_path)
all_count = df.shape[0]git

print(df)

malignant_df = df[df['pathology'] == 'MALIGNANT'][['patient_id', 'pathology']]
malignant_count = malignant_df.shape[0]
print(malignant_df)

print(all_count, malignant_count)

malignant_df.to_csv('malignant_data.csv')
