import pandas as pd

data = {
    'Name' : ['Alice', 'Bob', 'Charlie'],
    'Age' : [15, 20, 18],
}

print(data)

df = pd.DataFrame(data)

print(df)

df['Occupation'] = ['Engineer', 'Artist', 'Doctor']
print(df)

selected_rows = df[df['Age']<20]
print(selected_rows)
