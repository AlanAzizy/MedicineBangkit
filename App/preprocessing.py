import pandas as pd
from function import standardize_output, split_row_by_column


#Load Data
df_1 = pd.read_excel('./App/Medicine_Data.xlsx',0)
df_2 = pd.read_excel('./App/Medicine_Data.xlsx',1)
df_3 = pd.read_excel('./App/Medicine_Data.xlsx',2)
df = pd.concat([df_1,df_2,df_3])

df['Summary'] = df['Summary'].astype('str')
df['Category'] = df['Category'].astype('str')

#Data Cleaning
word_1 = 'Global:'
df['Category'] = df['Category'].str.replace(f'\\b{word_1}\\b', '', regex=True, case=False)
word_2 = 'Anatomical:'
df['Category'] = df['Category'].str.replace(f'\\b{word_2}\\b', '', regex=True, case=False)

df['Summary'] = df['Summary'].apply(standardize_output)

df1 = split_row_by_column(df, 'Category', '\n').reset_index()
df1 = split_row_by_column(df1, 'Category', ';').reset_index()
df2 = df1
df1 = df1[['Summary','Category','Disease','Drug']]

df1 = df1.drop_duplicates(keep='first')


#Data Filtering
unik_groupby = df1.groupby('Category').count().reset_index()
print(unik_groupby)
unik_satu = unik_groupby[unik_groupby['Summary']>100]
drugNameFiltered = unik_satu['Category'].to_numpy()
df1 = df1[df1['Category'].isin(drugNameFiltered)]

print(df1['Category'].unique())
df1 = df1.loc[df1['Category'].isin(['Respiratory diseases', 'Skin diseases', 'Gastrointestinal diseases'])]

#Save Data
filename = './App/training_data.xlsx'
df1.to_excel(filename, index=False)

