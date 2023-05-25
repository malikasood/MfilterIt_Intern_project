import pandas as pd

df = pd.read_excel("C:/Users/Malika Sood/Desktop/Intern/Mfilterit/Nissan Review Sentiment Analysis.xlsx")
final_df = df['Review']

temp_df = df.head(9)
temp_df['Review'] = temp_df['Review'].apply(lambda x: str(x))
final_df = pd.merge(final_df, temp_df, on = ['Review'], how = 'left')

final_df.to_csv("review2.csv")




    