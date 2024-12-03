import pandas as pd
import numpy as np

output_arr = []
gun_text_df = pd.read_csv("Output_Text_GunControl.csv")
gun_image_df = pd.read_csv("Output_Image_GunControl.csv")

gun_image_df['0'] = gun_image_df['0'].str.replace('.jpg', '', regex=False)



for _, row in gun_image_df.iterrows():
    for _, row_in in gun_text_df.iterrows():
        if row.iloc[0] == row_in.iloc[7]:
            output_arr.append([row_in.iloc[0],row_in.iloc[1],row_in.iloc[2],row_in.iloc[3],row_in.iloc[4],row_in.iloc[5],row_in.iloc[6],row.iloc[1],row.iloc[2],row.iloc[3],row.iloc[4],row.iloc[5]])

output_df = pd.DataFrame(output_arr, columns=["Text Question 1","Text Question 2","Text Question 3","Text Question 4","Text Question 5","Text Question 6","Text Question 7","Image Question 1","Image Question 2","Image Question 3","Image Question 4","Image Question 5",])

output_df.to_csv("gun_train_data.csv")