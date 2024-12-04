import pandas as pd
import numpy as np

output_arr = []
gun_text_df = pd.read_csv("Output_Text_GunControl.csv")
gun_image_df = pd.read_csv("Output_Image_GunControl.csv")
data_df = pd.read_csv("data/data-20241202T145651Z-001/data/gun_control_train.csv")


gun_image_df['0'] = gun_image_df['0'].str.replace('.jpg', '', regex=False)

gun_text_df['tweet_id'] = gun_text_df['tweet_id'].astype('int64')
data_df['tweet_id'] = data_df['tweet_id'].astype('int64')

print(gun_text_df)
print(data_df)

for _, row_in in gun_text_df.iterrows():
    flag = False
    working_arr = []
    for _, row in gun_image_df.iterrows():
        if str(row.iloc[0])[:5] == str(int(row_in.iloc[7]))[:5]:
            working_arr += [row_in.iloc[0],row_in.iloc[1],row_in.iloc[2],row_in.iloc[3],row_in.iloc[4],row_in.iloc[5],row_in.iloc[6],row.iloc[1],row.iloc[2],row.iloc[3],row.iloc[4],row.iloc[5]]
            flag = True
            break
    if not flag:
        working_arr += [row_in.iloc[0],row_in.iloc[1],row_in.iloc[2],row_in.iloc[3],row_in.iloc[4],row_in.iloc[5],row_in.iloc[6],0.0,0.0,0.0,0.0,0.0]

    flag = False
    for _, row in data_df.iterrows():
        if str(int(row.iloc[0]))[:4] == str(int(row_in.iloc[7]))[:4]:
            
            if row.iloc[3] == "oppose":
                working_arr += [0.0]
            else:
                working_arr += [1.0]

            
            if row.iloc[4] == "no":
                working_arr += [0.0]
            else:
                working_arr += [1.0]
            
            flag = True
            break

    if not flag:
                working_arr += [0.0, 0.0]

    output_arr.append(working_arr)
    

output_df = pd.DataFrame(output_arr, columns=["Text Question 1","Text Question 2","Text Question 3","Text Question 4","Text Question 5","Text Question 6","Text Question 7","Image Question 1","Image Question 2","Image Question 3","Image Question 4","Image Question 5","Support?","Pursuasive?"])

output_df.to_csv("gun_train_data.csv", index=False)