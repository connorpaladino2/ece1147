from run_text import run_text
from run_image import run_image
import pandas as pd
import os
from PIL import Image

def do_gun_tweet_train():
    input_df = pd.read_csv("data/data-20241202T145651Z-001/data/gun_control_train.csv")

    j = 0
    output_arr = []

    for _, row in input_df.iterrows():
        arr = []
        text = row.iloc[2]
        text = ''.join(char for char in text if 32 <= ord(char) <= 126)
        j += 1
        print(f"Row {j}: \"{text}\"")

        output = run_text(f"{row.iloc[2]}")
    
        arr.append(int(row.iloc[0]))

        if row.iloc[3] == "oppose":
            arr.append(0.0)
        else:
            arr.append(1.0)

            
        if row.iloc[4] == "no":
            arr.append(0.0)
        else:
            arr.append(1.0)

        i = 0
        for line in output.strip().split("\n"):
            if "Question" in line:  # Look for lines containing "Question"
                try:
                    answer = float(line.split(":")[1].strip())
                    arr.append(answer)
                except (IndexError, ValueError):
                    print(f"Skipping malformed line: {line}")
                i += 1

        print(arr)
        output_arr.append(arr)

    print(output_arr)

    output_df = pd.DataFrame(output_arr)
    output_df.to_csv("Output_Text_GunControl.csv")


def do_gun_tweet_test():
    input_df = pd.read_csv("data/data-20241202T145651Z-001/data/gun_control_dev.csv")

    j = 0
    output_arr = []

    for _, row in input_df.iterrows():
        arr = []
        text = row.iloc[2]
        text = ''.join(char for char in text if 32 <= ord(char) <= 126)
        j += 1
        print(f"Row {j}: \"{text}\"")

        output = run_text(f"{row.iloc[2]}")

        arr.append(int(row.iloc[0]))

        if row.iloc[3] == "oppose":
            arr.append(0.0)
        else:
            arr.append(1.0)

            
        if row.iloc[4] == "no":
            arr.append(0.0)
        else:
            arr.append(1.0)

        i = 0
        for line in output.strip().split("\n"):
            if "Question" in line:  # Look for lines containing "Question"
                try:
                    answer = float(line.split(":")[1].strip())
                    arr.append(answer)
                except (IndexError, ValueError):
                    print(f"Skipping malformed line: {line}")
                i += 1

        print(arr)
        output_arr.append(arr)

    print(output_arr)

    output_df = pd.DataFrame(output_arr)
    output_df.to_csv("Output_Text_GunControl_Test.csv")



def do_abortion_tweet_train():
    input_df = pd.read_csv("data/data-20241202T145651Z-001/data/abortion_train.csv")

    j = 0
    output_arr = []

    for _, row in input_df.iterrows():
        arr = []
        text = row.iloc[2]
        text = ''.join(char for char in text if 32 <= ord(char) <= 126)
        j += 1
        print(f"Row {j}: \"{text}\"")

        output = run_text(f"{row.iloc[2]}")

        arr.append(int(row.iloc[0]))

        if row.iloc[3] == "oppose":
            arr.append(0.0)
        else:
            arr.append(1.0)

            
        if row.iloc[4] == "no":
            arr.append(0.0)
        else:
            arr.append(1.0)

        i = 0
        for line in output.strip().split("\n"):
            if "Question" in line:  # Look for lines containing "Question"
                try:
                    answer = float(line.split(":")[1].strip())
                    arr.append(answer)
                except (IndexError, ValueError):
                    print(f"Skipping malformed line: {line}")
                i += 1

        print(arr)
        output_arr.append(arr)

    print(output_arr)

    output_df = pd.DataFrame(output_arr)
    output_df.to_csv("Output_Text_Abortion.csv")


def do_abortion_tweet_test():
    input_df = pd.read_csv("data/data-20241202T145651Z-001/data/abortion_dev.csv")

    j = 0
    output_arr = []

    for _, row in input_df.iterrows():
        arr = []
        text = row.iloc[2]
        text = ''.join(char for char in text if 32 <= ord(char) <= 126)
        j += 1
        print(f"Row {j}: \"{text}\"")

        output = run_text(f"{row.iloc[2]}")

        arr.append(int(row.iloc[0]))

        if row.iloc[3] == "oppose":
            arr.append(0.0)
        else:
            arr.append(1.0)

            
        if row.iloc[4] == "no":
            arr.append(0.0)
        else:
            arr.append(1.0)

        i = 0
        for line in output.strip().split("\n"):
            if "Question" in line:  # Look for lines containing "Question"
                try:
                    answer = float(line.split(":")[1].strip())
                    arr.append(answer)
                except (IndexError, ValueError):
                    print(f"Skipping malformed line: {line}")
                i += 1

        print(arr)
        output_arr.append(arr)

    print(output_arr)

    output_df = pd.DataFrame(output_arr)
    output_df.to_csv("Output_Text_Abortion_Test.csv")


def do_gun_images():
    input = "data/data-20241202T145651Z-001/data/images/gun_control"

    file_names = os.listdir(input)

    j = 0
    output_arr = []

    for img in file_names:
        arr = []
        j += 1
        print(f"Row {j}: \"{img}\"")

        try:
            with Image.open("data/data-20241202T145651Z-001/data/images/gun_control/" + img) as Img:
                Img.verify()  # Verify image integrity
        except Exception as e:
            print(e)
            arr = [0.0, 0.0, 0.0, 0.0, 0.0]
            output_arr.append([img] + arr)
            continue
    
        output = run_image(img)

        print(output)

        i = 0
        for line in output.strip().split("\n"):
            if "Question" in line:  # Look for lines containing "Question"
                try:
                    answer = float(line.split(":")[1].strip())
                    arr.append(answer)
                except (IndexError, ValueError):
                    print(f"Skipping malformed line: {line}")
                i += 1

        if arr == []:
            arr = [0.0, 0.0, 0.0, 0.0, 0.0]
        print([img] + arr)
        output_arr.append([img] + arr)

    print(output_arr)

    output_df = pd.DataFrame(output_arr)
    output_df.to_csv("Output_Image_GunControl.csv")
    


#do_gun_tweet_train()
do_gun_tweet_test()
#do_abortion_tweet_train()
#do_abortion_tweet_test()
#do_gun_images()