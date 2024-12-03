from run_text import run_text
import pandas as pd


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