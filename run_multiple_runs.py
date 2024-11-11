import subprocess
import torch
import os
print(torch.cuda.is_available())
print(torch.version.cuda)

NUM_RUNS = 2
ARGUMENTS = ["FF", "5", "results4", "50"]

for i in range(NUM_RUNS):
    print(f" Executing Run {i+1} out of {NUM_RUNS}")
    #subprocess.run(["python", "main.py"] + ARGUMENTS)
    os.system(f"python main.py FF 2 results4 50")
