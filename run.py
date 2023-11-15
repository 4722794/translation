#%%
import subprocess
import multiprocessing
import argparse
from dotenv import load_dotenv
import os
load_dotenv()
instance_id = os.getenv("INSTANCE_ID")
# Add the optional argument
# parser.add_argument("-s", "--stop", help="Stop Aws", action="store_true")
# args = parser.parse_args()

# Define a function to run a script
def run_script(script_path):
    process = subprocess.Popen(["python", script_path])
    process.wait()
    return process.returncode


# Define the paths to the scripts to run
script_paths = ["main.py"]

# # Create a pool of worker processes
# pool = multiprocessing.Pool()
# Run the scripts in parallel
# results = pool.map_async(run_script, script_paths)
# Wait for all scripts to finish
# results.wait()

results = []
for script in script_paths:
    results.append(run_script(script))
print(results)
# Check the return codes
# return_codes = results.get()
# if all(rc == 0 for rc in return_codes):
if all(rc == 0 for rc in results):
    print("All scripts completed successfully")
    subprocess.run(
        f"aws ec2 stop-instances --instance-ids {instance_id} --region us-east-1",
        shell=True,
    )
else:
    print("One or more scripts failed")
    subprocess.run(
        f"aws ec2 stop-instances --instance-ids {instance_id} --region us-east-1",
        shell=True,
    )
