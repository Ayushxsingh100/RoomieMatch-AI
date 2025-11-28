import pandas as pd
import random
import numpy as np
import os

# CONFIGURATION
NUM_STUDENTS = 200
FILENAME = "hostel_users_clustered.csv"

# NAMES DATABASE
first_names = ["Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Sai", "Reyansh", "Ayaan", "Krishna", "Ishaan", "Diya", "Saanvi", "Ananya", "Aadhya", "Pari", "Kiara", "Naira", "Myra", "Riya", "Meera"]
last_names = ["Sharma", "Verma", "Gupta", "Malhotra", "Singh", "Kumar", "Mehta", "Reddy", "Patel", "Joshi"]
majors = ["CSE", "ECE", "Mech", "Civil", "IT", "AI&DS"]

data = []
print(f"🔄 Generating {NUM_STUDENTS} student profiles...")

for i in range(NUM_STUDENTS):
    name = f"{random.choice(first_names)} {random.choice(last_names)}"
    major = random.choice(majors)
    contact = f"+91 {random.randint(6000000000, 9999999999)}"
    
    # Create Hidden Clusters
    cluster_type = random.choice([0, 1, 2, 3])
    
    if cluster_type == 0:   # Scholars
        sleep, clean, social, noise = random.randint(1, 3), random.randint(7, 10), random.randint(1, 3), random.randint(1, 3)
    elif cluster_type == 1: # Party Animals
        sleep, clean, social, noise = random.randint(8, 10), random.randint(1, 5), random.randint(8, 10), random.randint(7, 10)
    elif cluster_type == 2: # Athletes
        sleep, clean, social, noise = random.randint(1, 4), random.randint(6, 9), random.randint(7, 9), random.randint(4, 7)
    elif cluster_type == 3: # Night Owls
        sleep, clean, social, noise = random.randint(9, 10), random.randint(3, 7), random.randint(1, 4), random.randint(1, 5)

    # Add noise
    sleep = int(np.clip(sleep + random.uniform(-0.5, 0.5), 1, 10))
    
    data.append([i+1000, name, major, contact, sleep, clean, social, noise])

# Save
df = pd.DataFrame(data, columns=['ID', 'Name', 'Major', 'Contact', 'Sleep', 'Cleanliness', 'Social', 'Noise'])
df.to_csv(FILENAME, index=False)

# VERIFICATION
if os.path.exists(FILENAME):
    print(f"✅ Success! Created '{FILENAME}' with {len(df)} rows.")
    print(f"📂 Location: {os.getcwd()}/{FILENAME}")
else:
    print("❌ Error: File was not created.")