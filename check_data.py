import os

# 🔤 Replace this list with your actual trained signs
actions = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 
           'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

data_path = 'MP_Data'

for action in actions:
    for seq in range(30):
        path = os.path.join(data_path, action, str(seq))
        if not os.path.exists(path):
            print(f"❌ Missing folder: {action}/{seq}")
        elif len(os.listdir(path)) != 30:
            print(f"⚠️ Incomplete data: {action}/{seq} has only {len(os.listdir(path))} files")
        else:
            print(f"✅ OK: {action}/{seq}")
