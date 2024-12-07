import pandas as pd
import matplotlib.pyplot as plt

# Đọc file CSV
file_path = "reward_data.csv"  # Thay "1900.csv" bằng đường dẫn đến file CSV của bạn
data = pd.read_csv(file_path)

# Vẽ đồ thị
plt.figure(figsize=(12, 12))

# Subplot 1: Total Reward
plt.subplot(3, 1, 1)
plt.bar(data['Episode'], data['Total Reward'], color='blue', label='Total Reward')
plt.title('Total Reward over Episodes')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.grid(axis='y')  # Chỉ hiển thị lưới theo trục Y
plt.legend()

# Subplot 2: Loss
plt.subplot(3, 1, 2)
plt.bar(data['Episode'], data['Loss'], color='orange', label='Loss')
plt.title('Loss over Episodes')
plt.xlabel('Episode')
plt.ylabel('Loss')
plt.grid(axis='y')  # Chỉ hiển thị lưới theo trục Y
plt.legend()

# Subplot 3: Epsilon
plt.subplot(3, 1, 3)
plt.bar(data['Episode'], data['Epsilon'], color='green', label='Epsilon')
plt.title('Epsilon over Episodes')
plt.xlabel('Episode')
plt.ylabel('Epsilon')
plt.grid(axis='y')  # Chỉ hiển thị lưới theo trục Y
plt.legend()

# Tăng khoảng cách giữa các subplots
plt.tight_layout()

# Hiển thị đồ thị
plt.show()

