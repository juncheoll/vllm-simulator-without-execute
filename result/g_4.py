import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
import re

file_path = "./result.txt"
with open(file_path, 'r') as file:
    log_data = file.read()

pattern = re.compile(
    r"\d{2}-\d{2} \d{2}:\d{2}:\d{2} - "
    r"num_cumulative_preemption:(\d+)\n"
    r"INFO \d{2}-\d{2} \d{2}:\d{2}:\d{2} metrics\.py:\d{3}\] "
    r"Avg prompt throughput: ([\d.]+) tokens/s, "
    r"Avg generation throughput: ([\d.]+) tokens/s, "
    r"Running: (\d+) reqs, "
    r"Swapped: (\d+) reqs, "
    r"Pending: (\d+) reqs, "
    r"GPU KV cache usage: ([\d.]+)%, "
    r"CPU KV cache usage: ([\d.]+)%."
)

pattern2 = re.compile(
    r"INFO \d{2}-\d{2} \d{2}:\d{2}:\d{2} metrics\.py:\d{3}\] "
    r"Prefix cache hit rate: GPU: ([\d.]+)%, CPU: ([\d.]+)%"
)

pattern3 = re.compile(
    r"step iter : (\d+)\n"  # 정수형 iter 값을 매칭
    r"step_time : (\d+\.\d+)\n"  # 소수점을 포함한 step_time
    r"schedule_time : (\d+\.\d+)\n"  # 소수점을 포함한 schedule_time
    r"execute_time : (\d+\.\d+)\n"  # 소수점을 포함한 execute_time
    r"schedule_per_step : (\d+\.\d+)%\n"  # 소수점을 포함한 schedule_per_step 값 (백분율)
    r"running : (\d+)\n"
    r"swapped : (\d+)\n"
    r"waiting : (\d+)\n"
    r"gpu_usage : (\d+\.\d+)\n"
    r"cpu_usage : (\d+\.\d+)\n"
)

data = []
for match in re.findall(pattern3, log_data):
    step_iter, step_time, schedule_time, execute_time, schedule_per_step, running, swapped, waiting, gpu_usage, cpu_usage = match
    data.append([int(step_iter), float(step_time), float(schedule_time), float(execute_time), float(schedule_per_step), int(running), int(swapped), int(waiting), float(gpu_usage), float(cpu_usage)])

# 데이터프레임 생성
columns = ['Step Iter', 'Step Time', 'Schedule Time', 'Execute Time', 'Schedule Per Step (%)', 'Running', 'Swapped', 'Waiting', 'GPU Usage (%)', 'CPU Usage (%)']
df = pd.DataFrame(data, columns=columns)

df['Schedule + Execute Time'] = df['Schedule Time'] + df['Execute Time']

window_size = 100
df['Step Time Avg'] = df['Step Time'].rolling(window=window_size).mean()
df['Schedule Time Avg'] = df['Schedule + Execute Time'].rolling(window=window_size).mean()
df['Execute Time Avg'] = df['Execute Time'].rolling(window=window_size).mean()

# 그래프 그리기
plt.figure(figsize=(12, 15))

# 첫 번째 subplot: Step Time, Schedule Time, Execute Time 그래프
ax1 = plt.subplot(5, 1, 1)
ax1.plot(df['Step Iter'], df['Step Time Avg'], label='Step Time (Avg)', linestyle='-')
ax1.fill_between(df['Step Iter'], df['Step Time Avg'], color='blue', alpha=0.3)
ax1.plot(df['Step Iter'], df['Schedule Time Avg'], label='Schedule Time (Avg)', linestyle='-')
ax1.fill_between(df['Step Iter'], df['Schedule Time Avg'], color='orange', alpha=0.3)
ax1.plot(df['Step Iter'], df['Execute Time Avg'], label='Execute Time (Avg)', linestyle='-')
ax1.fill_between(df['Step Iter'], df['Execute Time Avg'], color='green', alpha=0.3)
ax1.set_ylabel('Time (s)')
ax1.set_title('Step Time, Schedule Time, and Execute Time over Steps')
ax1.set_ylim(0, 0.07)
#ax1.yaxis.set_major_locator(plt.MaxNLocator(10))
ax1.legend()

# 두 번째 subplot: Schedule Per Step 그래프
ax2 = plt.subplot(5, 1, 2)
ax2.plot(df['Step Iter'], df['Schedule Per Step (%)'], label='Schedule Per Step (%)', linestyle='-', marker='x')
ax2.set_ylabel('Schedule Per Step (%)')
ax2.set_title('Schedule Per Step over Steps')
ax2.legend()

# 세 번째 subplot: 각 Time의 변화율 (diff 계산)
df['Step Time Change'] = df['Step Time'].diff().fillna(0)
df['Schedule Time Change'] = df['Schedule Time'].diff().fillna(0)
df['Execute Time Change'] = df['Execute Time'].diff().fillna(0)

ax3 = plt.subplot(5, 1, 3)
#ax3.plot(df['Step Iter'], df['Step Time Change'], label='Step Time Change', linestyle='--', color='r')
ax3.plot(df['Step Iter'], df['Schedule Time Change'], label='Schedule Time Change', linestyle='--', color='g')
ax3.plot(df['Step Iter'], df['Execute Time Change'], label='Execute Time Change', linestyle='--', color='b')
ax3.set_ylabel('Time Change (s)')
ax3.set_title('Time Changes over Steps')
ax3.legend()

# 네 번째 subplot: memory usage 그래프
ax4 = plt.subplot(5, 1, 4)
ax4.plot(df['Step Iter'], df['GPU Usage (%)'], label='GPU Usage (%)', linestyle='--', color='r')
ax4.plot(df['Step Iter'], df['CPU Usage (%)'], label='CPU Usage (%)', linestyle='--', color='g')
ax4.set_ylabel('Memory Usage Per Step (%)')
ax4.set_title('Memory Usage Per Step over Steps')
ax4.legend()


df['Swapped'] += df['Running']
df['Waiting'] += df['Swapped']

# 다섯 번째 subplot: request stat 그래프
ax5 = plt.subplot(5, 1, 5)
ax5.plot(df['Step Iter'], df['Waiting'], label='Waiting', linestyle='-')
ax5.fill_between(df['Step Iter'], df['Waiting'], color='blue', alpha=0.3)
ax5.plot(df['Step Iter'], df['Swapped'], label='Swapped', linestyle='-')
ax5.fill_between(df['Step Iter'], df['Swapped'], color='orange', alpha=0.3)
ax5.plot(df['Step Iter'], df['Running'], label='Running', linestyle='-')
ax5.fill_between(df['Step Iter'], df['Running'], color='green', alpha=0.3)
ax5.set_ylabel('Request Stat Per Step')
ax5.set_title('Request Stat Per Step over Steps')
ax5.legend()

plt.tight_layout()
plt.show()


