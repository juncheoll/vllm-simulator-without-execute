import matplotlib.pyplot as plt
import pandas as pd
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
)

data = []
for match in re.findall(pattern3, log_data):
    step_iter, step_time, schedule_time, execute_time, schedule_per_step = match
    data.append([int(step_iter), float(step_time), float(schedule_time), float(execute_time), float(schedule_per_step)])

# 데이터프레임 생성
columns = ['Step Iter', 'Step Time', 'Schedule Time', 'Execute Time', 'Schedule Per Step (%)']
df = pd.DataFrame(data, columns=columns)

step_ranges = [(0, 1000), (1000, 2000), (2000, 3000), (3000, 4000), (4000, 5000)]
dfs = []

for start, end in step_ranges:
    df_part = df[(df['Step Iter'] >= start) & (df['Step Iter'] < end)]
    dfs.append(df_part)

# 그래프 그리기
plt.figure(figsize=(12, 10))

for i, df_part in enumerate(dfs):
    # 첫 번째 subplot: Step Time, Schedule Time, Execute Time 그래프
    ax = plt.subplot(len(dfs), 1, i+1)
    ax.plot(df_part['Step Iter'], df_part['Step Time'], label='Step Time', linestyle='-', marker='o')
    ax.plot(df_part['Step Iter'], df_part['Schedule Time'], label='Schedule Time', linestyle='-', marker='o')
    ax.plot(df_part['Step Iter'], df_part['Execute Time'], label='Execute Time', linestyle='-', marker='o')
    ax.set_ylabel('Time (s)')
    ax.set_title('Step Time, Schedule Time, and Execute Time over Steps')
    ax.set_ylim(0, 0.5)
    #ax.yaxis.set_major_locator(plt.MaxNLocator(10))
    ax.legend()

'''
# 두 번째 subplot: Schedule Per Step 그래프
ax2 = plt.subplot(3, 1, 2)
ax2.plot(df['Step Iter'], df['Schedule Per Step (%)'], label='Schedule Per Step (%)', linestyle='-', marker='x')
ax2.set_ylabel('Schedule Per Step (%)')
ax2.set_title('Schedule Per Step over Steps')
ax2.legend()
'''
plt.tight_layout()
plt.show()
