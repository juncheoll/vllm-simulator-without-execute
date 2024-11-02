import matplotlib.pyplot as plt
import pandas as pd
import re

file_path = "./QA-experience/opt-1.3B/8k request/LRU, watermark=0.1"
with open(file_path, 'r') as file:
    log_data = file.read()

pattern = re.compile(
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

data1 = []
for match in re.findall(pattern, log_data):
    prompt_throughput, generation_throughput, running, swapped, pending, gpu_cache, cpu_cache = match
    data1.append([float(prompt_throughput), float(generation_throughput), int(running), int(swapped), int(pending), float(gpu_cache), float(cpu_cache)])

data2 = []
for match in re.findall(pattern2, log_data):
    prefix_cache_hit_gpu, prefix_cache_hit_cpu = match
    data2.append([float(prefix_cache_hit_gpu), float(prefix_cache_hit_cpu)])

columns_data1 = ['Avg Prompt Throughput', 'Avg Generation Throughput', 
                 'Running', 'Swapped', 'Pending', 'GPU KV Cache Usage', 'CPU KV Cache Usage']
df_data1 = pd.DataFrame(data1, columns=columns_data1)

columns_data2 = ['Prefix Cache Hit GPU', 'Prefix Cache Hit CPU']
df_data2 = pd.DataFrame(data2, columns=columns_data2)


# 그래프 그리기
plt.figure(figsize=(12, 8))

# 첫 번째 subplot: Throughput over time
ax1 = plt.subplot(3, 1, 1)

# Throughput 그래프만 그리기
ax1.plot(df_data1.index, df_data1['Avg Prompt Throughput'], label='Avg Prompt Throughput', linestyle='-')
ax1.plot(df_data1.index, df_data1['Avg Generation Throughput'], label='Avg Generation Throughput', linestyle='-')
ax1.set_ylabel('Throughput (tokens/s)')
ax1.set_title('Throughput over Time')
ax1.legend()

ax1.set_ylim(0, 1500)

# 두 번째 subplot: Cache usage over time with Num Preemption Change
ax2 = plt.subplot(3, 1, 2)

# Cache usage 그래프
ax2.plot(df_data1.index, df_data1['GPU KV Cache Usage'], label='GPU KV Cache Usage', linestyle='-')
ax2.plot(df_data1.index, df_data1['CPU KV Cache Usage'], label='CPU KV Cache Usage', linestyle='-')
ax2.set_ylabel('Cache Usage (%)')
ax2.set_title('KV Cache Usage and Num Preemption Change over Time')

# 범례 처리
ax2.legend(loc='upper left')
#ax3.legend(loc='upper right')

ax2.set_ylim(0, 100)


# 세 번째 subplot: Prefix cache hit rate
plt.subplot(3, 1, 3)
plt.plot(df_data2.index, df_data2['Prefix Cache Hit GPU'], label='Prefix Cache Hit GPU', marker='o')
plt.plot(df_data2.index, df_data2['Prefix Cache Hit CPU'], label='Prefix Cache Hit CPU', marker='o')
plt.ylabel('Cache Hit Rate (%)')
plt.xlabel('Time (Log Entries)')
plt.title('Prefix Cache Hit Rate over Time')
plt.legend()

plt.ylim(0, 70)

plt.tight_layout()
plt.show()

