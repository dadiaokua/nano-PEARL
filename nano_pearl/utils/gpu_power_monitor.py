"""
GPU Power Monitor for nano-PEARL
监控 GPU 功耗并计算总能耗（使用积分）
"""
import time
import threading
from typing import List, Dict, Optional
from dataclasses import dataclass, field
import numpy as np

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    print("Warning: pynvml not available. GPU power monitoring will be disabled.")
    print("Install with: pip install nvidia-ml-py3")


@dataclass
class GPUPowerSample:
    """单次 GPU 功耗采样"""
    timestamp: float
    gpu_id: int
    power_draw: float  # 瓦特 (W)
    temperature: float  # 摄氏度 (°C)
    utilization: float  # GPU 利用率 (%)
    memory_used: float  # 已使用显存 (MB)
    memory_total: float  # 总显存 (MB)


@dataclass
class GPUPowerStats:
    """GPU 功耗统计信息"""
    gpu_id: int
    samples: List[GPUPowerSample] = field(default_factory=list)
    
    @property
    def total_energy(self) -> float:
        """计算总能耗 (焦耳 J)，使用梯形积分"""
        if len(self.samples) < 2:
            return 0.0
        
        energy = 0.0
        for i in range(1, len(self.samples)):
            dt = self.samples[i].timestamp - self.samples[i-1].timestamp
            avg_power = (self.samples[i].power_draw + self.samples[i-1].power_draw) / 2
            energy += avg_power * dt  # 能量 = 功率 × 时间
        
        return energy
    
    @property
    def total_energy_kwh(self) -> float:
        """总能耗 (千瓦时 kWh)"""
        return self.total_energy / 3600000  # 1 kWh = 3600000 J
    
    @property
    def avg_power(self) -> float:
        """平均功耗 (W)"""
        if not self.samples:
            return 0.0
        return np.mean([s.power_draw for s in self.samples])
    
    @property
    def max_power(self) -> float:
        """最大功耗 (W)"""
        if not self.samples:
            return 0.0
        return max(s.power_draw for s in self.samples)
    
    @property
    def avg_temperature(self) -> float:
        """平均温度 (°C)"""
        if not self.samples:
            return 0.0
        return np.mean([s.temperature for s in self.samples])
    
    @property
    def avg_utilization(self) -> float:
        """平均 GPU 利用率 (%)"""
        if not self.samples:
            return 0.0
        return np.mean([s.utilization for s in self.samples])
    
    @property
    def duration(self) -> float:
        """监控持续时间 (秒)"""
        if len(self.samples) < 2:
            return 0.0
        return self.samples[-1].timestamp - self.samples[0].timestamp


class GPUPowerMonitor:
    """GPU 功耗监控器"""
    
    def __init__(self, gpu_ids: Optional[List[int]] = None, sample_interval: float = 0.1):
        """
        初始化 GPU 功耗监控器
        
        Args:
            gpu_ids: 要监控的 GPU ID 列表。如果为 None，则监控所有可用的 GPU
            sample_interval: 采样间隔（秒），默认 0.1 秒（100ms）
        """
        self.sample_interval = sample_interval
        self.gpu_ids = gpu_ids
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.stats: Dict[int, GPUPowerStats] = {}
        
        if not PYNVML_AVAILABLE:
            self.enabled = False
            return
        
        try:
            pynvml.nvmlInit()
            self.enabled = True
            
            # 如果没有指定 GPU ID，则获取所有可用的 GPU
            if self.gpu_ids is None:
                device_count = pynvml.nvmlDeviceGetCount()
                self.gpu_ids = list(range(device_count))
            
            # 初始化统计信息
            for gpu_id in self.gpu_ids:
                self.stats[gpu_id] = GPUPowerStats(gpu_id=gpu_id)
            
            print(f"GPU Power Monitor initialized for GPUs: {self.gpu_ids}")
            
        except Exception as e:
            print(f"Failed to initialize GPU Power Monitor: {e}")
            self.enabled = False
    
    def _sample_gpu(self, gpu_id: int) -> Optional[GPUPowerSample]:
        """采样单个 GPU 的功耗数据"""
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            
            # 获取功耗 (mW -> W)
            power_draw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            
            # 获取温度
            temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            
            # 获取 GPU 利用率
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            
            # 获取显存使用情况
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            memory_used = mem_info.used / (1024 ** 2)  # bytes -> MB
            memory_total = mem_info.total / (1024 ** 2)  # bytes -> MB
            
            return GPUPowerSample(
                timestamp=time.time(),
                gpu_id=gpu_id,
                power_draw=power_draw,
                temperature=temperature,
                utilization=utilization,
                memory_used=memory_used,
                memory_total=memory_total
            )
        except Exception as e:
            print(f"Error sampling GPU {gpu_id}: {e}")
            return None
    
    def _monitor_loop(self):
        """监控循环（在后台线程中运行）"""
        while self.monitoring:
            for gpu_id in self.gpu_ids:
                sample = self._sample_gpu(gpu_id)
                if sample:
                    self.stats[gpu_id].samples.append(sample)
            
            time.sleep(self.sample_interval)
    
    def start(self):
        """开始监控"""
        if not self.enabled:
            print("GPU Power Monitor is disabled")
            return
        
        if self.monitoring:
            print("GPU Power Monitor is already running")
            return
        
        # 清空之前的数据
        for gpu_id in self.gpu_ids:
            self.stats[gpu_id].samples.clear()
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print(f"GPU Power Monitor started (sampling every {self.sample_interval}s)")
    
    def stop(self):
        """停止监控"""
        if not self.enabled:
            return
        
        if not self.monitoring:
            return
        
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        print("GPU Power Monitor stopped")
    
    def get_stats(self) -> Dict[int, GPUPowerStats]:
        """获取统计信息"""
        return self.stats
    
    def print_summary(self):
        """打印功耗摘要"""
        if not self.enabled:
            print("GPU Power Monitor is disabled")
            return
        
        print("\n" + "=" * 80)
        print("GPU Power Consumption Summary")
        print("=" * 80)
        
        total_energy = 0.0
        total_energy_kwh = 0.0
        
        for gpu_id, stats in self.stats.items():
            if not stats.samples:
                print(f"\nGPU {gpu_id}: No data collected")
                continue
            
            total_energy += stats.total_energy
            total_energy_kwh += stats.total_energy_kwh
            
            print(f"\nGPU {gpu_id}:")
            print(f"  Duration:          {stats.duration:.2f} s")
            print(f"  Samples:           {len(stats.samples)}")
            print(f"  Avg Power:         {stats.avg_power:.2f} W")
            print(f"  Max Power:         {stats.max_power:.2f} W")
            print(f"  Total Energy:      {stats.total_energy:.2f} J ({stats.total_energy_kwh:.6f} kWh)")
            print(f"  Avg Temperature:   {stats.avg_temperature:.1f} °C")
            print(f"  Avg GPU Util:      {stats.avg_utilization:.1f} %")
            
            if stats.samples:
                last_sample = stats.samples[-1]
                print(f"  Memory Used:       {last_sample.memory_used:.0f} / {last_sample.memory_total:.0f} MB")
        
        print(f"\n{'─' * 80}")
        print(f"Total Energy (All GPUs): {total_energy:.2f} J ({total_energy_kwh:.6f} kWh)")
        
        # 估算成本（假设电价）
        electricity_cost_per_kwh = 0.15  # 美元/kWh，可根据实际情况调整
        estimated_cost = total_energy_kwh * electricity_cost_per_kwh
        print(f"Estimated Cost:          ${estimated_cost:.6f} (@ ${electricity_cost_per_kwh}/kWh)")
        print("=" * 80 + "\n")
    
    def export_to_csv(self, filepath: str):
        """导出数据到 CSV 文件"""
        import csv
        
        if not self.enabled:
            print("GPU Power Monitor is disabled")
            return
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'gpu_id', 'timestamp', 'power_draw_w', 'temperature_c', 
                'utilization_%', 'memory_used_mb', 'memory_total_mb'
            ])
            
            for gpu_id, stats in self.stats.items():
                for sample in stats.samples:
                    writer.writerow([
                        sample.gpu_id,
                        sample.timestamp,
                        sample.power_draw,
                        sample.temperature,
                        sample.utilization,
                        sample.memory_used,
                        sample.memory_total
                    ])
        
        print(f"Power monitoring data exported to: {filepath}")
    
    def __enter__(self):
        """上下文管理器入口"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop()
        self.print_summary()
    
    def __del__(self):
        """析构函数"""
        if self.enabled and hasattr(self, 'monitoring'):
            self.stop()
            try:
                pynvml.nvmlShutdown()
            except:
                pass

