import argparse
import numpy as np
import tensorflow as tf
import lzma
import os
import random

from script.config import TLITEConfig, load_config
from script.model import create_tlite_model
from script.prefetcher import TLITEPrefetcher

def parse_args():
    parser = argparse.ArgumentParser(description='Generate prefetch files using T-LITE model')
    parser.add_argument('--model-path', help='Path to model checkpoint', required=True)
    parser.add_argument('--clustering-path', help='Path to clustering information', required=True)
    parser.add_argument('--benchmark', help='Path to benchmark trace', required=True)
    parser.add_argument('--output', help='Path to output prefetch file', required=True)
    parser.add_argument('--config', default='./config/TLITE2debug1.yaml', help='Path to configuration file')
    return parser.parse_args()

def generate_prefetches(model, clustering_info, trace_path, config, output_path):
    """
    Generate prefetch file using T-LITE model with optimized stream processing
    """
    print(f"\n=== 开始生成预取文件 ===")
    print(f"处理轨迹文件: {trace_path}")
    print(f"输出文件: {output_path}")
    
    # 分析聚类信息
    page_to_cluster = clustering_info.get('page_to_cluster', {})
    if page_to_cluster:
        cluster_pages = list(page_to_cluster.keys())
        min_page = min(cluster_pages)
        max_page = max(cluster_pages)
        print(f"\n=== 聚类信息分析 ===")
        print(f"页面ID范围: {min_page:x} - {max_page:x}")
        print(f"聚类数量: {len(set(page_to_cluster.values()))}")
        print(f"映射页面数: {len(page_to_cluster)}")
    
    # 创建动态聚类映射器
    class DynamicClusterMapper:
        def __init__(self, static_mapping, num_clusters):
            self.static_mapping = static_mapping
            self.dynamic_mapping = {}
            self.num_clusters = num_clusters
            self.stats = {'static_hits': 0, 'dynamic_hits': 0, 'new_mappings': 0}
        
        def get_cluster(self, page_id):
            # 尝试静态映射
            if page_id in self.static_mapping:
                self.stats['static_hits'] += 1
                return self.static_mapping[page_id]
            
            # 尝试动态映射
            if page_id in self.dynamic_mapping:
                self.stats['dynamic_hits'] += 1
                return self.dynamic_mapping[page_id]
            
            # 创建新映射
            new_cluster = hash(page_id) % self.num_clusters
            self.dynamic_mapping[page_id] = new_cluster
            self.stats['new_mappings'] += 1
            return new_cluster
        
        def print_stats(self):
            total = sum(self.stats.values())
            if total > 0:
                print("\n映射统计:")
                print(f"- 静态映射命中: {self.stats['static_hits']} ({self.stats['static_hits']/total*100:.2f}%)")
                print(f"- 动态映射命中: {self.stats['dynamic_hits']} ({self.stats['dynamic_hits']/total*100:.2f}%)")
                print(f"- 新建映射数: {self.stats['new_mappings']} ({self.stats['new_mappings']/total*100:.2f}%)")
    
    # 添加流负载均衡监控
    class StreamLoadBalancer:
        def __init__(self, num_streams):
            self.num_streams = num_streams
            self.stream_loads = np.zeros(num_streams, dtype=np.int64)
            self.total_updates = 0
            self.last_rebalance = 0
            self.rebalance_interval = 100000
        
        def update(self, stream_id):
            self.stream_loads[stream_id] += 1
            self.total_updates += 1
            
            if self.total_updates - self.last_rebalance >= self.rebalance_interval:
                self.check_balance()
                self.last_rebalance = self.total_updates
        
        def check_balance(self):
            if self.total_updates == 0:
                return
            
            active_streams = np.where(self.stream_loads > 0)[0]
            if len(active_streams) == 0:
                return
            
            loads = self.stream_loads[active_streams]
            avg_load = np.mean(loads)
            max_load = np.max(loads)
            min_load = np.min(loads)
            imbalance = max_load / min_load if min_load > 0 else float('inf')
            
            if imbalance > 10:
                print(f"\n警告: 检测到严重的负载不均衡")
                print(f"- 最大负载: {max_load}")
                print(f"- 最小负载: {min_load}")
                print(f"- 平均负载: {avg_load:.0f}")
                print(f"- 不均衡度: {imbalance:.2f}x")
                print("考虑调整流分配算法或增加流的数量")
    
    # 优化1：增加流数量并改进流分配
    num_streams = 32
    print(f"使用 {num_streams} 个并行流")
    
    # 优化2：改进流ID计算函数
    def calculate_stream_id(pc, addr):
        """使用改进的哈希函数来计算流ID"""
        # 使用多个特征进行哈希
        pc_low = pc & 0xFFFF
        pc_high = (pc >> 16) & 0xFFFF
        addr_low = addr & 0xFFFF
        addr_high = (addr >> 16) & 0xFFFF
        
        # 使用FNV-1a哈希算法
        hash_value = 2166136261
        for value in [pc_low, pc_high, addr_low, addr_high]:
            hash_value = hash_value ^ value
            hash_value = (hash_value * 16777619) & 0xFFFFFFFF
        
        # 使用Wang哈希进行最终混合
        hash_value = hash_value ^ (hash_value >> 16)
        hash_value = (hash_value * 0x85ebca6b) & 0xFFFFFFFF
        hash_value = hash_value ^ (hash_value >> 13)
        hash_value = (hash_value * 0xc2b2ae35) & 0xFFFFFFFF
        hash_value = hash_value ^ (hash_value >> 16)
        
        return hash_value % num_streams
    
    # 优化3：增加批处理大小并使用内存预分配
    batch_size = 50000  # 增加到50000
    print(f"批处理大小: {batch_size}")
    
    # 优化4：预分配批处理数组以减少内存分配
    max_batch_size = batch_size + 1000  # 添加一些缓冲
    pages_batch = np.zeros(max_batch_size, dtype=np.int64)
    offsets_batch = np.zeros(max_batch_size, dtype=np.int32)
    pcs_batch = np.zeros(max_batch_size, dtype=np.int64)
    inst_ids_batch = np.zeros(max_batch_size, dtype=np.int64)
    stream_ids_batch = np.zeros(max_batch_size, dtype=np.int32)
    
    # 优化5：预分配模型输入数组
    cluster_histories = [[] for _ in range(max_batch_size)]
    offset_histories = [[] for _ in range(max_batch_size)]
    dpf_vectors_batch = [[] for _ in range(max_batch_size)]
    
    # 初始化负载均衡器
    load_balancer = StreamLoadBalancer(num_streams)
    
    # 初始化预取器和映射器
    prefetchers = []
    mappers = []
    for i in range(num_streams):
        prefetcher = TLITEPrefetcher(
            model=model,
            clustering_info=clustering_info,
            config=config
        )
        mapper = DynamicClusterMapper(page_to_cluster, config.num_clusters)
        prefetchers.append(prefetcher)
        mappers.append(mapper)
    
    # 优化6：使用TensorFlow的批处理优化
    @tf.function(experimental_relax_shapes=True, jit_compile=True)
    def predict_batch(cluster_histories, offset_histories, pcs, dpf_vectors):
        return model((cluster_histories, offset_histories, pcs, dpf_vectors))
    
    # 读取文件和计算总行数
    if trace_path.endswith('.txt.xz'):
        f = lzma.open(trace_path, mode='rt', encoding='utf-8')
    else:
        f = open(trace_path, 'r')
    
    print("计算总行数...")
    total_lines = sum(1 for line in f if not (line.startswith('***') or line.startswith('Read')))
    f.seek(0)
    print(f"需处理总行数: {total_lines}")
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 主处理循环
    current_batch_size = 0
    processed_lines = 0
    total_predictions = 0
    valid_prefetches = 0
    stream_stats = {i: {'updates': 0, 'prefetches': 0} for i in range(num_streams)}
    
    with open(output_path, 'w', buffering=1024*1024) as out_f:
        for line in f:
            if line.startswith('***') or line.startswith('Read'):
                continue
            
            # 解析行
            split = line.strip().split(', ')
            inst_id = int(split[0])
            pc = int(split[3], 16)
            addr = int(split[2], 16)
            
            # 计算页面和偏移
            page = (addr >> 6) >> config.offset_bits
            offset = (addr >> 6) & ((1 << config.offset_bits) - 1)
            
            # 使用新的流分配函数
            stream_id = calculate_stream_id(pc, addr)
            
            # 更新流统计
            stream_stats[stream_id]['updates'] += 1
            
            # 更新预分配数组
            pages_batch[current_batch_size] = page
            offsets_batch[current_batch_size] = offset
            pcs_batch[current_batch_size] = pc
            inst_ids_batch[current_batch_size] = inst_id
            stream_ids_batch[current_batch_size] = stream_id
            
            # 更新预取器状态
            prefetcher = prefetchers[stream_id]
            mapper = mappers[stream_id]
            cluster_id = mapper.get_cluster(page)
            prefetcher.update_history(cluster_id, offset, pc)
            
            # 在主循环中更新负载均衡器
            load_balancer.update(stream_id)
            
            current_batch_size += 1
            
            # 处理满批次
            if current_batch_size >= batch_size:
                # 准备模型输入
                valid_indices = []
                for i in range(current_batch_size):
                    stream_id = stream_ids_batch[i]
                    prefetcher = prefetchers[stream_id]
                    
                    cluster_history, offset_history, _, dpf_vectors = prefetcher.metadata_manager.prepare_model_inputs(
                        prefetcher.page_history,
                        prefetcher.offset_history,
                        pcs_batch[i],
                        prefetcher.cluster_mapping_utils
                    )
                    
                    cluster_histories[i] = cluster_history
                    offset_histories[i] = offset_history
                    dpf_vectors_batch[i] = dpf_vectors
                    valid_indices.append(i)
                
                # 批量预测
                if valid_indices:
                    candidate_logits, offset_logits = predict_batch(
                        tf.convert_to_tensor(cluster_histories[:current_batch_size], dtype=tf.int32),
                        tf.convert_to_tensor(offset_histories[:current_batch_size], dtype=tf.int32),
                        tf.convert_to_tensor(pcs_batch[:current_batch_size], dtype=tf.int32),
                        tf.convert_to_tensor(dpf_vectors_batch[:current_batch_size], dtype=tf.float32)
                    )
                    
                    # 处理预测结果（使用向量化操作）
                    candidates = tf.argmax(candidate_logits, axis=1).numpy()
                    offsets = tf.argmax(offset_logits, axis=1).numpy()
                    
                    # 批量生成预取
                    for j, i in enumerate(valid_indices):
                        if candidates[j] != config.num_candidates:
                            stream_id = stream_ids_batch[i]
                            prefetcher = prefetchers[stream_id]
                            candidate_pages = prefetcher.metadata_manager.get_candidate_pages(
                                prefetcher.page_history[-1]
                            )
                            
                            if candidate_pages and candidates[j] < len(candidate_pages):
                                prefetch_page = candidate_pages[candidates[j]][0]
                            else:
                                prefetch_page = pages_batch[i]
                            
                            prefetch_addr = (prefetch_page << (config.offset_bits + 6)) | (offsets[j] << 6)
                            out_f.write(f"{inst_ids_batch[i]} {prefetch_addr:x}\n")
                            stream_stats[stream_id]['prefetches'] += 1
                            valid_prefetches += 1
                    
                    total_predictions += len(valid_indices)
                
                # 重置批处理计数器
                current_batch_size = 0
                processed_lines += batch_size
                
                # 打印进度和流负载分布
                if processed_lines % (batch_size * 10) == 0:
                    print(f"进度: {processed_lines}/{total_lines} ({processed_lines/total_lines*100:.2f}%)")
                    print(f"当前预取率: {(valid_prefetches/total_predictions*100) if total_predictions > 0 else 0.00:.2f}%")
                    
                    # 打印流负载分布
                    if processed_lines % (batch_size * 100) == 0:
                        total_updates = sum(stats['updates'] for stats in stream_stats.values())
                        if total_updates > 0:
                            print("\n=== 流负载分布 ===")
                            # 只显示有活动的流
                            active_streams = [(sid, stats) for sid, stats in stream_stats.items() 
                                           if stats['updates'] > 0]
                            # 按更新次数排序
                            sorted_streams = sorted(active_streams, 
                                                 key=lambda x: x[1]['updates'], 
                                                 reverse=True)
                            # 显示前5个最活跃的流
                            for stream_id, stats in sorted_streams[:5]:
                                print(f"流 {stream_id}: {stats['updates']/total_updates*100:.2f}% "
                                     f"({stats['updates']} 更新)")
                            # 显示负载分布统计
                            active_count = len(active_streams)
                            print(f"\n活跃流数量: {active_count}/{num_streams}")
                            if active_count > 0:
                                avg_load = total_updates / active_count
                                max_load = max(stats['updates'] for _, stats in active_streams)
                                min_load = min(stats['updates'] for _, stats in active_streams)
                                print(f"平均负载: {avg_load:.0f} 更新/流")
                                print(f"负载范围: {min_load} - {max_load} 更新")
                                print(f"最大/最小负载比: {max_load/min_load:.2f}x")
    
    # 最终统计
    print("\n=== 最终统计 ===")
    print(f"总处理行数: {processed_lines}")
    print(f"总预测次数: {total_predictions}")
    print(f"有效预取数: {valid_prefetches}")
    if total_predictions > 0:
        print(f"整体预取率: {valid_prefetches/total_predictions*100:.2f}%")
    else:
        print("整体预取率: 0.00% (无预测)")
    
    # 打印最终流负载分布
    print("\n=== 最终流负载分布 ===")
    total_updates = sum(stats['updates'] for stats in stream_stats.values())
    if total_updates > 0:
        active_streams = [(sid, stats) for sid, stats in stream_stats.items() 
                       if stats['updates'] > 0]
        sorted_streams = sorted(active_streams, 
                             key=lambda x: x[1]['updates'], 
                             reverse=True)
        
        print("\n前10个最活跃的流:")
        for stream_id, stats in sorted_streams[:10]:
            prefetch_rate = (stats['prefetches'] / stats['updates'] * 100 
                           if stats['updates'] > 0 else 0)
            print(f"流 {stream_id}:")
            print(f"  负载: {stats['updates']/total_updates*100:.2f}% ({stats['updates']} 更新)")
            print(f"  预取: {prefetch_rate:.2f}% ({stats['prefetches']} 预取)")
        
        print("\n负载分布统计:")
        active_count = len(active_streams)
        print(f"活跃流数量: {active_count}/{num_streams}")
        if active_count > 0:
            avg_load = total_updates / active_count
            max_load = max(stats['updates'] for _, stats in active_streams)
            min_load = min(stats['updates'] for _, stats in active_streams)
            print(f"平均负载: {avg_load:.0f} 更新/流")
            print(f"负载范围: {min_load} - {max_load} 更新")
            print(f"最大/最小负载比: {max_load/min_load:.2f}x")

def main():
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Load model, ignore warnings and add debug output
    print("\n=== 加载模型 ===")
    model = create_tlite_model(config)
    model.load_weights(args.model_path).expect_partial()
    print(f"模型权重成功加载。模型结构：")
    print(model.summary())
    
    # Load clustering information with debug output
    print("\n=== 加载聚类信息 ===")
    clustering_info = np.load(args.clustering_path, allow_pickle=True).item()
    print(f"聚类信息已加载。包含的键：{clustering_info.keys()}")
    print(f"聚类信息包含{len(clustering_info.get('page_to_cluster', {}))}个页面映射")

    # 修改随机采样打印代码
    page_to_cluster = clustering_info.get('page_to_cluster', {})
    if page_to_cluster:
        sample_keys = random.sample(list(page_to_cluster.keys()), min(5, len(page_to_cluster)))
        print("随机采样5个映射示例：")
        for page in sample_keys:
            cluster = page_to_cluster[page]
            if isinstance(page, int):
                print(f"页面 {page:x} -> 聚类 {cluster}")
            else:
                print(f"页面 {page} -> 聚类 {cluster}")
    
    # Generate prefetches
    generate_prefetches(
        model=model,
        clustering_info=clustering_info,
        trace_path=args.benchmark,
        config=config,
        output_path=args.output
    )
    
    print("Prefetch generation complete!")

if __name__ == "__main__":
    main()