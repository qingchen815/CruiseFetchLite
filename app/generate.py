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
    Generate prefetch file using T-LITE model with multi-stream processing
    """
    print(f"\n=== 开始生成预取文件 ===")
    print(f"处理轨迹文件: {trace_path}")
    print(f"输出文件: {output_path}")
    
    # 初始化多流预取器
    num_streams = 16
    prefetchers = [TLITEPrefetcher(
        model=model,
        clustering_info=clustering_info,
        config=config
    ) for _ in range(num_streams)]
    
    # 预热过程
    print("\n=== 预取器预热 ===")
    for stream_id, prefetcher in enumerate(prefetchers):
        print(f"预热流 {stream_id}...")
        for i in range(10):
            fake_page = i + 1
            fake_offset = i % 64
            fake_pc = i * 1000
            prefetcher.update_history(fake_page, fake_offset, fake_pc)
            if i > 0:
                prefetcher.metadata_manager.update_page_access(
                    i, fake_page, (i-1) % 64, fake_offset
                )
        
        if stream_id < 3:
            print(f"流 {stream_id} 预热后状态:")
            print(f"- 页面历史: {prefetcher.page_history}")
            print(f"- 偏移历史: {prefetcher.offset_history}")
            print(f"- 元数据大小: {prefetcher.metadata_manager.get_metadata_size_kb():.2f} KB")
    
    # 批处理大小
    batch_size = 30000
    print(f"批处理大小: {batch_size}")
    
    # 预分配GPU内存
    @tf.function(experimental_relax_shapes=True)
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
    
    # 批处理数组
    pages_batch = []
    offsets_batch = []
    pcs_batch = []
    inst_ids_batch = []
    stream_ids_batch = []
    processed_lines = 0
    
    # 性能统计
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
            
            page = (addr >> 6) >> config.offset_bits
            offset = (addr >> 6) & ((1 << config.offset_bits) - 1)
            
            stream_id = (pc >> 4) % num_streams
            
            # 调试输出：显示页面历史更新
            if processed_lines < 100 and processed_lines % 10 == 0:
                print(f"\n更新前流 {stream_id} 状态:")
                print(f"- 页面历史: {prefetchers[stream_id].page_history}")
                print(f"- 元数据大小: {prefetchers[stream_id].metadata_manager.get_metadata_size_kb():.2f} KB")
            
            # 更新状态和元数据
            prefetchers[stream_id].update_history(page, offset, pc)
            if prefetchers[stream_id].page_history[-2] != 0:
                prefetchers[stream_id].metadata_manager.update_page_access(
                    prefetchers[stream_id].page_history[-2],
                    page,
                    prefetchers[stream_id].offset_history[-2],
                    offset
                )
            
            # 调试输出：显示更新后状态
            if processed_lines < 100 and processed_lines % 10 == 0:
                print(f"更新后流 {stream_id} 状态:")
                print(f"- 页面历史: {prefetchers[stream_id].page_history}")
                print(f"- 元数据大小: {prefetchers[stream_id].metadata_manager.get_metadata_size_kb():.2f} KB")
            
            # 收集批处理数据
            pages_batch.append(page)
            offsets_batch.append(offset)
            pcs_batch.append(pc)
            inst_ids_batch.append(inst_id)
            stream_ids_batch.append(stream_id)
            
            # 处理满批次
            if len(pages_batch) >= batch_size:
                print(f"\n--- 处理批次 {processed_lines//batch_size + 1} ---")
                
                # 准备模型输入
                cluster_histories = []
                offset_histories = []
                dpf_vectors_batch = []
                valid_indices = []
                
                # 收集所有样本，不再过滤
                for i, (page, offset, pc, stream_id) in enumerate(zip(pages_batch, offsets_batch, pcs_batch, stream_ids_batch)):
                    prefetcher = prefetchers[stream_id]
                    cluster_history, offset_history, _, dpf_vectors = prefetcher.metadata_manager.prepare_model_inputs(
                        prefetcher.page_history,
                        prefetcher.offset_history,
                        pc,
                        prefetcher.cluster_mapping_utils
                    )
                    
                    cluster_histories.append(cluster_history)
                    offset_histories.append(offset_history)
                    dpf_vectors_batch.append(dpf_vectors)
                    valid_indices.append(i)
                    
                    # 调试输出前几个样本
                    if processed_lines < batch_size * 3 and i < 5:
                        print(f"\n样本 {i} 详情:")
                        print(f"- cluster_history: {cluster_history}")
                        print(f"- offset_history: {offset_history}")
                        print(f"- 当前页面: {page}")
                        print(f"- 流ID: {stream_id}")
                
                # 执行预测
                if valid_indices:
                    candidate_logits, offset_logits = predict_batch(
                        tf.convert_to_tensor(cluster_histories, dtype=tf.int32),
                        tf.convert_to_tensor(offset_histories, dtype=tf.int32),
                        tf.convert_to_tensor([pcs_batch[i] for i in valid_indices], dtype=tf.int32),
                        tf.convert_to_tensor(dpf_vectors_batch, dtype=tf.float32)
                    )
                    
                    candidates = tf.argmax(candidate_logits, axis=1).numpy()
                    offsets = tf.argmax(offset_logits, axis=1).numpy()
                    
                    # 修改后的预取生成逻辑
                    for j, i in enumerate(valid_indices):
                        stream_id = stream_ids_batch[i]
                        prefetcher = prefetchers[stream_id]
                        
                        if candidates[j] != config.num_candidates:
                            candidate_pages = prefetcher.metadata_manager.get_candidate_pages(
                                prefetcher.page_history[-1]
                            )
                            
                            if candidate_pages and candidates[j] < len(candidate_pages):
                                prefetch_page = candidate_pages[candidates[j]][0]
                            else:
                                # 使用当前页面或默认页面
                                prefetch_page = prefetcher.page_history[-1]
                                if prefetch_page == 0:
                                    prefetch_page = pages_batch[i]  # 使用当前访问的页面
                            
                            prefetch_offset = offsets[j]
                            prefetch_addr = (prefetch_page << (config.offset_bits + 6)) | (prefetch_offset << 6)
                            
                            out_f.write(f"{inst_ids_batch[i]} {prefetch_addr:x}\n")
                            stream_stats[stream_id]['prefetches'] += 1
                            valid_prefetches += 1
                    
                    total_predictions += len(valid_indices)
                
                # 打印批次统计
                print(f"\n批次统计:")
                print(f"- 样本数: {len(valid_indices)}")
                print(f"- 生成预取数: {valid_prefetches}")
                if total_predictions > 0:
                    print(f"- 当前预取率: {valid_prefetches/total_predictions*100:.2f}%")
                
                # 清空批处理数组
                pages_batch = []
                offsets_batch = []
                pcs_batch = []
                inst_ids_batch = []
                stream_ids_batch = []
                
                # 更新进度
                processed_lines += batch_size
                print(f"进度: {processed_lines}/{total_lines} ({processed_lines/total_lines*100:.2f}%)")
                
                # 定期打印流统计
                if processed_lines % (batch_size * 10) == 0:
                    print("\n=== 流统计 ===")
                    for stream_id, stats in stream_stats.items():
                        if stats['updates'] > 0:
                            print(f"流 {stream_id}:")
                            print(f"- 更新次数: {stats['updates']}")
                            print(f"- 生成预取数: {stats['prefetches']}")
                            print(f"- 预取率: {stats['prefetches']/stats['updates']*100:.2f}%")
    
    # 最终统计（添加零检查）
    print("\n=== 预取生成完成 ===")
    print(f"总处理行数: {processed_lines}")
    print(f"总预测次数: {total_predictions}")
    print(f"有效预取数: {valid_prefetches}")
    if total_predictions > 0:
        print(f"整体预取率: {valid_prefetches/total_predictions*100:.2f}%")
    else:
        print("整体预取率: 0.00% (无有效预测)")
    
    # 打印每个流的最终统计
    print("\n=== 最终流统计 ===")
    for stream_id, stats in stream_stats.items():
        if stats['updates'] > 0:
            print(f"流 {stream_id}:")
            print(f"- 总更新次数: {stats['updates']}")
            print(f"- 总预取数: {stats['prefetches']}")
            print(f"- 预取率: {stats['prefetches']/stats['updates']*100:.2f}%")

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