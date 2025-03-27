import argparse
import numpy as np
import tensorflow as tf
import lzma
import os

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
    Generate prefetch file using T-LITE model with GPU-accelerated batch processing
    Output format:
    instruction_id prefetch_address
    Example:
    3659 a1b2c3d4
    """
    print(f"Generating prefetches for {trace_path}")
    
    # Initialize prefetcher
    prefetcher = TLITEPrefetcher(
        model=model,
        clustering_info=clustering_info,
        config=config
    )
    
    # 增加批处理大小，充分利用GPU
    batch_size = 10000  # 显著增加批处理大小
    
    # 预分配GPU内存，避免频繁的内存分配
    @tf.function(experimental_relax_shapes=True)
    def predict_batch(cluster_histories, offset_histories, pcs, dpf_vectors):
        return model((cluster_histories, offset_histories, pcs, dpf_vectors))
    
    # 读取和处理文件
    if trace_path.endswith('.txt.xz'):
        f = lzma.open(trace_path, mode='rt', encoding='utf-8')
    else:
        f = open(trace_path, 'r')
    
    print("Counting total lines...")
    total_lines = sum(1 for line in f if not (line.startswith('***') or line.startswith('Read')))
    f.seek(0)
    print(f"Total lines to process: {total_lines}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 预分配批处理数组
    pages_batch = []
    offsets_batch = []
    pcs_batch = []
    inst_ids_batch = []
    processed_lines = 0
    
    # 用于批量预测的数组
    cluster_histories = []
    offset_histories = []
    dpf_vectors_batch = []
    
    with open(output_path, 'w', buffering=1024*1024) as out_f:  # 增加文件缓冲区
        for line in f:
            if line.startswith('***') or line.startswith('Read'):
                continue
            
            # Parse line
            split = line.strip().split(', ')
            inst_id = int(split[0])
            pc = int(split[3], 16)
            addr = int(split[2], 16)
            
            # Convert address to page and offset
            page = (addr >> 6) >> config.offset_bits
            offset = (addr >> 6) & ((1 << config.offset_bits) - 1)
            
            # 收集批处理数据
            pages_batch.append(page)
            offsets_batch.append(offset)
            pcs_batch.append(pc)
            inst_ids_batch.append(inst_id)
            
            # 当批次满时进行处理
            if len(pages_batch) >= batch_size:
                # 准备模型输入
                for i in range(len(pages_batch)):
                    # 更新预取器状态并获取模型输入
                    prefetcher.update_history(pages_batch[i], offsets_batch[i], pcs_batch[i])
                    cluster_history, offset_history, _, dpf_vectors = prefetcher.metadata_manager.prepare_model_inputs(
                        prefetcher.page_history,
                        prefetcher.offset_history,
                        pcs_batch[i],
                        prefetcher.cluster_mapping_utils
                    )
                    cluster_histories.append(cluster_history)
                    offset_histories.append(offset_history)
                    dpf_vectors_batch.append(dpf_vectors)
                
                # 批量预测
                candidate_logits, offset_logits = predict_batch(
                    tf.convert_to_tensor(cluster_histories, dtype=tf.int32),
                    tf.convert_to_tensor(offset_histories, dtype=tf.int32),
                    tf.convert_to_tensor(pcs_batch, dtype=tf.int32),
                    tf.convert_to_tensor(dpf_vectors_batch, dtype=tf.float32)
                )
                
                # 处理预测结果
                candidates = tf.argmax(candidate_logits, axis=1).numpy()
                offsets = tf.argmax(offset_logits, axis=1).numpy()
                
                # 写入预测结果 - 修改这部分以匹配新格式
                for i in range(len(pages_batch)):
                    if candidates[i] != config.num_candidates:  # 不是no-prefetch
                        candidate_pages = prefetcher.metadata_manager.get_candidate_pages(prefetcher.page_history[-1])
                        if candidate_pages and candidates[i] < len(candidate_pages):
                            # 计算预取地址
                            prefetch_page = candidate_pages[candidates[i]][0]
                            prefetch_offset = offsets[i]
                            
                            # 构造完整的预取地址
                            # 页面地址 << (偏移位数 + 6) | (偏移 << 6)
                            prefetch_addr = (prefetch_page << (config.offset_bits + 6)) | (prefetch_offset << 6)
                            
                            # 以十六进制格式写入
                            out_f.write(f"{inst_ids_batch[i]} {prefetch_addr:x}\n")
                
                # 清空批处理数组
                pages_batch = []
                offsets_batch = []
                pcs_batch = []
                inst_ids_batch = []
                cluster_histories = []
                offset_histories = []
                dpf_vectors_batch = []
                
                # 更新进度
                processed_lines += batch_size
                print(f"Progress: {processed_lines}/{total_lines} lines ({processed_lines/total_lines*100:.2f}%)")
        
        # 处理最后一个不完整的批次
        if pages_batch:
            # 对最后一批数据执行相同的处理逻辑
            # ... 批处理预测代码 ...
            
            # 写入最后一批结果
            for i in range(len(pages_batch)):
                if candidates[i] != config.num_candidates:
                    candidate_pages = prefetcher.metadata_manager.get_candidate_pages(prefetcher.page_history[-1])
                    if candidate_pages and candidates[i] < len(candidate_pages):
                        prefetch_page = candidate_pages[candidates[i]][0]
                        prefetch_offset = offsets[i]
                        prefetch_addr = (prefetch_page << (config.offset_bits + 6)) | (prefetch_offset << 6)
                        out_f.write(f"{inst_ids_batch[i]} {prefetch_addr:x}\n")
            
            processed_lines += len(pages_batch)
    
    f.close()
    print(f"Generated prefetch file: {output_path}")
    print(f"Processed {processed_lines} lines total")
    prefetcher.print_stats()

def main():
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Load model, ignore warnings
    model = create_tlite_model(config)
    model.load_weights(args.model_path).expect_partial()
    
    # Load clustering information
    clustering_info = np.load(args.clustering_path, allow_pickle=True).item()
    
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