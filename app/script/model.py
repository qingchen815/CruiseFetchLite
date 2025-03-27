import tensorflow as tf
import numpy as np
import sys  # 用于调试输出

class TLITE(tf.keras.Model):
    '''
    T-LITE: A lightweight version of Twilight neural prefetcher
    with behavioral clustering and frequency-based candidate selection
    '''

    def __init__(self, config):
        super(TLITE, self).__init__()
        
        # 保存配置参数
        self.pc_embed_size = config.pc_embed_size           
        self.cluster_embed_size = config.cluster_embed_size  
        self.offset_embed_size = config.offset_embed_size    
        self.num_experts = config.num_experts                
        self.history_length = config.history_length          
        self.num_pcs = config.num_pcs                        
        self.num_clusters = config.num_clusters              
        self.offset_size = config.offset_size                
        self.num_candidates = config.num_candidates          
        self.dpf_history_length = config.dpf_history_length  
        self.steps_per_epoch = config.steps_per_epoch        
        
        # 跟踪训练状态的变量
        self.step = 0
        self.epoch = 0
        
        # 初始化模型组件
        self._init_embedding_layers()
        self._init_prediction_layers()
    
    def _init_embedding_layers(self):
        '''初始化嵌入层'''
        # PC嵌入层（限制为最常见的N个PC）
        self.pc_embedding = tf.keras.layers.Embedding(
            self.num_pcs, 
            self.pc_embed_size,
            embeddings_regularizer='l1'
        )
        
        # 聚类嵌入层
        self.cluster_embedding = tf.keras.layers.Embedding(
            self.num_clusters,
            self.cluster_embed_size,
            embeddings_regularizer='l1'
        )
        
        # 偏移嵌入层（分为多个专家）
        self.offset_embedding = tf.keras.layers.Embedding(
            self.offset_size,
            self.offset_embed_size,
            embeddings_regularizer='l1'
        )
        
        # 用于上下文感知偏移嵌入的多头注意力
        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=1,
            key_dim=self.cluster_embed_size,
            kernel_regularizer='l1'
        )
    
    def _init_prediction_layers(self):
        '''初始化预测层'''
        # 候选预测层（N+1输出用于N个候选+不预取选项）
        self.candidate_fc = tf.keras.layers.Dense(
            self.num_candidates + 1,
            activation=None,
            kernel_regularizer='l1'
        )
        
        # 偏移预测层
        self.offset_fc = tf.keras.layers.Dense(
            self.offset_size,
            activation=None,
            kernel_regularizer='l1'
        )
    
    def compute_context_aware_offset_embedding(self, cluster_history, offset_history, pc, training=False):
        '''
        计算上下文感知的偏移嵌入
        通过融合聚类和PC上下文信息，使用专家混合方法
        
        Args:
            cluster_history: 聚类ID历史 [batch, history_length]
            offset_history: 偏移历史 [batch, history_length]
            pc: 加载PC [batch, 1]
            training: 是否处于训练模式
            
        Returns:
            上下文感知的偏移嵌入 [batch, history_length, cluster_embed_size]
        '''
        # 获取批次大小
        batch_size = tf.shape(cluster_history)[0]
        
        # 获取原始嵌入
        cluster_embed = self.cluster_embedding(cluster_history)  # [batch, history, embed_dim]
        offset_embed = self.offset_embedding(offset_history)     # [batch, history, offset_embed_size]
        
        # 调试时打印形状
        # tf.print("batch_size:", batch_size, "history_length:", self.history_length, 
        #          "cluster_embed shape:", tf.shape(cluster_embed),
        #          "offset_embed shape:", tf.shape(offset_embed),
        #          output_stream=sys.stderr)
        
        # 重塑偏移嵌入为专家格式
        offset_embed = tf.reshape(
            offset_embed, 
            [batch_size, self.history_length, self.num_experts, self.cluster_embed_size]
        )
        
        # 准备查询 - 只使用聚类嵌入作为查询
        # 这种简化方法避免了过去连接操作导致的维度不匹配问题
        query = tf.reshape(
            cluster_embed, 
            [batch_size, self.history_length, 1, self.cluster_embed_size]
        )
        
        # 应用注意力机制获取加权偏移嵌入
        # 调试时可以打印查询和值的形状
        # tf.print("query shape:", tf.shape(query), "value shape:", tf.shape(offset_embed), output_stream=sys.stderr)
        
        context_aware_offset = self.mha(
            query=query,        # [batch, history, 1, cluster_embed_size]
            value=offset_embed, # [batch, history, num_experts, cluster_embed_size]
            training=training
        )
        
        # 重塑为最终嵌入维度
        context_aware_offset = tf.reshape(
            context_aware_offset, 
            [batch_size, self.history_length, self.cluster_embed_size]
        )
        
        return context_aware_offset
    
    def call(self, inputs, training=False):
        '''
        模型前向传播
        
        Args:
            inputs: 元组 (cluster_history, offset_history, pc, dpf_vectors)
                - cluster_history: 聚类ID历史 [batch, history_length]
                - offset_history: 偏移历史 [batch, history_length]
                - pc: 加载PC [batch, 1]
                - dpf_vectors: DPF分布向量 [batch, dpf_history_length, num_candidates]
            training: 是否处于训练模式
            
        Returns:
            candidate_logits: 候选预测logits [batch, num_candidates+1]
            offset_logits: 偏移预测logits [batch, offset_size]
        '''
        # 解包输入
        cluster_history, offset_history, pc, dpf_vectors = inputs
        
        # 获取批次大小用于一致的维度处理
        batch_size = tf.shape(cluster_history)[0]
        
        # 计算嵌入
        cluster_embed = self.cluster_embedding(cluster_history)  # [batch, history, embed_dim]
        context_aware_offset = self.compute_context_aware_offset_embedding(
            cluster_history, offset_history, pc, training
        )
        pc_embed = self.pc_embedding(pc)  # [batch, 1, pc_embed_dim]
        
        # 确保pc_embed是2D的 - 修复关键点
        pc_embed = tf.reshape(pc_embed, [batch_size, self.pc_embed_size])
        
        # 展平DPF向量
        dpf_flat = tf.reshape(
            dpf_vectors, 
            [batch_size, self.dpf_history_length * self.num_candidates]
        )
        
        # 展平嵌入
        cluster_flat = tf.reshape(cluster_embed, [batch_size, self.history_length * self.cluster_embed_size])
        offset_flat = tf.reshape(context_aware_offset, [batch_size, self.history_length * self.cluster_embed_size])
        
        # 连接所有特征
        combined_features = tf.concat([
            pc_embed,     # PC嵌入
            cluster_flat, # 聚类历史嵌入
            offset_flat,  # 上下文感知的偏移嵌入
            dpf_flat      # DPF分布向量
        ], axis=1)
        
        # 生成预测
        candidate_logits = self.candidate_fc(combined_features)
        offset_logits = self.offset_fc(combined_features)
        
        return candidate_logits, offset_logits
    
    def train_step(self, data):
        '''自定义训练步骤，更新内部计数器'''
        # 解包数据
        x, y = data
        
        # 前向传播
        with tf.GradientTape() as tape:
            candidate_logits, offset_logits = self(x, training=True)
            
            # 解包真实标签
            candidate_labels, offset_labels = y
            
            # 计算损失
            candidate_loss = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True
            )(candidate_labels, candidate_logits)
            
            offset_loss = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True
            )(offset_labels, offset_logits)
            
            # 总损失
            total_loss = candidate_loss + offset_loss
        
        # 计算梯度并更新权重
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # 更新内部计数器
        self.step += 1
        if self.step >= self.steps_per_epoch:
            self.step = 0
            self.epoch += 1
        
        # 更新指标
        self.compiled_metrics.update_state(y, (candidate_logits, offset_logits))
        
        # 返回指标
        results = {m.name: m.result() for m in self.metrics}
        results['loss'] = total_loss
        
        return results
    
    def load(self, model_path):
        '''加载模型权重'''
        self.load_weights(model_path).expect_partial()
    
    def save(self, model_path):
        '''保存模型权重'''
        self.save_weights(model_path)
    
    def quantize(self, bits=8):
        '''
        将模型权重量化为指定位数的精度
        
        这是一个简单实现 - 生产环境应使用TF的量化API
        
        Args:
            bits: 量化精度位数
            
        Returns:
            包含量化信息的字典
        '''
        # 获取所有权重
        weights = self.get_weights()
        quantized_weights = []
        
        for w in weights:
            # 基于最小/最大值计算缩放因子
            w_min = np.min(w)
            w_max = np.max(w)
            scale = (w_max - w_min) / (2**bits - 1)
            
            # 量化
            if scale > 0:  # 防止除零错误
                w_quant = np.round((w - w_min) / scale).astype(np.int8)
            else:
                w_quant = np.zeros_like(w, dtype=np.int8)
            
            # 存储量化权重
            quantized_weights.append(w_quant)
        
        # 设置量化权重
        self.set_weights(quantized_weights)
        
        return {
            'bits': bits,
            'orig_size_mb': sum(w.size * 4 for w in weights) / (1024 * 1024),
            'quant_size_mb': sum(w.size * bits / 8 for w in weights) / (1024 * 1024)
        }


def create_tlite_model(config):
    '''
    创建并编译T-LITE模型
    
    Args:
        config: 包含模型超参数的配置对象
        
    Returns:
        编译好的T-LITE模型
    '''
    # 创建模型
    model = TLITE(config)
    
    # 使用小批次测试模型结构
    batch_size = 2  # 使用>1的批次大小以测试批次处理逻辑
    
    # 定义测试输入构建模型
    dummy_cluster_history = tf.zeros((batch_size, model.history_length), dtype=tf.int32)
    dummy_offset_history = tf.zeros((batch_size, model.history_length), dtype=tf.int32)
    dummy_pc = tf.zeros((batch_size, 1), dtype=tf.int32)
    dummy_dpf = tf.zeros((batch_size, model.dpf_history_length, model.num_candidates), dtype=tf.float32)
    
    # 构建模型
    # 启用调试时取消注释以下行
    # tf.print("Testing model with batch size:", batch_size, output_stream=sys.stderr)
    model((dummy_cluster_history, dummy_offset_history, dummy_pc, dummy_dpf))
    
    # 导入指标
    try:
        from metrics import CandidateAccuracy, OffsetAccuracy, OverallAccuracy
        
        # 定义指标
        metrics = [
            CandidateAccuracy(), 
            OffsetAccuracy(),
            OverallAccuracy()
        ]
    except ImportError:
        print("警告: 未能导入指标模块，将使用默认指标")
        metrics = ['accuracy']
    
    # 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.learning_rate),
        loss=None,  # 损失在train_step中计算
        metrics=metrics
    )
    
    return model