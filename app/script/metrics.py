import tensorflow as tf

class CandidateAccuracy(tf.keras.metrics.Metric):
    '''Metric to track candidate prediction accuracy'''
    
    def __init__(self, name='candidate_accuracy', **kwargs):
        super(CandidateAccuracy, self).__init__(name=name, **kwargs)
        self.correct = self.add_weight(name='correct', initializer='zeros')
        self.total = self.add_weight(name='total', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Unpack
        candidate_labels, _ = y_true
        candidate_logits, _ = y_pred
        
        # Calculate predictions
        predictions = tf.argmax(candidate_logits, axis=-1)
        
        # Calculate correct predictions
        correct = tf.cast(tf.equal(predictions, tf.cast(candidate_labels, tf.int64)), tf.float32)
        
        # Update metrics
        if sample_weight is not None:
            self.correct.assign_add(tf.reduce_sum(correct * sample_weight))
            self.total.assign_add(tf.reduce_sum(sample_weight))
        else:
            self.correct.assign_add(tf.reduce_sum(correct))
            self.total.assign_add(tf.cast(tf.size(candidate_labels), tf.float32))
    
    def result(self):
        return self.correct / self.total * 100
    
    def reset_state(self):
        self.correct.assign(0.0)
        self.total.assign(0.0)


class OffsetAccuracy(tf.keras.metrics.Metric):
    '''Metric to track offset prediction accuracy'''
    
    def __init__(self, name='offset_accuracy', **kwargs):
        super(OffsetAccuracy, self).__init__(name=name, **kwargs)
        self.correct = self.add_weight(name='correct', initializer='zeros')
        self.total = self.add_weight(name='total', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Unpack
        _, offset_labels = y_true
        _, offset_logits = y_pred
        
        # Calculate predictions
        predictions = tf.argmax(offset_logits, axis=-1)
        
        # Calculate correct predictions
        correct = tf.cast(tf.equal(predictions, tf.cast(offset_labels, tf.int64)), tf.float32)
        
        # Update metrics
        if sample_weight is not None:
            self.correct.assign_add(tf.reduce_sum(correct * sample_weight))
            self.total.assign_add(tf.reduce_sum(sample_weight))
        else:
            self.correct.assign_add(tf.reduce_sum(correct))
            self.total.assign_add(tf.cast(tf.size(offset_labels), tf.float32))
    
    
    def result(self):
        return self.correct / self.total * 100
    
    def reset_state(self):
        self.correct.assign(0.0)
        self.total.assign(0.0)


class OverallAccuracy(tf.keras.metrics.Metric):
    '''Metric to track overall prediction accuracy (both candidate and offset correct)'''
    
    def __init__(self, name='overall_accuracy', **kwargs):
        super(OverallAccuracy, self).__init__(name=name, **kwargs)
        self.correct = self.add_weight(name='correct', initializer='zeros')
        self.total = self.add_weight(name='total', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Unpack
        candidate_labels, offset_labels = y_true
        candidate_logits, offset_logits = y_pred
        
        # Calculate predictions
        candidate_preds = tf.argmax(candidate_logits, axis=-1)
        offset_preds = tf.argmax(offset_logits, axis=-1)
        
        # Calculate correct predictions (both must be correct)
        candidate_correct = tf.equal(candidate_preds, tf.cast(candidate_labels, tf.int64))
        offset_correct = tf.equal(offset_preds, tf.cast(offset_labels, tf.int64))
        both_correct = tf.cast(tf.logical_and(candidate_correct, offset_correct), tf.float32)
        
        # Update metrics
        if sample_weight is not None:
            self.correct.assign_add(tf.reduce_sum(both_correct * sample_weight))
            self.total.assign_add(tf.reduce_sum(sample_weight))
        else:
            self.correct.assign_add(tf.reduce_sum(both_correct))
            self.total.assign_add(tf.cast(tf.size(candidate_labels), tf.float32))
    
    def result(self):
        return self.correct / self.total * 100
    
    def reset_state(self):
        self.correct.assign(0.0)
        self.total.assign(0.0)


class UsefulPrefetchRate(tf.keras.metrics.Metric):
    """Metric to track the rate of useful prefetches"""
    
    def __init__(self, name='useful_prefetch_rate', **kwargs):
        super(UsefulPrefetchRate, self).__init__(name=name, **kwargs)
        self.useful = self.add_weight(name='useful', initializer='zeros')
        self.total = self.add_weight(name='total', initializer='zeros')
    
    def update_state(self, prefetch_addresses, future_accesses, sample_weight=None):
        """
        Args:
            prefetch_addresses: Addresses that were prefetched
            future_accesses: Set of addresses that will be accessed in the future
        """
        # Check which prefetches will be useful (appear in future accesses)
        is_useful = tf.cast(tf.reduce_any(
            tf.equal(
                tf.expand_dims(prefetch_addresses, 1),
                tf.expand_dims(future_accesses, 0)
            ), 
            axis=1
        ), tf.float32)
        
        # Update metrics
        if sample_weight is not None:
            self.useful.assign_add(tf.reduce_sum(is_useful * sample_weight))
            self.total.assign_add(tf.reduce_sum(sample_weight))
        else:
            self.useful.assign_add(tf.reduce_sum(is_useful))
            self.total.assign_add(tf.cast(tf.size(prefetch_addresses), tf.float32))
    
    def result(self):
        return self.useful / self.total * 100
    
    def reset_state(self):
        self.useful.assign(0.0)
        self.total.assign(0.0)


class CoveragePrefetchRate(tf.keras.metrics.Metric):
    """Metric to track coverage of future memory accesses"""
    
    def __init__(self, name='coverage_rate', **kwargs):
        super(CoveragePrefetchRate, self).__init__(name=name, **kwargs)
        self.covered = self.add_weight(name='covered', initializer='zeros')
        self.total = self.add_weight(name='total', initializer='zeros')
    
    def update_state(self, prefetch_addresses, future_accesses, sample_weight=None):
        """
        Args:
            prefetch_addresses: Addresses that were prefetched
            future_accesses: Set of addresses that will be accessed in the future
        """
        # Check which future accesses are covered by prefetches
        is_covered = tf.cast(tf.reduce_any(
            tf.equal(
                tf.expand_dims(future_accesses, 1),
                tf.expand_dims(prefetch_addresses, 0)
            ), 
            axis=1
        ), tf.float32)
        
        # Update metrics
        if sample_weight is not None:
            self.covered.assign_add(tf.reduce_sum(is_covered * sample_weight))
            self.total.assign_add(tf.reduce_sum(sample_weight))
        else:
            self.covered.assign_add(tf.reduce_sum(is_covered))
            self.total.assign_add(tf.cast(tf.size(future_accesses), tf.float32))
    
    def result(self):
        return self.covered / self.total * 100
    
    def reset_state(self):
        self.covered.assign(0.0)
        self.total.assign(0.0)
