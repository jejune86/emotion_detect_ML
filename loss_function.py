import tensorflow as tf

# Focal Loss
def focal_loss_sparse(gamma=2., alpha=0.25):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0)
        y_pred_probs = tf.gather(y_pred, y_true[..., None], axis=-1, batch_dims=1)
        y_pred_probs = tf.squeeze(y_pred_probs, axis=-1)
        cross_entropy = -tf.math.log(y_pred_probs)
        p_t = tf.math.exp(-cross_entropy)
        loss = alpha * (1 - p_t) ** gamma * cross_entropy
        return tf.reduce_mean(loss)
    
    return loss