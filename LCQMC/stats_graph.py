import tensorflow as tf
def stats_graph(graph):
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))
    with open('flpos.txt', 'w') as f:
      f.write('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))
