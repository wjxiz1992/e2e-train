import argparse
import h5py
import io
import os
import pyarrow
import sys
from pyspark.sql import SparkSession

parser = argparse.ArgumentParser(description='Horovod Spark TensorFlow Training Demo',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--num-proc', help='The number of Spark Executors to run the application',
                    type=int, default=1, required=True)
parser.add_argument('--model-output-path', help='Path for best model',
                    type=str, required=True)
parser.add_argument('--input-data', help='Input path for parquet training dataset',
                    type=str, required=True)



def arrange_piece_indices_for_worker(pq_ds_pieces, global_rank, global_size):
    """
    Split row groups for each worker according to worker index and total number of workers
    :param pq_ds_pieces: The whole row groups for a parquet dataset. A parquet dataset may contain multiple parquet files,
                        each parquet file may contain multiple row groups.
    :param global_rank: index for the worker among all workers.
    :param global_size: total number of workers
    :return: [[row_group_indexes]]. It's a 2-d list due to the requirement of cudf parquet reader API.
    """
    import random
    from collections import OrderedDict
    # shuffle here?
    assert(global_rank < global_size)
    candidates = [index for index in range(len(pq_ds_pieces)) if index % global_size == global_rank]
    file_indices_od = OrderedDict()
    for i in candidates:
        if not pq_ds_pieces[i].path in file_indices_od.keys():
            file_indices_od[pq_ds_pieces[i].path] = [pq_ds_pieces[i].row_group]
        else:
            file_indices_od[pq_ds_pieces[i].path].append(pq_ds_pieces[i].row_group)
    row_group_list = list(file_indices_od.values())
    return row_group_list

def split_feature_label(gdf, label_name):
    """
    split a gdf(cuDF DataFrame) into feature gdf and label gdf. This is used to tensorflow training API which requires
    x(feature) and y(label)
    :param gdf: a cuDF contains feature columns and label columns
    :param label_name:name of the label column
    :return: feature gdf ,label gdf
    """
    return gdf.drop(columns=[label_name]), gdf[label_name].to_frame()


def cudf_to_tensor(gdf):
    """
    copied from https://github.com/NVIDIA/NVTabular/blob/main/nvtabular/loader/tensorflow.py#L330
    :param gdf: a cuDF dataframe
    :return: a TensorFlow Tensor
    """
    import tensorflow as tf
    if gdf.empty:
        return
    # checks necessary because of this bug
    # https://github.com/tensorflow/tensorflow/issues/42660
    if len(gdf.shape) == 1 or gdf.shape[1] == 1:
        dlpack = gdf.to_dlpack()
    elif gdf.shape[0] == 1:
        dlpack = gdf.values[0].toDlpack()
    else:
        dlpack = gdf.values.T.toDlpack()

    # catch error caused by tf eager context
    # not being initialized
    try:
        x = tf.experimental.dlpack.from_dlpack(dlpack)
    except AssertionError:
        tf.random.uniform((1,))
        x = tf.experimental.dlpack.from_dlpack(dlpack)

    if gdf.shape[0] == 1:
        # batch size 1 so got squashed to a vector
        x = tf.expand_dims(x, 0)
    elif len(gdf.shape) == 1 or len(x.shape) == 1:
        # sort of a generic check for any other
        # len(shape)==1 case, could probably
        # be more specific
        x = tf.expand_dims(x, -1)
    elif gdf.shape[1] > 1:
        # matrix which means we had to transpose
        # for the bug above, so untranspose
        x = tf.transpose(x)
    return x

if __name__ == '__main__':
    args = parser.parse_args()

    def serialize_model(model):
        """Serialize model into byte array."""
        bio = io.BytesIO()
        with h5py.File(bio) as f:
            model.save(f)
        return bio.getvalue()

    def deserialize_model(model_bytes, load_model_fn):
        """Deserialize model from byte array."""
        bio = io.BytesIO(model_bytes)
        with h5py.File(bio) as f:
            return load_model_fn(f)

    print('==============')
    print('Model training')
    print('==============')

    import tensorflow as tf
    from tensorflow import keras
    import horovod.spark
    import horovod.tensorflow.keras as hvd

    # Disable GPUs when building the model to prevent memory leaks
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(784,)),
                                        tf.keras.layers.Dense(128, activation='relu'),
                                        tf.keras.layers.Dropout(0.2),
                                        tf.keras.layers.Dense(10)
                                        ])
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    # hard-code lr here for demo
    opt = tf.keras.optimizers.Adam(lr=0.001, epsilon=1e-3)
    opt = hvd.DistributedOptimizer(opt)

    model.compile(opt,
                  loss=loss_fn,
                  metrics=['accuracy'],
                  experimental_run_tf_function=False)
    model_bytes = serialize_model(model)


    def train_fn(model_bytes):
        # Make sure pyarrow is referenced before anything else to avoid segfault due to conflict
        # with TensorFlow libraries.  Use `pa` package reference to ensure it's loaded before
        # functions like `deserialize_model` which are implemented at the top level.
        # See https://jira.apache.org/jira/browse/ARROW-3346
        pyarrow
        import atexit
        import cudf
        import tensorflow as tf
        import horovod.tensorflow.keras as hvd
        import tensorflow.keras.backend as K
        import tempfile
        import shutil
        from pyarrow import parquet as pq
        from petastorm.etl import dataset_metadata
        from horovod.spark.task import get_available_devices

        # Horovod: initialize Horovod inside the trainer.
        hvd.init()
        # make TF less agressive for GPU Memeory
        os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['OMPI_COMM_WORLD_LOCAL_RANK'] # This line doesn't work, see https://github.com/horovod/horovod/issues/3005
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
        # config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        # config.gpu_options.visible_device_list = get_available_devices()[0]
        # K.set_session(tf.Session(config=config))
        # Horovod: restore from checkpoint, use hvd.load_model under the hood.
        model = deserialize_model(model_bytes, hvd.load_model)

        # Horovod: adjust learning rate based on number of processes.
        scaled_lr = K.get_value(model.optimizer.lr) * hvd.size()
        K.set_value(model.optimizer.lr, scaled_lr)

        # Horovod: print summary logs on the first worker.
        verbose = 2 if hvd.rank() == 0 else 0

        callbacks = [
            # Horovod: broadcast initial variable states from rank 0 to all other processes.
            # This is necessary to ensure consistent initialization of all workers when
            # training is started with random weights or restored from a checkpoint.
            hvd.callbacks.BroadcastGlobalVariablesCallback(root_rank=0),
            # Horovod: average metrics among workers at the end of every epoch.
            #
            # Note: This callback must be in the list before the ReduceLROnPlateau,
            # TensorBoard, or other metrics-based callbacks.
            hvd.callbacks.MetricAverageCallback(),
            # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
            # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
            # the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
            hvd.callbacks.LearningRateWarmupCallback(initial_lr=scaled_lr, warmup_epochs=5, verbose=verbose),
            # Reduce LR if the metric is not improved for 10 epochs, and stop training
            # if it has not improved for 20 epochs.
            tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=10, verbose=verbose),
            tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', patience=20, verbose=verbose),
            tf.keras.callbacks.TerminateOnNaN()
        ]

        # Model checkpoint location.
        ckpt_dir = tempfile.mkdtemp()
        ckpt_file = os.path.join(ckpt_dir, 'checkpoint.h5')
        atexit.register(lambda: shutil.rmtree(ckpt_dir))

        # Horovod: save checkpoints only on the first worker to prevent other workers from corrupting them.
        if hvd.rank() == 0:
            callbacks.append(tf.keras.callbacks.ModelCheckpoint(ckpt_file, monitor='loss', mode='min',
                                                                save_best_only=True))

        # get input training data
        # local test here
        mnist_path = args.input_data
        pq_dataset = pq.ParquetDataset(mnist_path)
        # get all row groups
        row_groups = dataset_metadata.load_row_groups(pq_dataset)

        # get portion of data according to worker rank
        indices = arrange_piece_indices_for_worker(row_groups, hvd.rank(), hvd.size())
        gdf = cudf.read_parquet(mnist_path, row_groups=indices)
        # get feature and label data
        feature_gdf, label_gdf = split_feature_label(gdf, '784')
        # convert to TF Tensor
        train_feature_tensor_ = cudf_to_tensor(feature_gdf)
        train_label_tensor = cudf_to_tensor(label_gdf)

        # modify latter
        # steps_per_epoch = int(train_rows / args.batch_size / hvd.size()),
        # validation_steps = int(val_rows / args.batch_size / hvd.size()),
        history = model.fit(train_feature_tensor_, train_label_tensor,
                            callbacks=callbacks,
                            verbose=verbose,
                            epochs=100)

        globals()['_DATASET_FINALIZATION_HACK'] = model

        if hvd.rank() == 0:
            with open(ckpt_file, 'rb') as f:
                return history.history, f.read()


    spark = SparkSession.builder.appName('MNIST HVD TRAINING DEMO').getOrCreate()
    # Horovod: run training. Note: Modify num_proc to the number of GPUs(Spark executors)
    # 'env' is required here due to a bug in Horovod ,also described https://github.com/horovod/horovod/issues/3005
    # TODO: parameterized
    history, best_model_bytes = \
        horovod.spark.run(train_fn, args=(model_bytes,), num_proc=args.num_proc,
                          # env={'PATH':os.environ['PATH'], 'LD_LIBRARY_PATH':os.environ['LD_LIBRARY_PATH']},
                          stdout=sys.stdout, stderr=sys.stderr, verbose=2,
                          prefix_output_with_timestamp=True)[0]

    best_val_loss = min(history['loss'])
    print('Best loss: %f' % best_val_loss)

    # Write checkpoint.
    with open(args.model_output_path, 'wb') as f:
        f.write(best_model_bytes)
    print('Written checkpoint to %s' % args.model_output_path)

    spark.stop()
