(DRAFT version)

0. set up environment for Horovod
   ```shell
   //install cudnn and nccl
   sudo apt install libnccl2 libnccl-dev


   // install conda
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh
   source ~/.bashrc
   // install cudf, set those numbers accordingly
   conda install -c rapidsai -c nvidia -c numba -c conda-forge \
    cudf=21.06 python=3.8 cudatoolkit=11.2
   // install openmpi
   conda install openjdk=8 cmake openmpi openmpi-mpicc -y

   // install tensorflow
   pip install tensorflow

   // install horovod
   HOROVOD_WITH_MPI=1 HOROVOD_WITH_TENSORFLOW=1 HOROVOD_GPU_OPERATIONS=NCCL \
   pip install horovod[spark] --no-cache-dir

   // check if ok
   horovodrun --check-build
   ```
   should see:
   ```
   Horovod v0.22.1:

   Available Frameworks:
    [X] TensorFlow
    [X] PyTorch
    [ ] MXNet

   Available Controllers:
    [X] MPI
    [X] Gloo

   Available Tensor Operations:
    [X] NCCL
    [ ] DDL
    [ ] CCL
    [X] MPI
    [X] Gloo 
   ```

1. the training data could be downloaded from: https://drive.google.com/file/d/1lPBCTfUv1aSiWBGjT4WBv1zntNx0MAB2/view?usp=sharing, extract the content by
    ```
    tar -xvf train_100k_block_size.tar
    ```

2. Please set up your Spark cluster properly according to https://nvidia.github.io/spark-rapids/docs/get-started/getting-started-on-prem.html

3. Modify necessary parameters like `SPARK_HOME`, `SPARK_URL` etc. accordingly.

4. application specific parameters:
    1) `--num-proc` : The number of Spark Executors to run the application
    2) `--model-output-path`: the path to save best model
    3) `--input-data`: training dataset path

5. launch the app: 
    ```
    ./run.sh
    ```

Note: when using more than 1 executors, a known issue will be ovserved: https://github.com/horovod/horovod/issues/3005