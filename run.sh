SPARK_HOME=/home/allxu/Desktop/spark-3.1.2-bin-hadoop3.2
SPARK_URL=spark://allxu-home:7077

$SPARK_HOME/bin/spark-submit --master $SPARK_URL --deploy-mode client  \
--driver-memory 10G \
--executor-memory 10G \
--executor-cores 12 \
--conf spark.cores.max=12 \
--conf spark.task.cpus=12 \
--conf spark.locality.wait=0 \
--conf spark.yarn.maxAppAttempts=1 \
--conf spark.sql.shuffle.partitions=200 \
--conf spark.sql.files.maxPartitionBytes=1024m \
--conf spark.sql.warehouse.dir=$OUT \
--conf spark.task.resource.gpu.amount=0.08 \
--conf spark.executor.resource.gpu.amount=1 \
--conf spark.executor.resource.gpu.discoveryScript=./getGpusResources.sh \
--files $SPARK_HOME/examples/src/main/scripts/getGpusResources.sh \
./main.py --num-proc 1 --model-output-path . --input-data $PWD/train_100k_block_size

