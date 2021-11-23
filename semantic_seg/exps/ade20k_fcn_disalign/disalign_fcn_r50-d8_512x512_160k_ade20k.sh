export OMP_NUM_THREADS=1
GPU_NUM=8

WORKSPACE_PATH="${DISALIGN_SEG_HOME}/exps/ade20k_fcn_disalign/"
EXP_NAME="disalign_fcn_r50-d8_512x512_160k_ade20k"
# CONFIG_FILE=${MMSEG_HOME}/configs/ade20k_fcn_disalign/${EXP_NAME}.py
CONFIG_FILE=configs/ade20k_fcn_disalign/${EXP_NAME}.py
OUT_FOLDER=$WORKSPACE_PATH/$EXP_NAME/
PRETRAINED_CKPT_PATH=/data/mmseg_ckpts/model_zoo/fcn/ade20k/fcn_r50-d8_512x512_160k_ade20k_20200615_100713-4edbc3b4.pth

echo "workspace is:$WORKSPACE_PATH"
echo "config file is:$CONFIG_FILE"


mkdir $OUT_FOLDER

# Distributed Train
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} \
     --work-dir ${OUT_FOLDER}/${EXP_NAME} \
     --load-from ${PRETRAINED_CKPT_PATH} \
     --disalign

# Distributed Test
# ./tools/dist_test.sh ${CONFIG_FILE}  \
#      ${CKPT_PATH} \
#      ${GPU_NUM} --eval mIoU \
#      --options data.test.type=ADE20KLTDataset >> $OUT_FOLDER/${EXP_NAME}.log 2>&1 


# # Distributed Test with Augmentation
# ./tools/dist_test.sh ${CONFIG_FILE}  \
#      ${CKPT_PATH} \
#      ${GPU_NUM} --eval mIoU \
#      --options data.test.type=ADE20KLTDataset --aug-test >> $OUT_FOLDER/${EXP_NAME}_augtest.log 2>&1 
