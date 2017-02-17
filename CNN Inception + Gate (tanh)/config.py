GPU_ID = 10
BATCH_SIZE = 32 
VAL_BATCH_SIZE = 32
NUM_OUTPUT_UNITS = 3000 # This is the answer vocabulary size
MAX_WORDS_IN_QUESTION = 22 # Do not crop
MAX_ITERATIONS = 1000000
PRINT_INTERVAL = 1000
VALIDATE_INTERVAL = 110000 # We train on 'train' and test on 'val'. Set it to the number of iterations for training. Then the validation accuracy is the test accuracy.

# what data to use for training
TRAIN_DATA_SPLITS = 'train'

# what data to use for the vocabulary
QUESTION_VOCAB_SPACE = 'train'
ANSWER_VOCAB_SPACE = 'train'

# vqa tools - get from https://github.com/VT-vision-lab/VQA
VQA_TOOLS_PATH = '/tempspace/zwang6/VQA/PythonHelperTools'
VQA_EVAL_TOOLS_PATH = '/tempspace/zwang6/VQA/PythonEvaluationTools'

# location of the data
VQA_PREFIX = '/tempspace/zwang6/VQA/'
GENOME_PREFIX = '/tempspace/zwang6/vqa_mcb/genome/'
DATA_PREFIX = '/tempspace/zwang6/vqa_mcb/vqa-mcb/preprocess/'

DATA_PATHS = {
	'train': {
		'ques_file': VQA_PREFIX + '/Questions/OpenEnded_mscoco_train2014_questions.json',
		'ans_file': VQA_PREFIX + '/Annotations/mscoco_train2014_annotations.json',
		'features_prefix': DATA_PREFIX + '/image_features/resnet_res5c_bgrms_large/train2014/COCO_train2014_'
	},
	'val': {
		'ques_file': VQA_PREFIX + '/Questions/OpenEnded_mscoco_val2014_questions.json',
		'ans_file': VQA_PREFIX + '/Annotations/mscoco_val2014_annotations.json',
		'features_prefix': DATA_PREFIX + '/image_features/resnet_res5c_bgrms_large/val2014/COCO_val2014_'
	},
	'test-dev': {
		'ques_file': VQA_PREFIX + '/Questions/OpenEnded_mscoco_test-dev2015_questions.json',
		'features_prefix': DATA_PREFIX + '/image_features/resnet_res5c_bgrms_large/test2015/COCO_test2015_'
	},
	'test': {
		'ques_file': VQA_PREFIX + '/Questions/OpenEnded_mscoco_test2015_questions.json',
		'features_prefix': DATA_PREFIX + '/image_features/resnet_res5c_bgrms_large/test2015/COCO_test2015_'
	},
	# TODO it would be nice if genome also followed the same file format as vqa
	'genome': {
		'genome_file': GENOME_PREFIX + '/question_answers_prepro.json',
		'features_prefix': DATA_PREFIX + '/image_features/resnet_res5c_bgrms_large/selected/'
	}
}
