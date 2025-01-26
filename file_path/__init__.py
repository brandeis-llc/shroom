from pathlib import Path

DATA_PATH = Path(__file__).parent.parent.joinpath("source_data")

SHROOM_TRAIN_PATH = DATA_PATH.joinpath("SHROOM_unlabeled-training-data-v2")
SHROOM_AW_TRAIN_PATH = SHROOM_TRAIN_PATH.joinpath("train.model-aware.v2.json")
SHROOM_AG_TRAIN_PATH = SHROOM_TRAIN_PATH.joinpath("train.model-agnostic.json")

SHROOM_DEV_PATH = DATA_PATH.joinpath("SHROOM_dev-v2")
SHROOM_AW_DEV_PATH = SHROOM_DEV_PATH.joinpath("val.model-aware.v2.json")
SHROOM_AG_DEV_PATH = SHROOM_DEV_PATH.joinpath("val.model-agnostic.json")

SHROOM_AG_DEV_AMR_PATH = SHROOM_DEV_PATH.joinpath("shroom_ag_dev.json")

# does not exist yet
# SHROOM_AW_DEV_AMR_PATH = SHROOM_DEV_PATH.joinpath("shroom_aw_dev.json")

OUT_FILE_PATH = DATA_PATH.joinpath("output")

SHROOM_TEST_PATH = DATA_PATH.joinpath("SHROOM_test-unlabeled")
SHROOM_AW_TEST_PATH = SHROOM_TEST_PATH.joinpath("test.model-aware.json")
SHROOM_AG_TEST_PATH = SHROOM_TEST_PATH.joinpath("test.model-agnostic.json")

SHROOM_TRIAL_PATH = DATA_PATH.joinpath("SHROOM_trial-v1.1/trial-v1.json")

SHROOM_TEST_LABELED_PATH = DATA_PATH.joinpath("SHROOM_test-labeled")
SHROOM_AG_TEST_LABELED_PATH = SHROOM_TEST_LABELED_PATH.joinpath("test.model-agnostic.json")
SHROOM_AW_TEST_LABELED_PATH = SHROOM_TEST_LABELED_PATH.joinpath("test.model-aware.json")
