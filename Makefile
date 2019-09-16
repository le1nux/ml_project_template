.PHONY: clean parse process featurize train

#### Define Variables based on Config
RAW_DATA_PATH := $(shell python make_utils/parse_make_variables.py --keys task_file_paths/raw_data_path)
PARSED_DATA_PATH := $(shell python make_utils/parse_make_variables.py --keys task_file_paths/parsed_data_path)
PROCESSED_DATA_PATH := $(shell python make_utils/parse_make_variables.py --keys task_file_paths/processed_data_path)
FEATURIZED_DATA_PATH := $(shell python make_utils/parse_make_variables.py --keys task_file_paths/featurized_data_path)

#### Create task specifig sub config files for build management
CREATE_TASK_CONFIGS := $(shell python make_utils/create_task_configs.py)

#### Define Meta Tasks with dynamic prerequisites
parse: $(PARSED_DATA_PATH) $(CREATE_TASK_CONFIGS)
process: $(PROCESSED_DATA_PATH) $(CREATE_TASK_CONFIGS)
featurize: $(FEATURIZED_DATA_PATH) $(CREATE_TASK_CONFIGS)
train: experiments $(CREATE_TASK_CONFIGS)

#### Execute Meta Tasks if necessary
$(PARSED_DATA_PATH): src/main/parse_data.py $(RAW_DATA_PATH) make_utils/parse_data.json
	@echo Parse data.
	@python -m src.main.parse_data config.json
$(PROCESSED_DATA_PATH): src/main/process_data.py $(PARSED_DATA_PATH) make_utils/process_data.json
	@echo Process data.
	@python -m src.main.process_data config.json
$(FEATURIZED_DATA_PATH): src/main/featurize_data.py $(PROCESSED_DATA_PATH) make_utils/featurize_data.json
	@echo Featurize data.
	@python -m src.main.featurize_data config.json
experiments: src/main/train.py $(FEATURIZED_DATA_PATH) make_utils/train.json
	@echo Train model.
	@python -m src.main.train with config.json

#### Clean project for fresh build
clean:
	rm -rf experiments/* data/parsed/* data/processed/* data/featurized/* make_utils/*.json
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete