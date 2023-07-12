from dataclasses import dataclass


@dataclass
class TrainingConfig:
    debug = False  # whether to use a small dataset for debugging
    conditional = True  # whether to train a conditional model
    dataset = "mnist"  # the dataset to train on
    num_classes = 10  # the number of classes in the dataset
    channels = 1  # the number of input channels, 3 for RGB images
    image_size = 128  # the generated image resolution
    split_batches = False 
    train_batch_size = 16  # how many images per batch to use during training
    eval_batch_size = 16  # how many images to sample during evaluation
    num_train_timesteps = 1000
    num_eval_inference_steps = 1000
    eval_label = 2  # the label to see images of during evaluation
    num_epochs = 100
    gradient_accumulation_steps = 1
    learning_rate = 4e-4
    lr_warmup_steps = 500
    save_image_epochs = 1
    save_model_epochs = 10
    mixed_precision = "no"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "conditional-mnist"  # where to save the trained model

    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_private_repo = False
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0
