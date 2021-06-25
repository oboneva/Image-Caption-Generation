class data_config:
    data_dir = "./Data"
    train_batch_size = 8
    test_batch_size = 256
    val_batch_size = 256
    num_workers = 6


class model_config:
    embed_size = 300
    hidden_size = 256
    attention_size = 256
    encoder_size = 2048
    dropout_prob = 0.3


class train_config:
    log_dir = "./runs"
    checkpoint_path = "./checkpoints"
    continue_training = False
    checkpoint_epochs = 5
    epochs = 100
