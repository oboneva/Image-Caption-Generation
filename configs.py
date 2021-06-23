class data_config:
    data_dir = "./Data"
    train_batch_size = 256
    test_batch_size = 256
    val_batch_size = 256
    num_workers = 4


class model_config:
    embed_size = 500
    hidden_size = 50


class train_config:
    checkpoint_path = "./checkpoints"
    continue_training = False
    checkpoint_epochs = 5
    epochs = 100
