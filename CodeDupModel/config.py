class Config:
    def __init__(self):
        # Model configuration
        self.model_name = "microsoft/codebert-base"
        self.dataset_name = "guychuk/code-duplicates-across-languages"
        self.data_dir = "data"
        self.hidden_size = 768  # Adjust based on your encoder's output features
        self.intermediate_size = int(self.hidden_size / 2)  # Size of the intermediate layer in the transformation
        self.dropout_rate = 0.1  # Dropout rate
        self.learning_rate = 5e-5  # Learning rate for the optimizer
        self.num_train_epochs = 3  # Number of training epochs
        self.max_grad_norm = 1.0  # Max gradient norm for clipping
        self.adam_epsilon = 1e-8  # Epsilon for Adam optimizer
        self.warmup_steps = 0

        # Training configuration
        self.batch_size = 16  # Batch size for training
        self.num_epochs = 3  # Number of training epochs
        self.alpha = 0.1  # Alpha value for the gradient reversal layer

        # TODO: Add more configurations:
        self.max_seq_length = 512  # Maximum sequence length for tokenization
        # self.domain_adaptation_weight = 0.5  # Weight for the domain adaptation loss component

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __repr__(self):
        return str(self.__dict__)

    def __str__(self):
        return str(self.__dict__)