from tensorflow.keras.models import Sequential, load_model


class Model:
    _instance = None

    def __init__(self) -> None:
        self.model: Sequential = load_model('src/models/store/model_emotion.h5')

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Model, cls).__new__(cls, *args, **kwargs)
        return cls._instance


model = Model().model
