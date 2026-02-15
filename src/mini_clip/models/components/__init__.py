# Импорты нужны для регистрации компонентов в registry.
from .image_encoders import ResNet50Encoder  # noqa: F401
from .text_encoders import DistilBertEncoder  # noqa: F401
from .projection import LinearProjection, MLPProjection  # noqa: F401
