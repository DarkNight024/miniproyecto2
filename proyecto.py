import numpy as np
from gluoncv.model_zoo import get_model

# Crea o carga tu modelo aquí, por ejemplo:
net = get_model('cifar_resnet20_v1', classes=10, pretrained=True)

# Guarda los parámetros del modelo en un archivo llamado "net.params"
file_name = "net.params"
net.save_parameters(file_name)

print(f"Parámetros del modelo guardados en {file_name}")
