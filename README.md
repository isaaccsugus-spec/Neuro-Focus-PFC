# NEURO FOCUS
> **Proyecto Final de Ciclo (PFC) - Grado Superior en Desarrollo de Aplicaciones Multiplataforma (DAM).**

> **Autor:** ISAAC Sánchez
> 
> **Tutor:** JUAN Bonnín
> 
![icono.ico](icono.ico)

NEURO-FOCUS es un sistema híbrido y *Low-Cost* de accesibilidad universal diseñado para permitir el control total del cursor del sistema operativo a personas con movilidad reducida severa o diversidad funcional.

El proyecto fusiona Visión Artificial y Biometría mediante un enfoque de código abierto, eliminando la necesidad de hardware médico de coste prohibitivo.

## Características Principales / funcionalidades
* **Eye-Tracking Dinámico:** Seguimiento del iris en tiempo real utilizando MediaPipe Face Landmarker.
* **Calibración Matemática:** Mapeo de coordenadas mediante regresión por mínimos cuadrados adaptativa a cualquier resolución de pantalla.
* **Estabilización de Cursor:** Implementación nativa de un Filtro de Kalman para eliminar el ruido y los temblores oculares.
* **Clic Neuromuscular (EMG):** Integración por Bluetooth con la diadema NeuroSky MindWave Mobile 2 para la ejecución de clics a través de biometría facial, reduciendo la latencia a < 2ms.
* **Máquina del Tiempo (Rollback):** Sistema de búfer FIFO que compensa la desviación parasitaria del cursor producida por la contracción de la ceja al parpadear.

## Tecnologías y Librerías(para más detalle consultar el requirements.txt)
* **Lenguaje:** Python 3.x
* **Visión Artificial e IA:** OpenCV, MediaPipe
* **Interfaz Gráfica:** CustomTkinter
* **Interacción OS:** PyAutoGUI
* **Telemetría y Hardware:** PySerial, NumPy

## Instalación y Uso
1. Instala las dependencias necesarias: `pip install -r requirements.txt`
2. Enciende y empareja la diadema NeuroSky por Bluetooth.
3. Asegúrate de que la diadema está asignada al puerto COM10( detalles de la configuracion en el manual).
4. Ejecuta el sistema: `python final_neuro.py`



