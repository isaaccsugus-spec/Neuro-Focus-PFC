import customtkinter as ctk
import cv2
import pyautogui
import urllib.request
import os
import sys
import numpy as np
import math
import time
import threading
import subprocess
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import serial
import ctypes

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass


#AJUSTES DEL MOTOR DE CONDUCCIÓN (CALIBRACIÓN FINA)
SENSIBILIDAD_VUELO = 0.96 # suavizado del cursor
RADIO_PARADA = 80.0# distanacia eucliedea para parar el cursor
TIRON_ARRANQUE = 35.0 # fuerza necesaria para romper el paron
MULTIPLICADOR_X = 1.50 # ganancia de x e y
MULTIPLICADOR_Y = 1.65

# Configuración base de pyautogui
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0
screen_w, screen_h = pyautogui.size()

# Estética de la interfaz
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")
#--- LÓGICA DE DESCARGA CON RUTAS ABSOLUTAS ---
def obtener_ruta_base():
    """Devuelve la ruta exacta y real de la carpeta donde está el .exe"""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    else:
        return os.path.dirname(os.path.abspath(__file__))

URL_MODELO = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
NOMBRE_ARCHIVO = "face_landmarker.task"
# Forzamos la ruta absoluta combinando la carpeta real + el nombre del archivo
RUTA_MODELO_LOCAL = os.path.join(obtener_ruta_base(), NOMBRE_ARCHIVO)

def comprobar_modelo_ia(callback_exito, callback_error):
    """vereifica si existe el modelo de media pipe ya descargado"""
    if os.path.exists(RUTA_MODELO_LOCAL):
        callback_exito()
        return

    def hilo_descarga():
        try:
            print("Descargando modelo de Inteligencia Artificial...")
            urllib.request.urlretrieve(URL_MODELO, RUTA_MODELO_LOCAL)
            print("¡Descarga completa con éxito!")
            callback_exito()
        except Exception as e:
            print(f"Error en la descarga: {e}")
            callback_error()

    threading.Thread(target=hilo_descarga, daemon=True).start()
# -------------------------------------------------------------

# ─────────────────────────────────────────────────────────────
# 1. CLASE CALIBRADOR
# ─────────────────────────────────────────────────────────────
class AdvancedCalibrator:
    """gestion del mapeo usnado las coordenadas del iris obtenidas por media pipe
    y la resolucion del monitor mediante el modelo de regresion por minimos cuadrados"""
    def __init__(self):
        self.is_calibrated = False
        self.calibrating_phase1 = False
        self.calibrating_phase2 = False
        self.eye_points = []
        #matriz de nueve puntos
        self.screen_targets = [
            [50, 50], [screen_w // 2, 50], [screen_w - 50, 50],
            [50, screen_h // 2], [screen_w // 2, screen_h // 2], [screen_w - 50, screen_h // 2],
            [50, screen_h - 50], [screen_w // 2, screen_h - 50], [screen_w - 50, screen_h - 50]
        ]
        self.idx = 0
        self.coef_x = None
        self.coef_y = None
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.pending_mouse_move = False
        self.target_mouse_x = 0
        self.target_mouse_y = 0
        self.sample_buffer = []
        self.SAMPLES_P1 = 30
        self.SAMPLES_P2 = 45
        self.collecting = False
        self.progress_bar = 0
        self.point_start_time = 0
        self.WAIT_TIME = 0.8

    def start_calibration(self):
        self.is_calibrated = False
        self.calibrating_phase1 = True
        self.calibrating_phase2 = False
        self.idx = 0
        self.eye_points = []
        self.sample_buffer = []
        self.collecting = True
        self.request_mouse_move(self.screen_targets[self.idx])

    def request_mouse_move(self, target):
        self.target_mouse_x = int(target[0])
        self.target_mouse_y = int(target[1])
        self.pending_mouse_move = True
        self.point_start_time = time.time()

    def collect_point(self, eye_x, eye_y):
        if time.time() - self.point_start_time < self.WAIT_TIME: return
        # filtro usando para meddianre la mediana eliminar valores que hayan sido generados por parpadeos
        if self.calibrating_phase1 and self.collecting:
            self.sample_buffer.append([eye_x, eye_y])
            self.progress_bar = int(len(self.sample_buffer) / self.SAMPLES_P1 * 100)
            if len(self.sample_buffer) >= self.SAMPLES_P1:
                self.eye_points.append([np.median([s[0] for s in self.sample_buffer]),
                                        np.median([s[1] for s in self.sample_buffer])])
                self.sample_buffer = []
                self.progress_bar = 0
                self.idx += 1
                if self.idx >= 9:
                    self.finish_phase1()
                else:
                    self.request_mouse_move(self.screen_targets[self.idx])
        elif self.calibrating_phase2:
            self.sample_buffer.append([eye_x, eye_y])
            self.progress_bar = int(len(self.sample_buffer) / self.SAMPLES_P2 * 100)
            if len(self.sample_buffer) >= self.SAMPLES_P2:
                self._finish_phase2()

    def finish_phase1(self):
        """Primera fase del calibrado calculando la matriz de trasformacion
        mediante el uso del metodo de los minimos cuadrados"""
        self.calibrating_phase1 = False
        #conversion de las muestras a arrays de Numpy para el calculo matematico
        pts_eye = np.array(self.eye_points)
        pts_screen = np.array(self.screen_targets)
        ex, ey = pts_eye[:, 0], pts_eye[:, 1]
        # constucion de la matriz(A) ajuste polinomico
        # para modelar las no linealidades del seguimiento del ojos
        #se usan las variables ex, ey, el término cruzado (ex*ey) y el sesgo (bias) que son:
        # ex: coordenada x del ojo ey: coordenada y del ojo
        # bias: término constante para permitir traslaciones
        # termino curzado: para capturar interacciones entre ex e ey, como cambios en la forma del ojo que afectan ambas coordenadas
        A = np.column_stack([ex, ey, ex * ey, np.ones(9)])
        # Resolución del sistema de ecuaciones sobredeterminado A*coef = target_screen.
        # Se emplea el método de mínimos cuadrados (least squares) para minimizar el
        # error cuadrático medio entre la posición predicha y el objetivo real.
        self.coef_x, _, _, _ = np.linalg.lstsq(A, pts_screen[:, 0], rcond=None)
        self.coef_y, _, _, _ = np.linalg.lstsq(A, pts_screen[:, 1], rcond=None)
        # Transición a la fase 2 para el ajuste de precisión central (offset final)
        self.calibrating_phase2 = True
        self.sample_buffer = []
        self.progress_bar = 0
        self.request_mouse_move([screen_w // 2, screen_h // 2])

    def _predict_raw(self, ex, ey):
        row = np.array([ex, ey, ex * ey, 1.0])
        sx = float(np.dot(self.coef_x, row))
        sy = float(np.dot(self.coef_y, row))
        return sx, sy

    def _finish_phase2(self):
        """offset final para medinate la mediana coregir los desvios generados en la fase 1 """
        arr = np.array(self.sample_buffer)
        med_ex = float(np.median(arr[:, 0]))
        med_ey = float(np.median(arr[:, 1]))
        pred_x, pred_y = self._predict_raw(med_ex, med_ey)
        self.offset_x = (screen_w / 2) - pred_x
        self.offset_y = (screen_h / 2) - pred_y
        self.calibrating_phase2 = False
        self.is_calibrated = True
        self.collecting = False
        self.sample_buffer = []
        try:
            cv2.destroyWindow('Calibracion')
        except:
            pass

    def map_to_screen(self, eye_x, eye_y):
        # si no existe una calibracion va al centro de la pantalla
        if not self.is_calibrated: return screen_w / 2, screen_h / 2
        # obtencion de las cordenadas crudas
        sx, sy = self._predict_raw(eye_x, eye_y)
        # aplicacion de los valores de correcion llamados offsets
        sx += self.offset_x
        sy += self.offset_y
        # calculo del centro de la pantalla
        cx, cy = screen_w / 2, screen_h / 2
        # escalado de la sensibilidas del centro al borde
        final_x = cx + (sx - cx) * MULTIPLICADOR_X
        final_y = cy + (sy - cy) * MULTIPLICADOR_Y
        # delimitado de bordes
        return max(0, min(screen_w - 1, final_x)), max(0, min(screen_h - 1, final_y))

    def draw_calibration_screen(self):
        """funcion encargada de dibujar la interfaz de calibracion con las instrucciones y el punto objetivo"""
        cal = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        is_waiting = (time.time() - self.point_start_time) < self.WAIT_TIME
        if self.calibrating_phase1:
            tx, ty = self.screen_targets[self.idx]
            color_dot = (0, 0, 255) if is_waiting else (0, 255, 0)
            msg = "Enfoca el punto..." if is_waiting else "Capturando, no te muevas"
            cv2.putText(cal, f"Punto {self.idx + 1} / 9 - {msg}", (screen_w // 2 - 300, screen_h - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_dot, 2)
            cv2.circle(cal, (tx, ty), 30, color_dot, 2)
            cv2.circle(cal, (tx, ty), 6, color_dot, -1)
            bw = 400;
            bx = screen_w // 2 - bw // 2;
            by = screen_h - 100
            cv2.rectangle(cal, (bx, by), (bx + bw, by + 10), (40, 40, 40), -1)
            if not is_waiting: cv2.rectangle(cal, (bx, by), (bx + int(bw * self.progress_bar / 100), by + 10),
                                             (0, 200, 150), -1)
        elif self.calibrating_phase2:
            tx, ty = screen_w // 2, screen_h // 2
            color_dot = (0, 0, 255) if is_waiting else (0, 165, 255)
            msg = "Mira al centro..." if is_waiting else "Ajustando precision final..."
            cv2.circle(cal, (tx, ty), 35, color_dot, 2)
            cv2.circle(cal, (tx, ty), 8, color_dot, -1)
            cv2.putText(cal, f"FASE 2 - {msg}", (screen_w // 2 - 250, screen_h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        color_dot, 2)
            bw = 400;
            bx = screen_w // 2 - bw // 2;
            by = screen_h - 100
            cv2.rectangle(cal, (bx, by), (bx + bw, by + 10), (40, 40, 40), -1)
            if not is_waiting: cv2.rectangle(cal, (bx, by), (bx + int(bw * self.progress_bar / 100), by + 10),
                                             (0, 100, 255), -1)

        cv2.namedWindow('Calibracion', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('Calibracion', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setWindowProperty('Calibracion', cv2.WND_PROP_TOPMOST, 1)
        cv2.imshow('Calibracion', cal)


# ─────────────────────────────────────────────────────────────
# 2. MOTOR DE SEGUIMIENTO EN SEGUNDO PLANO (MOVIMIENTO VISUAL)
# ─────────────────────────────────────────────────────────────
class VisionThread(threading.Thread):
    """hilo asincrono que se encgara de obtener los fotogramas y procesarlos usnado media pipe"""
    def __init__(self, camera_index):
        super().__init__()
        self.camera_index = camera_index
        self.running = True
        self.trigger_calib = False
        self.latest_frame = None
        self.current_state = "DESACTIVADO"
        self.pos_x = 0
        self.pos_y = 0

    def run(self):
        calibrator = AdvancedCalibrator()
        # inicio del filtro 4 variables dinamicas y 2 de medicion (x, y, dx, dy) e (x, y)
        kalman = cv2.KalmanFilter(4, 2)
        # matriz de medcion(H) releciona el estado interno con la medicion
        kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        # matris de transicion (f) es el modelo fisico funcinamiento = posicion actual + velocidad
        kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        # matriz de covarianza / ruido del proceso (Q) es la confianza que tenemos en el modelo fisico, a mayor valor mas lento se adapta a cambios rapidos
        kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.01
        # matriz de covarianza para el reido en la medicion (R) encargado del ajuste de tembores
        kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 15.0

        def reset_k(x, y):
            s = np.array([[x], [y], [0], [0]], dtype=np.float32)
            kalman.statePre = s.copy()
            kalman.statePost = s.copy()

        reset_k(screen_w / 2, screen_h / 2)

        # Al descargarse de forma inteligente, solo apuntamos al archivo local
        model_path = RUTA_MODELO_LOCAL
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
        detector = vision.FaceLandmarker.create_from_options(options)

        cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        prev_mouse_x = screen_w // 2
        prev_mouse_y = screen_h // 2
        # buffer para almacenar las cordenadas usandas luego en el roolback
        POSITION_HISTORY = []
        is_stopped = False

        BLINK_THRESHOLD_CLOSE = 0.009
        BLINK_WIDEN_THRESHOLD = 0.030
        blink_state = "NORMAL"
        EYE_HISTORY = []
        cursor_frozen = False
        tiempo_descongelacion = 0

        while self.running and cap.isOpened():
            success, frame = cap.read()
            if not success: continue

            if self.trigger_calib:
                cursor_frozen = calibrator.is_calibrated = False
                reset_k(screen_w / 2, screen_h / 2)
                calibrator.start_calibration()
                self.trigger_calib = False

            frame = cv2.resize(frame, (640, 480))
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            detection_result = detector.detect(mp_image)

            if detection_result.face_landmarks:
                lm = detection_result.face_landmarks[0]
                avg_iris_x = (lm[468].x + lm[473].x) / 2
                avg_iris_y = (lm[468].y + lm[473].y) / 2
                ref_x = (lm[133].x + lm[362].x) / 2
                ref_y = (lm[133].y + lm[362].y) / 2

                face_scale = max(1e-6, math.hypot(lm[263].x - lm[33].x, lm[263].y - lm[33].y))
                rel_eye_x = (avg_iris_x - ref_x) / face_scale
                rel_eye_y = (avg_iris_y - ref_y) / face_scale

                # --- LÓGICA DE CONGELACIÓN con calculo de la distancaia euclideana entre parpados
                eye_openness = math.hypot(lm[159].x - lm[145].x, lm[159].y - lm[145].y)
                EYE_HISTORY.append(eye_openness)
                if len(EYE_HISTORY) > 8: EYE_HISTORY.pop(0) # bufer ciruclar con algoridmo fifo en 8 muestras
                #obtencion de la media movil
                avg_openness = float(np.mean(EYE_HISTORY))
                # activacion por gestos oculares
                if calibrator.is_calibrated:
                    # si el ojo se entorna <threshold_close> o se abre mucho > <threshold> se activa
                    if blink_state == "NORMAL" and (
                            avg_openness < BLINK_THRESHOLD_CLOSE or avg_openness > BLINK_WIDEN_THRESHOLD):
                        blink_state = "DEFORMADO"
                        cursor_frozen = True
                        # temporizador de congelacion
                        tiempo_descongelacion = time.time() + 1.2

                        # --- LA MÁQUINA DEL TIEMPO (ROLLBACK) ---
                        # Si tenemos historial, recuperamos la posición de hace ~15 frames (0.5 segundos) almaceenada en e buffer FIFO
                        # Esto deshace la desviación causada por la ceja bajando antes de hacer clic.
                        if len(POSITION_HISTORY) >= 5:
                            safe_x, safe_y = POSITION_HISTORY[0]  # Cogemos el dato más antiguo
                            pyautogui.moveTo(safe_x, safe_y)  # Teletransporte del ratón
                            prev_mouse_x, prev_mouse_y = safe_x, safe_y

                            # Estabilizamos el predictor de Kalman para que no pegue tirones al despertar
                            medida = np.array([[safe_x], [safe_y]], dtype=np.float32)
                            kalman.correct(medida)
                            kalman.statePost = np.array([[safe_x], [safe_y], [0], [0]], dtype=np.float32)
                            kalman.statePre = kalman.statePost.copy()
                    # Gestion de estados del parpadeo encargado de liberar el cursor
                    elif blink_state == "DEFORMADO" and (BLINK_THRESHOLD_CLOSE < avg_openness < BLINK_WIDEN_THRESHOLD):
                        blink_state = "NORMAL"
                    #evaluado del bloqueo del cursor
                    if blink_state == "NORMAL" and time.time() > tiempo_descongelacion:
                        cursor_frozen = False
                    else:
                        cursor_frozen = True
                # ---------------------------------------------

                if calibrator.pending_mouse_move:
                    pyautogui.moveTo(calibrator.target_mouse_x, calibrator.target_mouse_y)
                    calibrator.pending_mouse_move = False
                #flujo de estimaciones y suavizado dinamico
                elif calibrator.is_calibrated and not cursor_frozen:
                    raw_x, raw_y = calibrator.map_to_screen(rel_eye_x, rel_eye_y)
                    medida = np.array([[raw_x], [raw_y]], dtype=np.float32)
                    #fase de correcion y predicion con el filtro de kalman
                    kalman.correct(medida)
                    pred = kalman.predict()
                    # Truncamiento geométrico (clipping) para acotar las coordenadas dentro de la resolución útil
                    pred_x = float(np.clip(pred[0][0], 0, screen_w - 1))
                    pred_y = float(np.clip(pred[1][0], 0, screen_h - 1))
                    # Cálculo de la distancia euclidiana del desplazamiento actual para modular la acelaracion del cursor
                    dist = math.hypot(pred_x - prev_mouse_x, pred_y - prev_mouse_y)

                    if is_stopped:
                        if dist > TIRON_ARRANQUE: is_stopped = False
                    else:
                        if dist < RADIO_PARADA: is_stopped = True

                    if is_stopped:
                        final_x, final_y = prev_mouse_x, prev_mouse_y
                    else:
                        #SUAVIZADO DEL MOVIMIENTO POR LAS VARIBLES INDICADAS AL PRINCIPIO DEL ARCHIVO
                        SMOOTH = SENSIBILIDAD_VUELO + (0.98 - SENSIBILIDAD_VUELO) * math.exp(-dist / 35.0)
                        final_x = prev_mouse_x * SMOOTH + pred_x * (1 - SMOOTH)
                        final_y = prev_mouse_y * SMOOTH + pred_y * (1 - SMOOTH)

                    pyautogui.moveTo(final_x, final_y)
                    prev_mouse_x, prev_mouse_y = final_x, final_y

                    # --- GUARDAMOS HISTORIAL PARA EL ROLLBACK (Max 15 frames) ---
                    POSITION_HISTORY.append((final_x, final_y))
                    if len(POSITION_HISTORY) > 15: POSITION_HISTORY.pop(0)

                    self.current_state = "BLOQUEADO" if cursor_frozen else ("PARADO" if is_stopped else "MOVIENDO")
                    self.pos_x, self.pos_y = int(final_x), int(final_y)

                if calibrator.calibrating_phase1 and not calibrator.pending_mouse_move:
                    calibrator.collect_point(rel_eye_x, rel_eye_y)
                elif calibrator.calibrating_phase2 and not calibrator.pending_mouse_move:
                    calibrator.collect_point(rel_eye_x, rel_eye_y)

                cv2.circle(rgb_frame, (int(avg_iris_x * w), int(avg_iris_y * h)), 5,
                           (0, 0, 255) if cursor_frozen else (0, 255, 0), -1)
                cv2.circle(rgb_frame, (int(ref_x * w), int(ref_y * h)), 3, (255, 0, 0), -1)

            if calibrator.calibrating_phase1 or calibrator.calibrating_phase2:
                calibrator.draw_calibration_screen()
                self.current_state = "CALIBRANDO"

            cv2.waitKey(1)
            img_pil = Image.fromarray(rgb_frame)
            self.latest_frame = img_pil

        cap.release()
        try:
            cv2.destroyAllWindows()
        except:
            pass


# ─────────────────────────────────────────────────────────────
# 3. HILO DE LA DIADEMA (CLIC POR BLUETOOTH)
# ─────────────────────────────────────────────────────────────
class DiademaThread(threading.Thread):
    """Este hilo es el encargado de la telemetiria con la diadema
    funciona de forma independiente para evitar congelar el sitema principal
    capturando las señales neuromusculares"""
    def __init__(self, puerto="COM10", baudios=57600):
        super().__init__()
        self.puerto = puerto
        self.baudios = baudios
        self.running = True
        self.diadema = None
        self.calidad_sq = 200 # valor base que idica que la calidad de la señal es mala o esta en desconexion

    def run(self):
        """bucle de decodificado de tramas conectado a la diadema en el COM10"""
        try:
            self.diadema = serial.Serial(self.puerto, self.baudios, timeout=1)
            print(f"Diadema conectada en {self.puerto}")
        except Exception as e:
            print(f"Error conectando diadema en {self.puerto}: {e}")
            return

        tiempo_ultimo_clic = 0
        tiempo_señal_perfecta = 0
        #oermanece a la esucha siempre que el puerto serie este abierto
        while self.running and self.diadema.is_open:
            try:
                # comprobar si hay datos de entrada en el puerto serie
                if self.diadema.in_waiting > 0:
                    # Buscar el inicio del paquete (0xAA 0xAA)
                    if self.diadema.read() == b'\xaa' and self.diadema.read() == b'\xaa':
                        payload_len = self.diadema.read()
                        if not payload_len: continue
                        p_len = ord(payload_len)
                        if p_len > 170: continue  # borrado de corruptos
                        # leer paquete de datos completo y el cheksum (1 byte) encargado de la validacion
                        payload = self.diadema.read(p_len)
                        self.diadema.read()  # checksum

                        i = 0
                        # analisis de cargaa en busca de los codigos de calidad de señal y parpadeo
                        while i < p_len:
                            code = payload[i]
                            if code == 0x02:  # Calidad de señal
                                sq = payload[i + 1]
                                self.calidad_sq = int(sq)
                                if sq == 0:  # si la señal es cero el contacto sensor usuario es perfecto
                                    tiempo_señal_perfecta = time.time()

                                # gatillo facil
                                if sq > 0 and (time.time() - tiempo_señal_perfecta < 3.0):
                                    # gestion del tiempo entre clics
                                    if time.time() - tiempo_ultimo_clic > 0.8:
                                        print(f"CLIC HACK (Señal: {sq})")
                                        # pitido
                                        try:
                                            import winsound
                                            winsound.Beep(1000, 150)
                                        except:
                                            pass
                                        # ejecucion del clic
                                        pyautogui.click()
                                        tiempo_ultimo_clic = time.time()
                                        tiempo_señal_perfecta = 0  # reinicio contador

                                i += 2
                            elif code == 0x16:  # Parpadeo oficial
                                blink = payload[i + 1]
                                # SENSIBILIDAD MÁXIMA
                                if blink > 15:
                                    if time.time() - tiempo_ultimo_clic > 0.8:
                                        print(f"CLIC OFICIAL (Fuerza: {blink})")
                                        try:
                                            import winsound
                                            winsound.Beep(1500, 150)
                                        except:
                                            pass
                                        pyautogui.click()
                                        tiempo_ultimo_clic = time.time()
                                i += 2
                                #datos irelevantes por lo que son ignorados
                            elif code in [0x04, 0x05]:
                                i += 2
                            elif code == 0x80:
                                i += 4
                            elif code == 0x83:
                                i += 25
                            else:
                                if code > 0x7F:
                                    i += 2 + payload[i + 1]
                                else:
                                    i += 2
            except Exception:
                pass
        #CIERRE SEGURO
        if self.diadema and self.diadema.is_open:
            self.diadema.close()
            print("Diadema desconectada correctamente.")


# ─────────────────────────────────────────────────────────────
# 4. INTERFAZ GRÁFICA PRINCIPAL
# ─────────────────────────────────────────────────────────────
class NeuroFocusUI(ctk.CTk):
    """CLASE PRINCIPAL hereda de CUSTOMTKINTER se encagra del control de los hilos y eventos  """
    def __init__(self):
        super().__init__()

        # Oculta la ventana nada más nacer para evitar el salto visual
        self.withdraw()

        self.title("Neuro-Focus v4.4")
        self.minsize(800, 500)

        self.sistema_activo = False
        self.vision_thread = None
        self.diadema_thread = None

        self.blank_image = ctk.CTkImage(light_image=Image.new("RGB", (640, 480), (30, 30, 30)), size=(640, 480))
        self.current_image_ref = self.blank_image

        self.bind("<Escape>", self.emergencia_esc)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.lista_camaras = ["Buscando cámaras..."]

        self.sidebar = ctk.CTkFrame(self, width=220, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")

        ctk.CTkLabel(self.sidebar, text="NEURO-FOCUS", font=ctk.CTkFont(size=20, weight="bold")).grid(row=0, column=0,
                                                                                                      padx=20,
                                                                                                      pady=(20, 10))
        #barra lateral
        ctk.CTkLabel(self.sidebar, text="Seleccionar Cámara:", font=ctk.CTkFont(size=12)).grid(row=1, column=0, padx=20,
                                                                                               pady=(10, 0), sticky="w")

        self.cam_selector = ctk.CTkOptionMenu(self.sidebar, values=self.lista_camaras)
        self.cam_selector.grid(row=2, column=0, padx=20, pady=(5, 20))

        self.btn_iniciar = ctk.CTkButton(self.sidebar, text="▶ Iniciar Sistema", fg_color="green",
                                         hover_color="darkgreen", height=40, command=self.toggle_sistema)
        self.btn_iniciar.grid(row=3, column=0, padx=20, pady=10)

        self.btn_calibrar = ctk.CTkButton(self.sidebar, text="Calibrar Pantalla", height=40, state="disabled",
                                          command=self.lanzar_calibracion)
        self.btn_calibrar.grid(row=4, column=0, padx=20, pady=10)

        self.lbl_herramientas = ctk.CTkLabel(self.sidebar, text="--- HERRAMIENTAS ---", text_color="gray")
        self.lbl_herramientas.grid(row=6, column=0, padx=20, pady=(15, 5))

        self.btn_teclado = ctk.CTkButton(self.sidebar, text="⌨️ Teclado Virtual", fg_color="#4A4A4A",
                                         hover_color="#6B6B6B", command=self.abrir_teclado_os)
        self.btn_teclado.grid(row=7, column=0, padx=20, pady=5)

        self.btn_dictado = ctk.CTkButton(self.sidebar, text="🎤 Dictado por Voz", fg_color="#4A4A4A",
                                         hover_color="#6B6B6B", command=self.iniciar_dictado_os)
        self.btn_dictado.grid(row=8, column=0, padx=20, pady=5)
        # --- INDICADORES DE LA DIADEMA ---
        self.frame_telemetria = ctk.CTkFrame(self.sidebar)
        self.frame_telemetria.grid(row=9, column=0, padx=20, pady=10, sticky="ew")

        self.lbl_estado_diadema = ctk.CTkLabel(
            self.frame_telemetria,
            text="DIADEMA: Buscando...",
            font=("Arial", 14, "bold"),
            text_color="orange"
        )
        self.lbl_estado_diadema.pack(pady=5)

        self.lbl_calidad = ctk.CTkLabel(
            self.frame_telemetria,
            text="Calidad de Señal: --",
            font=("Arial", 12)
        )
        self.lbl_calidad.pack(pady=5)

        # Lanzamos el motor de refresco en cuanto se crea la ventana
        self.bucle_actualizacion_diadema()


        self.sidebar.grid_rowconfigure(10, weight=1)

        self.sidebar.grid_rowconfigure(9, weight=1)

        self.center = ctk.CTkFrame(self, fg_color="transparent")
        self.center.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.center.grid_rowconfigure(0, weight=3)
        self.center.grid_columnconfigure(0, weight=1)

        self.visor = ctk.CTkFrame(self.center)
        self.visor.grid(row=0, column=0, sticky="nsew")

        self.lbl_estado = ctk.CTkLabel(self.visor, text="ESTADO: DESACTIVADO", font=ctk.CTkFont(size=18, weight="bold"))
        self.lbl_estado.pack(side="bottom", pady=20)

        self.video_label = ctk.CTkLabel(self.visor, text="[ VIDEO DESACTIVADO ]", image=self.blank_image,
                                        text_color="gray", compound="center")
        self.video_label.pack(side="top", expand=True)

        self.after(200, self.detectar_camaras_rapido)
        self.update_gui_loop()
        self.mostrar_bienvenida()

        #Muestra la ventana de golpe y ya maximizada
        def revelar_ventana():
            self.state('zoomed')
            self.deiconify()

        self.after(10, revelar_ventana)

    def abrir_teclado_os(self):
        """funcion para la apertura del teclado nativo de windows """
        print("Intentando abrir el teclado clásico (OSK)...")
        try:
            subprocess.Popen(['cmd', '/c', 'start', 'osk'])
        except Exception as e:
            print(f"Fallo al usar CMD: {e}")
            try:
                os.system("osk")
            except Exception as e_final:
                print(f"Fallo total al intentar abrir el teclado: {e_final}")

    def iniciar_dictado_os(self):
        self.btn_dictado.configure(state="disabled", fg_color="#E6AA68", text_color="black")
        self.cuenta_atras_dictado(12)

    def cuenta_atras_dictado(self, segundos):
        #temporizador de apertura del dictado permite que el usuario llegue al compo de texto
        if segundos > 0:
            self.btn_dictado.configure(text=f"Haz clic en el texto... {segundos}s")
            self.after(1000, self.cuenta_atras_dictado, segundos - 1)
        else:
            self.ejecutar_dictado_real()

    def ejecutar_dictado_real(self):
        try:
            pyautogui.hotkey('win', 'h')
        except Exception as e:
            print(f"Error al iniciar dictado: {e}")
        finally:
            self.btn_dictado.configure(state="normal", text="🎤 Dictado por Voz", fg_color="#4A4A4A",
                                       text_color="#DCE4EE")

    def mostrar_bienvenida(self):
        """mensaje de bienvenida donde se muestran los consejos etc"""
        self.sidebar.grid_remove()
        self.center.grid_remove()

        self.welcome_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.welcome_frame.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=20, pady=10)

        ctk.CTkLabel(self.welcome_frame, text="Bienvenido a Neuro-Focus", font=ctk.CTkFont(size=32, weight="bold"),
                     text_color="#E6AA68").pack(pady=(5, 5))
        ctk.CTkLabel(self.welcome_frame, text="Instrucciones de uso y consejos.", font=ctk.CTkFont(size=14),
                     text_color="gray").pack(pady=(0, 10))

        instrucciones_frame = ctk.CTkFrame(self.welcome_frame, fg_color="#2B2B2B", corner_radius=10)
        instrucciones_frame.pack(fill="x", padx=40, pady=5)

        t1 = "1. ENTORNO Y POSTURA\nAsegúrese de estar centrado frente a la cámara web con el rostro bien iluminado. Evite focos de luz intensa o ventanas directamente a su espalda (contraluz) para que la Inteligencia Artificial detecte sus facciones correctamente."
        ctk.CTkLabel(instrucciones_frame, text=t1, justify="left", font=ctk.CTkFont(size=14), wraplength=700).pack(
            anchor="w", padx=20, pady=(10, 5))

        t2 = "2. CALIBRACIÓN INICIAL (CRÍTICA)\nPulse 'Iniciar Sistema' y luego 'Calibrar Pantalla'. Aparecerá un punto verde recorriendo los bordes del monitor. Sígalo ÚNICAMENTE con la mirada. Es de vital importancia mantener la cabeza completamente quieta durante este breve proceso."
        ctk.CTkLabel(instrucciones_frame, text=t2, justify="left", font=ctk.CTkFont(size=14), wraplength=700).pack(
            anchor="w", padx=20, pady=5)

        t3 = "3. CONTROL DEL CURSOR Y CLIC\nEl cursor se moverá hacia donde dirija su mirada. Si desea leer sin que el ratón se desplace (Freno de Mano), entorne los ojos levemente. Para ejecutar un clic izquierdo, levante las cejas de forma rápida o frunza el ceño. Un pitido sonoro confirmará que la orden ha sido recibida."
        ctk.CTkLabel(instrucciones_frame, text=t3, justify="left", font=ctk.CTkFont(size=14), wraplength=700).pack(
            anchor="w", padx=20, pady=(5, 10))

        ctk.CTkLabel(self.welcome_frame,
                     text="MECANISMO DE EMERGENCIA: Pulse la tecla ESC en su teclado en cualquier momento para detener la aplicación de forma segura.",
                     text_color="#FF6B6B", font=ctk.CTkFont(weight="bold")).pack(pady=10)

        self.btn_aceptar = ctk.CTkButton(self.welcome_frame, text="Entendido, Comencemos", height=40,
                                         font=ctk.CTkFont(size=16, weight="bold"), command=self.iniciar_app_real)
        self.btn_aceptar.pack(pady=10)

    #Métodos para manejar la comprobación y descarga asíncrona de la IA
    def iniciar_app_real(self):
        # Etiqueta temporal de carga
        self.lbl_cargando = ctk.CTkLabel(self.welcome_frame,
                                         text="Verificando componentes de Inteligencia Artificial... Por favor, espere.",
                                         font=ctk.CTkFont(weight="bold"), text_color="#E6AA68")
        self.lbl_cargando.pack(pady=10)

        # Bloqueamos el botón
        self.btn_aceptar.configure(state="disabled", text="Procesando...")

        # Lanzamos comprobación
        comprobar_modelo_ia(self.descarga_exitosa, self.descarga_fallida)

    def descarga_exitosa(self):
        self.after(0, self._finalizar_entrada)

    def _finalizar_entrada(self):
        self.welcome_frame.destroy()
        self.sidebar.grid()
        self.center.grid()

    def descarga_fallida(self):
        def _mostrar_error():
            self.lbl_cargando.configure(
                text="Error al descargar el modelo de IA. Verifique su conexión a Internet y reinicie.",
                text_color="#FF6B6B")
            self.btn_aceptar.configure(state="normal", text="Reintentar")

        self.after(0, _mostrar_error)



    def detectar_camaras_rapido(self):
        """funcion pensada para la detecion de hasta 5 camaras conectadas a un mismo equipo
        permite selecionaar la mejor se ha puesto el limite de 5 por asegurar que van a ser detectadas suficientes camaras
        realmente con dos valdria para cambiar eso solo ha que cambiar el range de 5 a 2 """
        camaras = []
        for i in range(5):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                camaras.append(f"Cámara {i}")
                cap.release()

        nueva_lista = camaras if camaras else ["No se detectan cámaras"]
        self.cam_selector.configure(values=nueva_lista)
        self.cam_selector.set(nueva_lista[0])

    def toggle_sistema(self):
        """enecndido del sistema y pausa inicia todo lso componentes """
        if not self.sistema_activo:
            seleccion = self.cam_selector.get()
            if "No se detectan" in seleccion or "Buscando" in seleccion: return
            id_cam = int(seleccion.split(" ")[1])

            self.sistema_activo = True
            self.btn_iniciar.configure(text="■ Detener Sistema", fg_color="red", hover_color="darkred")
            self.btn_calibrar.configure(state="normal")
            self.cam_selector.configure(state="disabled")

            # Arrancar el hilo visual
            self.vision_thread = VisionThread(camera_index=id_cam)
            self.vision_thread.start()

            # Arrancar hilo de la diadema
            self.diadema_thread = DiademaThread(puerto="COM10")
            self.diadema_thread.start()
        else:
            self.sistema_activo = False
            self.btn_iniciar.configure(text="▶ Iniciar Sistema", fg_color="green", hover_color="darkgreen")
            self.btn_calibrar.configure(state="disabled")
            self.cam_selector.configure(state="normal")

            self.current_image_ref = self.blank_image
            self.video_label.configure(image=self.blank_image, text="[ VIDEO DESACTIVADO ]")
            self.lbl_estado.configure(text="ESTADO: DESACTIVADO", text_color="gray")

            if self.vision_thread:
                self.vision_thread.running = False
                self.vision_thread = None

            if hasattr(self, 'diadema_thread') and self.diadema_thread:
                self.diadema_thread.running = False
                self.diadema_thread = None

    def lanzar_calibracion(self):
        if self.vision_thread:
            self.vision_thread.trigger_calib = True

    def update_gui_loop(self):
        """bucle que se encagrda de refrescar los datos del hilo visual"""
        if self.sistema_activo and self.vision_thread:
            img_pil = self.vision_thread.latest_frame
            if img_pil:
                self.current_image_ref = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=(640, 480))
                self.video_label.configure(image=self.current_image_ref, text="")

            estado = self.vision_thread.current_state
            x, y = self.vision_thread.pos_x, self.vision_thread.pos_y

            color_txt = "#00FF00" if estado == "MOVIENDO" else "#FF0000" if estado == "BLOQUEADO" else "#FFA500" if estado == "CALIBRANDO" else "#FFFF00"
            self.lbl_estado.configure(text=f"ESTADO: [{estado}]  |  Pos: X:{x} Y:{y}", text_color=color_txt)

        self.after(30, self.update_gui_loop)

    def emergencia_esc(self, event):
        self.on_closing()

    def on_closing(self):
        """cierre de manera ordeanda para evitar errores en el cerrado de puertos
        evitando el lanzado de excepciones """
        if self.vision_thread:
            self.vision_thread.running = False
            self.vision_thread.join(timeout=1.0)

        if hasattr(self, 'diadema_thread') and self.diadema_thread:
            self.diadema_thread.running = False
            self.diadema_thread.join(timeout=1.0)

        self.destroy()

    def bucle_actualizacion_diadema(self):
        """Monitorea asíncronamente el estado del hilo de la diadema para la GUI"""
        if hasattr(self, 'diadema_thread') and self.diadema_thread and self.diadema_thread.is_alive():
            sq_actual = self.diadema_thread.calidad_sq

            if sq_actual == 0:
                self.lbl_estado_diadema.configure(text="DIADEMA: CONECTADA", text_color="#a3e635")  # Verde
                self.lbl_calidad.configure(text=f"Calidad de Señal: Excelente (sq: {sq_actual})", text_color="#a3e635")
            elif sq_actual == 200:
                self.lbl_estado_diadema.configure(text="DIADEMA: SIN CONTACTO", text_color="#ef4444")  # Rojo
                self.lbl_calidad.configure(text="Ajuste la diadema en la frente / oreja", text_color="#ef4444")
            else:
                self.lbl_estado_diadema.configure(text="DIADEMA: SEÑAL INESTABLE", text_color="#f59e0b")  # Naranja
                self.lbl_calidad.configure(text=f"Ruido detectado (sq: {sq_actual})", text_color="#f59e0b")
        else:
            self.lbl_estado_diadema.configure(text="DIADEMA: DESCONECTADA", text_color="#737373")  # Gris
            self.lbl_calidad.configure(text="Verifique el puerto COM10 y el Bluetooth", text_color="#737373")

        # Se vuelve a llamar a sí misma cada 200 milisegundos de forma segura
        self.after(200, self.bucle_actualizacion_diadema)

if __name__ == "__main__":
    app = NeuroFocusUI()
    app.mainloop()