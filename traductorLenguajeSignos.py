import cv2
import mediapipe as mp

# Inicializar MediaPipe
mp_hands = mp.solutions.hands

# Crear un renderizador de dibujo
mp_drawing = mp.solutions.drawing_utils

# Crear una captura de video
cap = cv2.VideoCapture(0)

while True:
    # Leer un fotograma
    ret, frame = cap.read()

    # Convertir la imagen a RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detectar manos
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    results = hands.process(image)

    # Si se detectan manos
    if results.multi_hand_landmarks:
        # Dibujar marcas de las manos en la imagen
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Procesar las marcas de las manos para clasificar el gesto
        # Puedes agregar tu código de clasificación de gestos aquí

    # Mostrar la imagen
    cv2.imshow('Traducción de lenguaje de señas', frame)

    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
