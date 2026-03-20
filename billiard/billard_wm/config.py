"""Constantes et paramètres par défaut."""

# --- Fenêtre ---
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 750
FPS = 60

# --- Table ---
TABLE_MARGIN = 80
TABLE_X = TABLE_MARGIN
TABLE_Y = TABLE_MARGIN
TABLE_WIDTH = WINDOW_WIDTH - 2 * TABLE_MARGIN
TABLE_HEIGHT = WINDOW_HEIGHT - 2 * TABLE_MARGIN - 60  # place pour UI en bas
BAND_THICKNESS = 12

# --- Physique ---
BALL_RADIUS = 10
FRICTION = 0.985  # coefficient multiplicatif par frame
RESTITUTION = 0.8  # élasticité des rebonds sur bandes
MAX_SPEED = 20.0
MIN_SPEED = 0.15  # en dessous, on arrête la bille
GRAVITY_SCALE = 0.06  # force de gravité par degré d'inclinaison
MAX_TILT = 15  # degrés max d'inclinaison

# --- World Model ---
WM_LEARNING_RATE = 0.12
WM_PREDICTION_STEPS = 600  # nb de steps pour la trajectoire prédite
WM_PREDICTION_DT = 1.0  # pas de temps pour la simulation du WM

# --- Couleurs ---
COLOR_BG = (30, 30, 35)
COLOR_TABLE = (0, 100, 50)
COLOR_BAND = (60, 30, 10)
COLOR_WHITE_BALL = (240, 240, 240)
COLOR_TARGET_BALL = (200, 40, 40)
COLOR_PREDICTED = (100, 180, 255, 120)  # bleu translucide
COLOR_ACTUAL = (255, 255, 100, 200)  # jaune
COLOR_TEXT = (220, 220, 220)
COLOR_ERROR = (255, 80, 80)
COLOR_SUCCESS = (80, 255, 120)
COLOR_TILT_ARROW = (255, 200, 50)
COLOR_UI_BG = (40, 40, 45)

# --- Étapes pédagogiques ---
TOP_K_COUNT = 5
CANDIDATES_REVEAL_RATE = 12  # ms entre chaque trajectoire révélée

# Durées des phases (ms) — ajustables pour répétition
PHASE_DURATION_IMAGINING = 3000
PHASE_DURATION_PLANNING = 1500
PHASE_DURATION_PREDICTING = 1000
PHASE_DURATION_OBSERVING = 1500
PHASE_DURATION_LEARNING = 1500

# Couleurs des étapes
COLOR_IMAGINATION = (180, 100, 255)   # violet
COLOR_PLANNING = (255, 165, 50)       # orange
COLOR_PREDICTION = (80, 160, 255)     # bleu vif
COLOR_ACTION = (255, 255, 100)        # jaune
COLOR_OBSERVATION = (255, 255, 255)   # blanc
COLOR_LEARNING = (80, 255, 120)       # vert
