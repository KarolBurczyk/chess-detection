# Chess AI — live chess board detection

To repozytorium to aktywna wersja projektu do rozpoznawania szachownicy z kamery w czasie rzeczywistym. Główna logika działa w pakiecie `chess_ai` i opiera się na dwóch komponentach:

- detekcji planszy (krawędzie szachownicy + homografia 8x8),
- detekcji figur YOLO i przypisywaniu ich do konkretnych pól.

## Cel projektu

Celem jest zbudowanie prostego, działającego pipeline'a, który:

1. łapie obraz z kamery,
2. wykrywa krawędzie planszy,
3. mapuje obraz na układ 8x8,
4. wykrywa figury przy użyciu modelu YOLO,
5. przypisuje wykrycia do pól na szachownicy,
6. wyświetla wynik w oknie podglądu i pozwala dalej rozwijać logikę stanu gry.

## Aktualna architektura

- `chess_ai/` — aktywny kod główny projektu
  - `detector.py` — główna pętla aplikacji
  - `board_detector.py` — wykrywanie planszy i geometria szachownicy
  - `piece_detector.py` — klasy YOLO do detekcji figur
  - `board_state.py` — stan gry / logika planszy
  - `config.py` — konfiguracja modelu i źródła wejścia
- `chess_piece_prototype/` — prototyp treningu i przygotowania danych
- `dataset/` — gotowe zbiory do trenowania / walidacji
- `runs/` — output treningu i wytrenowane wagi
- `yolo11n.pt` — bazowy model YOLO
- `requirements.txt` — zależności projektu

## Jak uruchomić

### 1. Instalacja zależności

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 2. Uruchomienie detektora

Najprościej z katalogu głównego:

```bash
PYTHONPATH=. python3 -m chess_ai.detector
```

Lub bez pakietu z katalogu `chess_ai`:

```bash
cd chess_ai
python3 detector.py
```

Domyślne źródło kamery jest ustawione w `chess_ai/config.py` jako:

```python
SOURCE = 0
```

Jeśli chcesz użyć innego wejścia, zmień wartość `SOURCE` na ścieżkę do pliku wideo lub numer kamery.

## Co robi model

Projekt wykorzystuje model YOLO do detekcji obiektów na obrazie. Następnie:

- wyliczana jest homografia pomiędzy obrazem z kamery a układem planszy,
- punkty wykryć są mapowane na pola 8x8,
- obliczana jest orientacja planszy (np. `h1` jako lewy dolny narożnik),
- wykrycia są oznaczane na obrazie wraz z nazwą pola i confidence.

## Dane treningowe i retraining

Jeśli chcesz trenować albo przebudować model:

```bash
cd chess_piece_prototype
python3 prepare_dataset.py
python3 train_yolo.py --data data.yaml --epochs 100
```

Wartości i ścieżki do modelu są w `chess_ai/config.py`.

## Status projektu

Aktualnie projekt jest skupiony na:

- stabilnym wykrywaniu planszy,
- mapowaniu figur na pola,
- podglądzie w czasie rzeczywistym,
- dalszym rozwoju w kierunku pełnego stanu gry i logiki szachowej.

To jest wersja robocza i użyteczna do rozpoznawania pozycji z kamery, a nie finalna aplikacja do pełnej gry szachowej z analizą ruchów.

## Dalsze kroki

Najczęstsze następne etapy to:

1. stabilizacja detekcji przy różnych kątach kamery,
2. agregacja stanu na kolejnych klatkach,
3. porównywanie stanów i wyznaczanie ruchów,
4. integracja z `python-chess` do walidacji legalności pozycji i ruchów.
