# Данные

В соревновании используется один файл `data_set_1.csv` — выгрузка из CAEN CoMPASS
с записями сцинтилляционного детектора ГЕЛИС: 33 932 события × 496 АЦП-отсчётов
плюс метаданные `BOARD; CHANNEL; TIMETAG; ENERGY; ENERGYSHORT; FLAGS; SAMPLES`.

## Как получить

1. Зайти на [Kaggle: 2026-ml](https://www.kaggle.com/competitions/2026-ml/data).
2. Скачать `data_set_1.csv` (≈100 МБ) и положить в эту папку:
   ```
   data/data_set_1.csv
   ```

Через CLI:

```bash
pip install kaggle
kaggle competitions download -c 2026-ml -p data/
cd data && unzip 2026-ml.zip && cd ..
```

Файл не коммитится в репозиторий из соображений размера и правил Kaggle.
