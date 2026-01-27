# HW06 – Report

## 1. Dataset

- Какой датасет выбран: `S06-hw-dataset-01.csv`
- Размер: 12000 строк, 30 столбцов
- Целевая переменная: `target` (классы и их доли: класс 0 – ~0.677, класс 1 – ~0.323)
- Признаки: 24 вещественных признака (`float64`) и 5 целочисленных (`int64`), часть целочисленных признаков категориально-подобные (`cat_*`)

## 2. Protocol

- Разбиение: train/test с `test_size=0.25`, `random_state=42`, `stratify=y` (стратификация по `target` для сохранения баланса классов)
- Подбор: `GridSearchCV` на train, 3-фолдовый `StratifiedKFold(n_splits=3, shuffle=True, random_state=42)`, оптимизируем `roc_auc` как основную метрику для бинарной задачи
- Метрики: accuracy, F1, ROC-AUC – accuracy даёт долю верных ответов, F1 балансирует полноту и точность для минорного класса, ROC-AUC оценивает качество ранжирования по вероятностям и устойчивость к выбору порога, что особенно важно при умеренном дисбалансе

## 3. Models

- DummyClassifier (baseline по самой частой категории, стратегия `most_frequent`)
- LogisticRegression (baseline из S05: `StandardScaler` + `LogisticRegression(max_iter=1000, random_state=42)`)
- DecisionTreeClassifier (контроль сложности: подбор `max_depth ∈ {3, 5, 8}`, `min_samples_leaf ∈ {1, 5}`, `ccp_alpha ∈ {0.0, 0.001}`)
- RandomForestClassifier (ансамбль деревьев: подбор `n_estimators=150`, `max_depth ∈ {None, 10}`, `min_samples_leaf ∈ {1, 5}`, `max_features=\"sqrt\"`)
- GradientBoostingClassifier (boosting: подбор `n_estimators=100`, `learning_rate ∈ {0.05, 0.1}`, `max_depth ∈ {2, 3}`)

## 4. Results

- Таблица финальных метрик на test (по данным `artifacts/metrics_test.json`):
  - DummyClassifier: accuracy ≈ 0.677, F1 ≈ 0.000, ROC-AUC = 0.500
  - LogisticRegression: accuracy ≈ 0.830, F1 ≈ 0.715, ROC-AUC ≈ 0.879
  - DecisionTreeClassifier: accuracy ≈ 0.862, F1 ≈ 0.780, ROC-AUC ≈ 0.895
  - RandomForestClassifier: accuracy ≈ 0.931, F1 ≈ 0.889, ROC-AUC ≈ 0.970
  - GradientBoostingClassifier: accuracy ≈ 0.906, F1 ≈ 0.845, ROC-AUC ≈ 0.958
- Победитель по ROC-AUC: **RandomForestClassifier** (ROC-AUC ≈ 0.970, F1 ≈ 0.889, accuracy ≈ 0.931). Лес существенно опережает одиночные модели и логистическую регрессию как по AUC, так и по F1, при этом остаётся устойчивым к дисбалансу.

## 5. Analysis

- Устойчивость: по результатам файла `artifacts/stability.json` (несколько `random_state` для LogisticRegression и RandomForest) видно, что случайный лес демонстрирует меньший разброс метрик на разных разбиениях, чем одиночные модели.
- Ошибки: confusion matrix для лучшей модели (`artifacts/figures/confusion_matrix.png`) показывает, что RandomForest хорошо распознаёт минорный класс, а основные ошибки связаны с небольшим количеством ложных отрицаний.
- Интерпретация: permutation importance (top-10/15 признаков, `artifacts/figures/permutation_importance.png`) показывает, что несколько числовых признаков (`num0X`) и отдельные категориально-подобные признаки (`cat_*`) дают основной вклад в предсказание оттока, что согласуется с интуитивным ожиданием по данным.

## 6. Conclusion

- Дерево решений без ограничения сложности легко переобучается, поэтому подбор `max_depth`, `min_samples_leaf` и `ccp_alpha` критичен для честной оценки.
- Ансамбли (Random Forest и Gradient Boosting) дают заметно более высокое качество по ROC-AUC и F1 по сравнению с Dummy, LogisticRegression и одиночным деревом.
- Честный протокол (фиксированный train/test с `random_state=42` + CV на train) позволяет корректно сравнивать модели и не «подглядывать» в test.
- Перестановочная важность помогает понять, какие признаки действительно влияют на предсказание и как они соотносятся с предметной областью и бизнес-логикой задачи.


