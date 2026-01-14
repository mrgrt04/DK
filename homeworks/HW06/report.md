# HW06 – Report

## 1. Dataset

- Датасет выбран: `S06-hw-dataset-03.csv`
- Размер: 15000 строк, 30 столбцов
- Целевая переменная: `target` (3 класса с долями: класс 0 – ~0.543, класс 1 – ~0.302, класс 2 – ~0.155)
- Признаки: 28 вещественных признаков (`float64`) и 1 целочисленный признак (`int64`); все признаки числовые, что упрощает работу деревьев и ансамблей в мультиклассовой задаче

## 2. Protocol

- Разбиение: train/test с `test_size=0.25`, `random_state=42`, `stratify=y` (стратифицированное разбиение по классам)
- Подбор: `GridSearchCV` на train, 3 фолда `StratifiedKFold(n_splits=3, shuffle=True, random_state=42)`, оптимизируем `f1_macro` как более честную метрику для мультиклассовой задачи
- Метрики: accuracy, F1-macro, ROC-AUC (multi-class OVR) – accuracy показывает общую долю верных ответов, `f1_macro` усредняет качество по всем классам и чувствителен к минорным классам, multi-class ROC-AUC (OVR) отражает качество ранжирования вероятностей по всем классам

## 3. Models

- DummyClassifier (baseline по самой частой категории, стратегия `most_frequent`)
- LogisticRegression (baseline: `StandardScaler` + `LogisticRegression(max_iter=1000, random_state=42)`)
- DecisionTreeClassifier (контроль сложности: подбор `max_depth ∈ {3, 5, 8}`, `min_samples_leaf ∈ {1, 5}`, `ccp_alpha ∈ {0.0, 0.001}`)
- RandomForestClassifier (ансамбль деревьев: подбор `n_estimators=150`, `max_depth ∈ {None, 10}`, `min_samples_leaf ∈ {1, 5}`, `max_features=\"sqrt\"`)
- GradientBoostingClassifier (boosting: подбор `n_estimators=100`, `learning_rate ∈ {0.05, 0.1}`, `max_depth ∈ {2, 3}`)

## 4. Results

- Таблица финальных метрик на test (по данным `artifacts/metrics_test.json`):
  - DummyClassifier: accuracy ≈ 0.542, F1-macro ≈ 0.234, ROC-AUC = 0.500
  - LogisticRegression: accuracy ≈ 0.720, F1-macro ≈ 0.663, ROC-AUC ≈ 0.847
  - DecisionTreeClassifier: accuracy ≈ 0.777, F1-macro ≈ 0.713, ROC-AUC ≈ 0.864
  - RandomForestClassifier: accuracy ≈ 0.884, F1-macro ≈ 0.855, ROC-AUC ≈ 0.952
  - GradientBoostingClassifier: accuracy ≈ 0.828, F1-macro ≈ 0.788, ROC-AUC ≈ 0.927
- Победитель по согласованному критерию `f1_macro` (и по multi-class ROC-AUC): **RandomForestClassifier**. Он лучше остальных обрабатывает все три класса, особенно минорный, при этом показывает наибольший ROC-AUC.

## 5. Analysis

- Устойчивость: по файлу `artifacts/stability.json` видно, что ансамблевые модели (в частности RandomForest) демонстрируют меньший разброс F1 и ROC-AUC при смене `random_state`, чем одиночные модели, что особенно важно для мультиклассовой задачи.
- Ошибки: confusion matrix для лучшей модели (`figures/confusion_matrix.png`) показывает, что большинство ошибок связано с путаницей между соседними классами; минорный класс (2) распознаётся хуже, но всё равно существенно лучше, чем у baseline.
- Интерпретация: permutation importance (`figures/permutation_importance.png`) позволяет выделить 10–15 наиболее важных признаков; видно, что несколько признаков отвечают за разделение каждого из классов, что делает поведение ансамбля интерпретируемым даже в мультиклассовом случае.

## 6. Conclusion

- Для мультиклассовых задач важно использовать метрики, учитывающие качество по всем классам (`f1_macro`), а не только общую accuracy.
- Ансамблевые модели (RandomForest и GradientBoosting) дают более устойчивый и высокий результат по F1-macro и ROC-AUC, чем одиночные деревья и LogisticRegression.
- Честный протокол (фиксация train/test, `random_state=42`, CV на train) помогает избежать утечки в test и даёт корректное сравнение моделей.
- Перестановочная важность даёт наглядное представление о том, какие признаки важнее всего для разделения трёх классов, что облегчает интерпретацию сложных ансамблей.



