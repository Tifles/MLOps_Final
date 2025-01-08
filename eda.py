import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_data(file_path: str = "./data/Credit_Default.csv") -> pd.DataFrame:
    """
    Загружаем изначальный датсет.

    Аргументы:
        file_path (str): путь до CSV-файа датасета

    Возвращаемые переменные:
        pd.DataFrame: загруженный датасет
    """
    with open(
            os.path.join(
                os.path.dirname(__file__), file_path
            ),
            "rb",
    ) as file:
        df = pd.read_csv(file)
    return df


def get_empty_values(df: pd.DataFrame) -> dict:
    """
    Анализ пропущенных значений.

    Аргументы:
        df (pd.DataFrame): Исходный датасет

    Возвращаемые переменные:
        dict: Словарь с результатами анализа пропущенных значений
    """
    missing_values = df.isnull().sum()
    missing_percentages = (missing_values / len(df)) * 100
    # Формирем график

    plt.figure(figsize=(10, 6))
    missing_percentages.plot(kind='bar')
    plt.title('Пропущено значений по столбцам')
    plt.ylabel('Процент пропущенных значений')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('./for_the_report/png/missing_values.png')
    plt.close()

    return {
        'missing_counts': missing_values.to_dict(),
        'missing_percentages': missing_percentages.to_dict()
    }


def get_pairplots(df: pd.DataFrame) -> None:
    """
    Построение диаграмм попарного распределения признаков.

    Аргументы:
        df (pd.DataFrame): Исходный датасет
    """
    numerical_cols = ["Income", "Age", "Loan", "Default"]

    plt.figure(figsize=(8, 6))
    sns.pairplot(
        df[numerical_cols],
        hue='Default',
        diag_kind='hist',
        palette="hls"
    )
    plt.savefig('./for_the_report/png/pairplot.png')
    plt.close()


def get_corr(df: pd.DataFrame) -> None:
    """
    Корреляционный анализ и визуализация.

    Аргументы:
        df (pd.DataFrame): Исходный датасет
    """
    # Список числовых признаков за исключением целевого
    numerical_cols = ["Income", "Loan", "Loan to Income"]
    corr_matrix = df[numerical_cols].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap='coolwarm',
        center=0,
        fmt='.2f')
    plt.title('Корреляция числовых признаков')
    plt.tight_layout()
    plt.savefig('./for_the_report/png/correlation_matrix.png')
    plt.close()


def get_class_balance(df: pd.DataFrame) -> dict:
    """
    Анализ баланса классов.

    Аргументы:
        df (pd.DataFrame): Исходный датасет

    Returns:
        dict: Словарь со статистикой по балансу классов
    """
    class_counts = df['Default'].value_counts()
    class_percentages = ((class_counts / len(df)) * 100).round(2)

    plt.figure(figsize=(8, 6))
    class_counts.plot(kind='bar')
    plt.title('Class Distribution')
    plt.xlabel('Default Status')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('./for_the_report/png/class_balance.png')
    plt.close()

    return {
        'counts': class_counts.to_dict(),
        'percentages': class_percentages.to_dict()
    }


def main():
    """Основная функция для запуска EDA."""
    # Загружаем датасет
    df = load_data()

    # Выводим общую информацию о датасете
    print("\nПредварительный обзор данных:\n")
    print("\nПервые пять строк:")
    print(df.head())
    print("\nРазмерность:", df.shape)
    print("\nНазвания признаков:", list(df.columns))
    print("\nТипы данных:\n", df.dtypes, sep='')

    # Анализ пропущенных значений
    missing_analysis = get_empty_values(df)
    print("\n\nАнализ пропущенных значений:\n")
    print("Пропущенно значений:", missing_analysis['missing_counts'])
    print(
        "Процент пропущенных значений:",
        missing_analysis['missing_percentages']
    )

    # Формируем графики длля визуализации
    get_pairplots(df)
    get_corr(df)
    class_balance = get_class_balance(df)

    # Анализ баланса классов
    print("\n\nАнализ баланса классов:\n")
    print("Class counts:", class_balance['counts'])
    print("Class percentages:", class_balance['percentages'])

    # Generate Quarto report
    generate_quarto_report(
        df,
        missing_analysis,
        class_balance,
    )


def generate_quarto_report(
    df: pd.DataFrame,
    missing_analysis: dict,
    class_balance: dict,
) -> None:
    """
    Формирование отчёта Quarto по результатам EDA.

    Аргументы:
        df (pd.DataFrame): Исходный датасет
        missing_analysis (dict): Результаты анализа пропущенных значений
        class_balance (dict): Статистик по балансу классов
    """
    if sum(missing_analysis['missing_counts'].values()) > 0:
        result = "\n"  # noqa: E501
    else:
        result = """В наборе данных не было обнаружено пропущенных значений.
Это упростит обчение модели, так как нам не нужно искать
способ их правильно обработать"""

    report_content = f"""---
title: "Отчет по результатам EDA на предложенном датасете"
---

## Предварительный обзор данных

В этом отчете представлены результаты разведочного анализа данных (EDA),
проведенного на датасете Credit Default.
В наборе данных содержится информация о клиентах,
включая их доход, возраст, сумму кредита и статус дефолта.

### Основные признаки
- Income — годовой доход клиента;
- Age — возраст клиента;
- Loan — сумма кредита, который был предоставлен клиенту;
- Loan to Income — отношение суммы кредита к доходу клиента;
- Default — бинарный таргет, указывающий на дефолт клиента (1 = дефолт,
0 = нет).

Несколько первых строк набора данных:
```python
{df.head()}
```

## Анализ пропущенных значений

```python
Число пропущенных значений: {missing_analysis['missing_counts']}
Процент пропущенных значений: {missing_analysis['missing_percentages']}
```
![Распределение пропущенных значений](./png/missing_values.png)
{result}

## Диаграммы попарного распределения признаков

![Попарные распределения признаков](./png/pairplot.png)

По данным графикам можно сделать следующие предположения:
1. Наибольшая вероятность дефолта среди людей:
- С доходом ниже 30000;
- Возрастом до 40 лет;
- Суммой кередита свыше 2500;
2. Масимальная сумма кредита (Loan) прямо зависит от годового дохода (Income);
3. С ростом дохода поднимается величина суммы кредита с минимальной
вероятностью дефолта.

## Корреляционный анализ

Корреляционная матрица, показывающая взаимосвязь между числовыми признаками:

![Корреляционная матрица](./png/correlation_matrix.png)

По полученной матрице заметна сильная корреляция между Loan и Loan to Income.

## Анализ баланса классов

```python
Число представителей каждого класса: {class_balance['counts']}
Процентное соотношение классов: {class_balance['percentages']}
```

![Баланс классов](./png/class_balance.png)

В наборе данных число попаших в дефолт почти в пять раз меньше числа
не попавших.

## Заключение и выводы

По итогам EDA можно сделать следующие выводы:

1. В исходном датасете отсутсвуют пропущенные значения, что упрощает
дальнейшее обучение модели;
2. Наибольшая вероятность дефолта среди людей c доходом менее 30000,
возрстом менее 40 лет и суммой кредита более 2500;
3. Масимальная сумма кредита (Loan) прямо зависит от годового дохода (Income);
4. Чем выше доход, тем выше минимальная сумма кредитаа без риска дефолта;
5.  В исходном датасете сильная корреляция между Loan и Loan to Income.
Это объясняется тем, что Loan to Income является произодной от Loan и Income.
Поэтому для обучения данную  характеристику нужно будет убрать, чтобы меньшить
линейную зависимость признаков.

"""

    try:
        with open(
            "./for_the_report/index.qmd",
            "w",
            encoding="utf-8"
        ) as f:
            f.write(report_content)
        print("Файл 'index.qmd' успешно сохранен")
    except Exception as e:
        print(f"Ошибка при сохранении файла: {e}")


if __name__ == "__main__":
    main()
