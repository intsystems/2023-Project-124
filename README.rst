|test| |codecov| |docs|

.. |test| image:: https://github.com/intsystems/ProjectTemplate/workflows/test/badge.svg
    :target: https://github.com/intsystems/ProjectTemplate/tree/master
    :alt: Test status
    
.. |codecov| image:: https://img.shields.io/codecov/c/github/intsystems/ProjectTemplate/master
    :target: https://app.codecov.io/gh/intsystems/ProjectTemplate
    :alt: Test coverage
    
.. |docs| image:: https://github.com/intsystems/ProjectTemplate/workflows/docs/badge.svg
    :target: https://intsystems.github.io/ProjectTemplate/
    :alt: Docs status


.. class:: center

    :Название исследуемой задачи: Ускорение семплирования из диффузионных моделей с помощью состязательных сетей
    :Тип научной работы: M1P
    :Автор: Никита Владимирович Охотников
    :Научный руководитель: кандидат ф-м. наук, Исаченко Роман Владимирович
    :Научный консультант(при наличии): степень, Фамилия Имя Отчество

Abstract
========

В последние годы широкое распространение получили диффузионные генеративные модели, показывающие высокое качество получаемых семплов и хорошее покрытие исходного распределения. Главный их недостаток -- скорость семлирования: для получения одного объекта требуется от сотен до тысяч итераций. Активно исследуются способы ускорения этого процесса. В работе анализируется один из таких способов -- использование состязательных моделей для сокращения числа шагов, необходимых для получения семпла. Экспериментально исследуются влияние гиперпараметров представленной ранее модели Denoising Diffusion GAN \cite{https://doi.org/10.48550/arxiv.2112.07804} на скорость генерации, а также качество и разнообразие получаемых семплов. Рассматриваются альтернативные варианты задания сложного распределения в обратном диффузионном процессе.

Research publications
===============================
1. 

Presentations at conferences on the topic of research
================================================
1. 

Software modules developed as part of the study
======================================================
1. A python package *mylib* with all implementation `here <https://github.com/intsystems/ProjectTemplate/tree/master/src>`_.
2. A code with all experiment visualisation `here <https://github.comintsystems/ProjectTemplate/blob/master/code/main.ipynb>`_. Can use `colab <http://colab.research.google.com/github/intsystems/ProjectTemplate/blob/master/code/main.ipynb>`_.
